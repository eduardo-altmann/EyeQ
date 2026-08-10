import os
import argparse
import json
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from progress.bar import Bar
import torchvision.transforms as transforms
from dataloader.EyeQ_loader import DatasetGenerator
from utils.trainer import train_step, validation_step, save_output
from utils.metric import compute_metric

import pandas as pd
from networks.densenet_mcf import dense121_mcs

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Subset
from sklearn.model_selection import train_test_split


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def selection_value(metrics, validation_loss, name):
    if name == 'kappa':
        return float(metrics['Kappa'])
    if name == 'macro_f1':
        return float(metrics['macro-F1'])
    if name == 'auc':
        return float(np.mean(metrics['AUC']))
    return -float(validation_loss)

def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def main():
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    from torch.utils.tensorboard import SummaryWriter
    writer = None
    if local_rank == 0:
        writer = SummaryWriter()

    data_root = '../EyeQ_preprocess/'

    parser = argparse.ArgumentParser(description='EyeQ_dense121_tuned')
    parser.add_argument('--model_dir', type=str, default='./result/')
    parser.add_argument('--pre_model', type=str, default=None,
                        help='Set to model name (without .tar) to resume training, or None for fresh start')
    parser.add_argument('--save_model', type=str, default='DenseNet121_v3_tuned')
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--label_idx', type=list, default=['Good', 'Usable', 'Reject'])
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--epochs', default=60, type=int)
    parser.add_argument('--batch-size', default=4, type=int,
                        help='Increase from 4 to 16 for better gradient estimation')
    parser.add_argument('--batch_size_ref', type=int, default=4)
    parser.add_argument('--lr', default=0.004, type=float,
                        help='Lower LR since we use pretrained weights (was 0.01)')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='SGD momentum (was missing)')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='L2 regularization (was missing)')
    parser.add_argument('--loss_w', default=[0.1, 0.1, 0.1, 0.1, 0.6], type=list)
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'none'],
                        help='LR scheduler type')
    parser.add_argument('--lr_step_size', default=40, type=int,
                        help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='Gamma for StepLR scheduler')
    parser.add_argument('--warmup_epochs', default=5, type=int,
                        help='Number of warmup epochs with linear LR ramp')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use ImageNet pretrained DenseNet121 backbone')
    parser.add_argument('--seed', type=int, default=0,
                        help='Global RNG seed; vary it to report mean +/- std over runs')
    parser.add_argument('--tuning', action='store_true', default=False,
                        help='Tuning mode: select on a held-out validation split and '
                             'NEVER touch the test set (avoids test-set leakage during HPO)')
    parser.add_argument('--val_split', type=float, default=0.1,
                        help='Fraction of the training set held out (stratified) for validation')
    parser.add_argument('--selection_metric', type=str, default='kappa',
                        choices=['kappa', 'macro_f1', 'auc', 'loss'],
                        help='Metric used for best-epoch model selection and the Optuna objective')
    parser.add_argument('--lr_scaling', action='store_true', default=False,
                        help='Apply the linear LR scaling rule (Goyal et al. 2017). '
                             'Off by default so --lr IS the effective LR and the search space '
                             'is not silently coupled to batch size.')
    parser.add_argument('--sync_bn', action='store_true', default=False,
                        help='Convert BatchNorm to SyncBatchNorm (recommended for small per-GPU batches)')
    parser.add_argument('--class_weighted_loss', action='store_true', default=False,
                        help='Weight the fused BCE loss by inverse class frequency to help the '
                             'minority (Reject) grade')
    parser.add_argument('--progress_file', type=str, default=None,
                        help='If set, rank 0 appends per-epoch JSON lines here for Optuna pruning')

    args = parser.parse_args()

    set_seed(args.seed)

    train_images_dir = data_root + '/train'
    label_train_file = '../data/Label_EyeQ_train.filtered.csv'
    test_images_dir = data_root + '/test'
    label_test_file = '../data/Label_EyeQ_test.filtered.csv'

    save_file_name = args.model_dir + args.save_model + '.csv'

    best_score = -np.inf
    best_val_loss = np.inf
    best_iter = 0
    cudnn.benchmark = True

    model = dense121_mcs(n_class=args.n_classes, pretrained=args.pretrained)

    if args.pre_model is not None:
        loaded_model = torch.load(os.path.join(args.model_dir, args.pre_model + '.tar'))
        model.load_state_dict(loaded_model['state_dict'])

        if local_rank==0: print(f'Loaded pretrained model: {args.pre_model}')

    model.to(device)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    criterion = torch.nn.BCELoss(reduction='mean')
    num_gpus = dist.get_world_size()
    global_batch_size = args.batch_size * num_gpus
    if args.lr_scaling:
        effective_lr = args.lr * (global_batch_size / args.batch_size_ref)
    else:
        effective_lr = args.lr

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=effective_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True
    )

    if args.lr_scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epochs, eta_min=1e-6
        )
    elif args.lr_scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    else:
        scheduler = None


    def warmup_lr(optimizer, epoch, warmup_epochs, base_lr):
        lr = base_lr * (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    if local_rank==0:
        print('=' * 60)
        print('Tuned Training Configuration:')
        print(f'  Mode: {"TUNING (val-based, no test)" if args.tuning else "FINAL (test eval)"}')
        print(f'  Seed: {args.seed}')
        print(f'  Selection metric: {args.selection_metric}')
        print(f'  Pretrained backbone: {args.pretrained}')
        print(f'  Number of GPUs: {num_gpus}')
        print(f'  Batch size per GPU: {args.batch_size}')
        print(f'  Global batch size: {global_batch_size}')
        print(f'  LR scaling rule: {args.lr_scaling}')
        print(f'  Requested LR: {args.lr} | Effective LR: {effective_lr}')
        print(f'  Momentum: {args.momentum}')
        print(f'  Weight decay: {args.weight_decay}')
        print(f'  LR scheduler: {args.lr_scheduler}')
        print(f'  Warmup epochs: {args.warmup_epochs}')
        print(f'  Total epochs: {args.epochs}')
        print(f'  Class-weighted loss: {args.class_weighted_loss}')
        print(f'  Loss weights: {args.loss_w}')
        print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
        print('=' * 60)

    transform_list1 = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=(-180, +180)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    ])

    transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    transform_list_val1 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])

    data_train_aug = DatasetGenerator(data_dir=train_images_dir, list_file=label_train_file,
                                      transform1=transform_list1, transform2=transformList2,
                                      n_class=args.n_classes, set_name='train')
    data_train_plain = DatasetGenerator(data_dir=train_images_dir, list_file=label_train_file,
                                        transform1=transform_list_val1, transform2=transformList2,
                                        n_class=args.n_classes, set_name='val')

    labels_int = np.array([int(np.argmax(np.asarray(l))) for l in data_train_aug.labels])
    all_idx = np.arange(len(labels_int))
    train_idx, val_idx = train_test_split(
        all_idx, test_size=args.val_split, random_state=args.seed, stratify=labels_int
    )

    data_train = Subset(data_train_aug, train_idx)
    data_val = Subset(data_train_plain, val_idx)

    train_sampler = DistributedSampler(data_train, shuffle=True)
    train_loader = torch.utils.data.DataLoader(dataset=data_train,
                                               batch_size=args.batch_size,
                                               sampler=train_sampler,
                                               num_workers=4,
                                               pin_memory=True
                                               )

    val_sampler = DistributedSampler(data_val, shuffle=False, drop_last=False)
    val_loader = torch.utils.data.DataLoader(dataset=data_val, sampler=val_sampler,
                                             batch_size=args.batch_size, shuffle=False,
                                             num_workers=4, pin_memory=True)

    data_test = DatasetGenerator(data_dir=test_images_dir, list_file=label_test_file, transform1=transform_list_val1,
                                transform2=transformList2, n_class=args.n_classes, set_name='test')

    if args.class_weighted_loss:
        counts = np.bincount(labels_int[train_idx], minlength=args.n_classes).astype('float32')
        inv = counts.sum() / (counts + 1e-6)
        class_weights = torch.tensor(inv / inv.mean(), device=device, dtype=torch.float32)
        criterion = torch.nn.BCELoss(weight=class_weights, reduction='mean')
    else:
        class_weights = None


    if local_rank==0:
        print(f'\nTrain split: {len(data_train)} images | Val split: {len(data_val)} images')
        print(f'Test set (held out): {len(data_test)} images\n')

    dist.barrier()
    t0 = time.time()

    for epoch in range(0, args.epochs):
        train_sampler.set_epoch(epoch)
        
        if epoch < args.warmup_epochs:
            warmup_lr(optimizer, epoch, args.warmup_epochs, effective_lr)

        current_lr = optimizer.param_groups[0]['lr']
        if local_rank == 0:
            print(f'\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.6f}')

        train_loss = train_step(train_loader, model, epoch, optimizer, criterion, args, device, local_rank)
        validation_loss, val_predictions, val_labels = validation_step(val_loader, model, criterion, device, local_rank)

        val_loss_tensor = torch.tensor(validation_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        validation_loss = (val_loss_tensor / dist.get_world_size()).item()

        world_size = dist.get_world_size()
        predictions_list = [torch.empty_like(val_predictions) for _ in range(world_size)]
        labels_list = [torch.empty_like(val_labels) for _ in range(world_size)]
        dist.all_gather(predictions_list, val_predictions)
        dist.all_gather(labels_list, val_labels)

        if local_rank == 0:
            all_val_predictions = torch.cat(predictions_list, dim=0)
            all_val_labels = torch.cat(labels_list, dim=0)

            pred_np = all_val_predictions.cpu().numpy()
            labels_np = all_val_labels.cpu().numpy()

            label_list = args.label_idx
            val_metrics = compute_metric(np.argmax(labels_np, axis=1), pred_np, target_names=label_list)

            val_accuracy = np.mean(val_metrics['Accuracy'])
            val_f1_good = val_metrics['F1'][0]
            val_f1_usable = val_metrics['F1'][1]
            val_f1_reject = val_metrics['F1'][2]
            val_sensitivity_reject = val_metrics['Sensitivity'][2]

            score = selection_value(val_metrics, validation_loss, args.selection_metric)
            print('Epoch {} | val_loss={:.4f} kappa={:.4f} macroF1={:.4f} AUC={:.4f} | '
                  'selection[{}]={:.4f} | best={:.4f} @epoch {}'.format(
                      epoch + 1, validation_loss, val_metrics['Kappa'], val_metrics['macro-F1'],
                      float(np.mean(val_metrics['AUC'])), args.selection_metric, score,
                      best_score, best_iter + 1))

        if epoch >= args.warmup_epochs and scheduler is not None:
            scheduler.step()

        if local_rank == 0 and score > best_score:
            best_score = score
            best_val_loss = validation_loss
            best_iter = epoch
            if not os.path.exists(args.model_dir):
                os.makedirs(args.model_dir)
            model_save_file = os.path.join(args.model_dir, args.save_model + '.tar')
            torch.save({'state_dict': model.module.state_dict(),
                        'best_score': best_score,
                        'best_val_loss': best_val_loss,
                        'selection_metric': args.selection_metric}, model_save_file)
            print('Model saved to %s' % model_save_file)

        if local_rank == 0 and args.progress_file is not None:
            with open(args.progress_file, 'a') as pf:
                pf.write(json.dumps({
                    'epoch': epoch,
                    'val_loss': validation_loss,
                    'kappa': float(val_metrics['Kappa']),
                    'macro_f1': float(val_metrics['macro-F1']),
                    'auc': float(np.mean(val_metrics['AUC'])),
                    'score': float(score),
                    'best_score': float(best_score),
                }) + '\n')

        if local_rank == 0 and writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/validation", validation_loss, epoch)
            writer.add_scalar("Learning_Rate", current_lr, epoch)
            writer.add_scalar("Selection/score", score, epoch)
            writer.add_scalar("Kappa/validation", val_metrics['Kappa'], epoch)

            writer.add_scalar("Accuracy/validation", val_accuracy, epoch)
            writer.add_scalar("F1/Good", val_f1_good, epoch)
            writer.add_scalar("F1/Usable", val_f1_usable, epoch)
            writer.add_scalar("F1/Reject", val_f1_reject, epoch)
            writer.add_scalar("Sensitivity/Reject", val_sensitivity_reject, epoch)
            writer.flush()

    dist.barrier()
    training_time = time.time() - t0
    if local_rank == 0:
        print(f'\nTraining complete. Best {args.selection_metric} score: {best_score:.4f} '
              f'(val_loss {best_val_loss:.4f}) at epoch {best_iter+1}')
        print(f'Training time: {training_time:.2f} seconds')
        if writer is not None:
            writer.flush()
            writer.close()

    if local_rank == 0 and args.tuning:
        metrics_path = os.path.join(args.model_dir, args.save_model + '_metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"Objective: {best_score:.6f}\n")
            f.write(f"Selection_Metric: {args.selection_metric}\n")
            f.write(f"Best_Val_Loss: {best_val_loss:.6f}\n")
            f.write(f"Best_Val_Score: {best_score:.6f}\n")
            f.write(f"Best_Epoch: {best_iter + 1}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Training Time: {training_time:.1f}s\n")
        print(f'[TUNING] Objective ({args.selection_metric}) = {best_score:.6f} written to {metrics_path}')

    if local_rank == 0 and not args.tuning:
        best_model_path = os.path.join(args.model_dir, args.save_model + '.tar')
        eval_model = dense121_mcs(n_class=args.n_classes, pretrained=args.pretrained)
        eval_model.to(device)

        checkpoint = torch.load(best_model_path, map_location=device)
        eval_model.load_state_dict(checkpoint['state_dict'])
        eval_model.eval()

        eval_loader = torch.utils.data.DataLoader(
            dataset=data_test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        outPRED_mcs = torch.empty((0, args.n_classes), device=device)
        iters_per_epoch = len(eval_loader)
        bar = Bar('Processing {}'.format('inference'), max=len(eval_loader))
        bar.check_tty = False

        for epochID, (imagesA, imagesB, imagesC, labels) in enumerate(eval_loader):
            imagesA = imagesA.to(device, non_blocking=True)
            imagesB = imagesB.to(device, non_blocking=True)
            imagesC = imagesC.to(device, non_blocking=True)

            with torch.no_grad():
                begin_time = time.time()
                _, _, _, _, result_mcs = eval_model(imagesA, imagesB, imagesC)
                outPRED_mcs = torch.cat((outPRED_mcs, result_mcs), 0)
            batch_time = time.time() - begin_time
            bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(
                epochID + 1, len(eval_loader),
                batch_time=batch_time * (iters_per_epoch - epochID) / 60
            )
            bar.next()

        bar.finish()

        processed_image_names = data_test.csv_image_names
        save_output(label_test_file, outPRED_mcs, args, save_file=save_file_name, processed_images=processed_image_names)

        df_gt = pd.read_csv(label_test_file)
        label_list = args.label_idx

        df_tmp = pd.read_csv(save_file_name)
        img_num = len(df_tmp)

        processed_image_names = df_tmp["image_name"].tolist()
        df_gt_filtered = df_gt[df_gt["image"].isin(processed_image_names)].reset_index(drop=True)
        GT_QA_list = np.array(df_gt_filtered["quality"].tolist())

        predict_tmp = np.zeros([img_num, len(label_list)])
        for idx in range(len(label_list)):
            predict_tmp[:, idx] = np.array(df_tmp[label_list[idx]].tolist())
        tmp_report = compute_metric(GT_QA_list, predict_tmp, target_names=label_list)

        print('\n' + '=' * 60)
        print('FINAL TEST RESULTS (seed {}):'.format(args.seed))
        print(' Accuracy: ' + str("{:0.4f}".format(np.mean(tmp_report['Accuracy']))) +
              ' Precision: ' + str("{:0.4f}".format(np.mean(tmp_report['Precision']))) +
              ' Sensitivity: ' + str("{:0.4f}".format(np.mean(tmp_report['Sensitivity']))) +
              ' F1: ' + str("{:0.4f}".format(np.mean(tmp_report['F1']))) +
              ' AUC: ' + str("{:0.4f}".format(np.mean(tmp_report['AUC']))) +
              ' Kappa: ' + str("{:0.4f}".format(tmp_report['Kappa'])))
        print('=' * 60)

        with open(os.path.join(args.model_dir, args.save_model + '_metrics.txt'), 'w') as f:
            f.write(f"Objective: {best_score:.6f}\n")
            f.write(f"Selection_Metric: {args.selection_metric}\n")
            f.write(f"Best_Val_Loss: {best_val_loss:.6f}\n")
            f.write(f"Best_Val_Score: {best_score:.6f}\n")
            f.write(f"Best_Epoch: {best_iter + 1}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Accuracy    : {np.mean(tmp_report['Accuracy']):.4f}\n")
            f.write(f"Precision   : {np.mean(tmp_report['Precision']):.4f}\n")
            f.write(f"Sensitivity : {np.mean(tmp_report['Sensitivity']):.4f}\n")
            f.write(f"F1          : {np.mean(tmp_report['F1']):.4f}\n")
            f.write(f"AUC         : {np.mean(tmp_report['AUC']):.4f}\n")
            f.write(f"Kappa       : {tmp_report['Kappa']:.4f}\n")
            f.write(f"Training Time: {training_time:.1f}s\n")

    cleanup_ddp()

if __name__ == "__main__":
    main()
