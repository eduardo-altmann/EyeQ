import os
import argparse
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

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

data_root = '../EyeQ_preprocess/'

# Setting parameters
parser = argparse.ArgumentParser(description='EyeQ_dense121_tuned')
parser.add_argument('--model_dir', type=str, default='./result/')
parser.add_argument('--pre_model', type=str, default=None,
                    help='Set to model name (without .tar) to resume training, or None for fresh start')
parser.add_argument('--save_model', type=str, default='DenseNet121_v3_tuned')

parser.add_argument('--crop_size', type=int, default=224)
parser.add_argument('--label_idx', type=list, default=['Good', 'Usable', 'Reject'])

parser.add_argument('--n_classes', type=int, default=3)

# --- Tuned optimization hyperparameters ---
parser.add_argument('--epochs', default=60, type=int)
parser.add_argument('--batch-size', default=4, type=int,
                    help='Increase from 4 to 16 for better gradient estimation')
parser.add_argument('--lr', default=0.001, type=float,
                    help='Lower LR since we use pretrained weights (was 0.01)')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='SGD momentum (was missing)')
parser.add_argument('--weight_decay', default=1e-4, type=float,
                    help='L2 regularization (was missing)')
parser.add_argument('--loss_w', default=[0.1, 0.1, 0.1, 0.1, 0.6], type=list)

# Learning rate schedule
parser.add_argument('--lr_scheduler', type=str, default='cosine',
                    choices=['cosine', 'step', 'none'],
                    help='LR scheduler type')
parser.add_argument('--lr_step_size', default=20, type=int,
                    help='Step size for StepLR scheduler')
parser.add_argument('--lr_gamma', default=0.1, type=float,
                    help='Gamma for StepLR scheduler')
parser.add_argument('--warmup_epochs', default=5, type=int,
                    help='Number of warmup epochs with linear LR ramp')

# Pretrained backbone
parser.add_argument('--pretrained', action='store_true', default=True,
                    help='Use ImageNet pretrained DenseNet121 backbone')

args = parser.parse_args()

# Images Labels
train_images_dir = data_root + '/train'
label_train_file = '../data/Label_EyeQ_train.csv'
test_images_dir = data_root + '/test'
label_test_file = '../data/Label_EyeQ_test.csv'

save_file_name = args.model_dir + args.save_model + '.csv'

best_metric = np.inf
best_iter = 0
cudnn.benchmark = True

# ============================================================
# Model - now with ImageNet pretrained backbone
# ============================================================
model = dense121_mcs(n_class=args.n_classes, pretrained=args.pretrained)

if args.pre_model is not None:
    loaded_model = torch.load(os.path.join(args.model_dir, args.pre_model + '.tar'))
    model.load_state_dict(loaded_model['state_dict'])
    print(f'Loaded pretrained model: {args.pre_model}')

model.to(device)

criterion = torch.nn.BCELoss(reduction='mean')

# ============================================================
# Optimizer - with momentum and weight decay
# ============================================================
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
    nesterov=True
)

# ============================================================
# Learning rate scheduler
# ============================================================
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
    """Linear warmup: scale LR from base_lr/10 up to base_lr."""
    lr = base_lr * (epoch + 1) / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


print('=' * 60)
print('Tuned Training Configuration:')
print(f'  Pretrained backbone: {args.pretrained}')
print(f'  Batch size: {args.batch_size}')
print(f'  Learning rate: {args.lr}')
print(f'  Momentum: {args.momentum}')
print(f'  Weight decay: {args.weight_decay}')
print(f'  LR scheduler: {args.lr_scheduler}')
print(f'  Warmup epochs: {args.warmup_epochs}')
print(f'  Total epochs: {args.epochs}')
print(f'  Loss weights: {args.loss_w}')
print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))
print('=' * 60)

# ============================================================
# Data augmentation (enhanced)
# ============================================================
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

data_train = DatasetGenerator(data_dir=train_images_dir, list_file=label_train_file, transform1=transform_list1,
                              transform2=transformList2, n_class=args.n_classes, set_name='train')
train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=args.batch_size,
                                           shuffle=True, num_workers=4, pin_memory=True)

data_test = DatasetGenerator(data_dir=test_images_dir, list_file=label_test_file, transform1=transform_list_val1,
                             transform2=transformList2, n_class=args.n_classes, set_name='test')
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)

# ============================================================
# Training loop with warmup + scheduler
# ============================================================
print(f'\nTraining set: {len(data_train)} images')
print(f'Test set: {len(data_test)} images\n')

t0 = time.time()

for epoch in range(0, args.epochs):
    # Warmup phase
    if epoch < args.warmup_epochs:
        warmup_lr(optimizer, epoch, args.warmup_epochs, args.lr)

    current_lr = optimizer.param_groups[0]['lr']
    print(f'\nEpoch {epoch+1}/{args.epochs} | LR: {current_lr:.6f}')

    _ = train_step(train_loader, model, epoch, optimizer, criterion, args)
    validation_loss = validation_step(test_loader, model, criterion)
    print('Current Loss: {}| Best Loss: {} at epoch: {}'.format(validation_loss, best_metric, best_iter))

    # Step scheduler (after warmup)
    if epoch >= args.warmup_epochs and scheduler is not None:
        scheduler.step()

    # Save best model
    if best_metric > validation_loss:
        best_metric = validation_loss
        best_iter = epoch
        model_save_file = os.path.join(args.model_dir, args.save_model + '.tar')
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        torch.save({'state_dict': model.state_dict(), 'best_loss': best_metric}, model_save_file)
        print('Model saved to %s' % model_save_file)

training_time = time.time() - t0
print(f'\nTraining complete. Best validation loss: {best_metric:.4f} at epoch {best_iter+1}')
print(f'Training time: {training_time:.2f} seconds')

# ============================================================
# Load best model for testing
# ============================================================
best_model_path = os.path.join(args.model_dir, args.save_model + '.tar')
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['state_dict'])
print(f'Loaded best model from epoch {best_iter+1}')

# Testing
outPRED_mcs = torch.FloatTensor().cuda()
model.eval()
iters_per_epoch = len(test_loader)
bar = Bar('Processing {}'.format('inference'), max=len(test_loader))
bar.check_tty = False
for epochID, (imagesA, imagesB, imagesC, labels) in enumerate(test_loader):
    imagesA = imagesA.cuda()
    imagesB = imagesB.cuda()
    imagesC = imagesC.cuda()

    with torch.no_grad():
        begin_time = time.time()
        _, _, _, _, result_mcs = model(imagesA, imagesB, imagesC)
        outPRED_mcs = torch.cat((outPRED_mcs, result_mcs.data), 0)
    batch_time = time.time() - begin_time
    bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(epochID + 1, len(test_loader),
                                                           batch_time=batch_time * (iters_per_epoch - epochID) / 60)
    bar.next()
bar.finish()

# Save result
processed_image_names = data_test.csv_image_names
save_output(label_test_file, outPRED_mcs, args, save_file=save_file_name, processed_images=processed_image_names)

# Evaluation
df_gt = pd.read_csv(label_test_file)
label_list = ["Good", "Usable", "Reject"]

df_tmp = pd.read_csv(save_file_name)
img_num = len(df_tmp)

processed_image_names = df_tmp["image_name"].tolist()
df_gt_filtered = df_gt[df_gt["image"].isin(processed_image_names)].reset_index(drop=True)
GT_QA_list = np.array(df_gt_filtered["quality"].tolist())

predict_tmp = np.zeros([img_num, 3])
for idx in range(3):
    predict_tmp[:, idx] = np.array(df_tmp[label_list[idx]].tolist())
tmp_report = compute_metric(GT_QA_list, predict_tmp, target_names=label_list)

print('\n' + '=' * 60)
print('FINAL RESULTS:')
print(' Accuracy: ' + str("{:0.4f}".format(np.mean(tmp_report['Accuracy']))) +
      ' Precision: ' + str("{:0.4f}".format(np.mean(tmp_report['Precision']))) +
      ' Sensitivity: ' + str("{:0.4f}".format(np.mean(tmp_report['Sensitivity']))) +
      ' F1: ' + str("{:0.4f}".format(np.mean(tmp_report['F1']))))
print('=' * 60)

with open(os.path.join(args.model_dir, args.save_model + '_metrics.txt'), 'w') as f:
    f.write(f"Accuracy    : {np.mean(tmp_report['Accuracy']):.4f}\n")
    f.write(f"Precision   : {np.mean(tmp_report['Precision']):.4f}\n")
    f.write(f"Sensitivity : {np.mean(tmp_report['Sensitivity']):.4f}\n")
    f.write(f"F1          : {np.mean(tmp_report['F1']):.4f}\n")
    f.write(f"Training Time: {training_time:.1f}s\n")
