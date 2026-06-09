import os
import argparse
import numpy as np
import torch
import time
from progress.bar import Bar
import torchvision.transforms as transforms
from dataloader.EyeQ_loader import DatasetGenerator
from utils.trainer import save_output
from utils.metric import compute_metric

import pandas as pd
from networks.densenet_mcf import dense121_mcs


def main():
    parser = argparse.ArgumentParser(description='EyeQ_dense121_test')
    parser.add_argument('--model_dir', type=str, default='./result/')
    parser.add_argument('--save_model', type=str, default='DenseNet121_v3_tuned')
    parser.add_argument('--batch-size', default=4, type=int)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--label_idx', type=list, default=['Good', 'Usable', 'Reject'])
    parser.add_argument('--pretrained', action='store_true', default=False)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    data_root = '../EyeQ_preprocess/'
    test_images_dir = data_root + '/test'
    label_test_file = '../data/Label_EyeQ_test.filtered.csv'
    save_file_name = args.model_dir + args.save_model + '.csv'

    # Model
    model = dense121_mcs(n_class=args.n_classes, pretrained=args.pretrained)
    model.to(device)

    best_model_path = os.path.join(args.model_dir, args.save_model + '.tar')
    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    # Data
    transform_list_val1 = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
    ])

    transformList2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    data_test = DatasetGenerator(data_dir=test_images_dir, list_file=label_test_file,
                                 transform1=transform_list_val1, transform2=transformList2,
                                 n_class=args.n_classes, set_name='test')
    test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size,
                                              shuffle=False, num_workers=4, pin_memory=True)

    # Testing
    outPRED_mcs = torch.empty((0, args.n_classes), device=device)
    iters_per_epoch = len(test_loader)
    bar = Bar('Processing {}'.format('inference'), max=len(test_loader))
    bar.check_tty = False

    for epochID, (imagesA, imagesB, imagesC, labels) in enumerate(test_loader):
        imagesA = imagesA.to(device, non_blocking=True)
        imagesB = imagesB.to(device, non_blocking=True)
        imagesC = imagesC.to(device, non_blocking=True)

        with torch.no_grad():
            begin_time = time.time()
            _, _, _, _, result_mcs = model(imagesA, imagesB, imagesC)
            outPRED_mcs = torch.cat((outPRED_mcs, result_mcs), 0)
        batch_time = time.time() - begin_time
        bar.suffix = '{} / {} | Time: {batch_time:.4f}'.format(
            epochID + 1, len(test_loader),
            batch_time=batch_time * (iters_per_epoch - epochID) / 60
        )
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


if __name__ == '__main__':
    main()
