import os
import csv
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import time
from datetime import datetime
from progress.bar import Bar
import torchvision.transforms as transforms
from dataloader.EyeQ_loader import DatasetGenerator
from utils.trainer import train_step, validation_step, save_output
from utils.metric import compute_metric

import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no display on a SLURM/cluster node
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from networks.densenet_mcf import dense121_mcs

data_root = '../EyeQ_preprocess/'

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

test_images_dir = data_root + '/test'
label_test_file = '../data/Label_EyeQ_test.csv'
save_file_name = args.model_dir + args.save_model + '.csv'


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = dense121_mcs(n_class=args.n_classes, pretrained=args.pretrained)

if args.pre_model is not None:
    loaded_model = torch.load(os.path.join(args.model_dir, args.pre_model + '.tar'))
    model.load_state_dict(loaded_model['state_dict'])
    print(f'Loaded pretrained model: {args.pre_model}')

model.to(device)

transformList2 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_list_val1 = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

data_test = DatasetGenerator(data_dir=test_images_dir, list_file=label_test_file, transform1=transform_list_val1,
                             transform2=transformList2, n_class=args.n_classes, set_name='test')
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.batch_size,
                                          shuffle=False, num_workers=4, pin_memory=True)

# ============================================================
# Load best model for testing
# ============================================================
best_model_path = os.path.join(args.model_dir, args.save_model + '.tar')
checkpoint = torch.load(best_model_path)
model.load_state_dict(checkpoint['state_dict'])

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

mean_accuracy = float(tmp_report['Accuracy'])          # already a single scalar
mean_precision = float(np.mean(tmp_report['Precision']))  # macro-avg over classes
mean_sensitivity = float(np.mean(tmp_report['Sensitivity']))  # macro-avg recall
mean_f1 = float(tmp_report['macro-F1'])                 # already computed for us
mean_auc_macro = float(tmp_report['AUC'])               # mean of per-class AUCs

print('\n' + '=' * 60)
print('FINAL RESULTS:')
print(' Accuracy: ' + str("{:0.4f}".format(mean_accuracy)) +
      ' Precision: ' + str("{:0.4f}".format(mean_precision)) +
      ' Sensitivity: ' + str("{:0.4f}".format(mean_sensitivity)) +
      ' F1: ' + str("{:0.4f}".format(mean_f1)))
print('=' * 60)

with open(os.path.join(args.model_dir, args.save_model + '_metrics.txt'), 'w') as f:
    f.write(f"Accuracy    : {mean_accuracy:.4f}\n")
    f.write(f"Precision   : {mean_precision:.4f}\n")
    f.write(f"Sensitivity : {mean_sensitivity:.4f}\n")
    f.write(f"F1          : {mean_f1:.4f}\n")

# ============================================================
# AUC-ROC, exported as PDF
# Per-class curves come straight from compute_metric's own ROC_curve dict
# (same fpr/tpr/AUC it used internally) so the plot matches the reported
# numbers exactly instead of being recomputed separately.
# ============================================================
n_classes = len(label_list)
roc_data = tmp_report['ROC_curve']
auc_per_class = tmp_report['AUC_per_class'].ravel()

# micro-average (pools all classes together) isn't produced by compute_metric,
# so this one genuinely is computed fresh here
y_true_bin = np.zeros((img_num, n_classes))
for i in range(n_classes):
    y_true_bin[:, i] = (GT_QA_list == i).astype(int)
fpr_micro, tpr_micro, _ = roc_curve(y_true_bin.ravel(), predict_tmp.ravel())
auc_micro = auc(fpr_micro, tpr_micro)

plt.figure(figsize=(7, 7))
colors = ['#2ca02c', '#ff7f0e', '#d62728']
for i, color in zip(range(n_classes), colors):
    plt.plot(roc_data[f'ROC_fpr_{i}'], roc_data[f'ROC_tpr_{i}'], color=color, lw=2,
              label=f'{label_list[i]} (AUC = {auc_per_class[i]:.3f})')

plt.plot(fpr_micro, tpr_micro, color='deeppink', linestyle=':', lw=2,
          label=f'micro-average (AUC = {auc_micro:.3f})')
plt.plot([], [], ' ', label=f'macro-average AUC = {mean_auc_macro:.3f}')  # from compute_metric

plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC Curve — {args.save_model}')
plt.legend(loc='lower right', fontsize=9)
plt.tight_layout()

roc_pdf_path = os.path.join(args.model_dir, args.save_model + '_roc.pdf')
plt.savefig(roc_pdf_path, format='pdf')
plt.close()
print(f'Saved ROC curve to: {roc_pdf_path}')

# ============================================================
# Metrics tracking: append this run to a persistent CSV history
# ============================================================
history_path = os.path.join(args.model_dir, 'metrics_history.csv')
history_exists = os.path.isfile(history_path)

with open(history_path, 'a', newline='') as f:
    writer = csv.writer(f)
    if not history_exists:
        writer.writerow(['timestamp', 'model', 'accuracy', 'precision',
                          'recall', 'f1', 'auc_micro', 'auc_macro'])
    writer.writerow([
        datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        args.save_model,
        f'{mean_accuracy:.4f}',
        f'{mean_precision:.4f}',
        f'{mean_sensitivity:.4f}',
        f'{mean_f1:.4f}',
        f'{auc_micro:.4f}',
        f'{mean_auc_macro:.4f}',
    ])
print(f'Appended run to: {history_path}')