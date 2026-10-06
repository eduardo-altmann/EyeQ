"""
Gera as saídas do MCF-Net (fused, `combine3`) em validação e teste, uma única vez,
para o ajuste de prior pós-hoc (prior_adjustment.py).

Roda em UM processo/GPU, sem DistributedSampler (ele repete amostras no último lote
e distorce as métricas). Precisa do protocolo patient_stratified, pois é o único com
conjunto de validação.

Uso (a partir de MCF_Net/):
  python dump_scores.py --model_dir ./result --save_model DenseNet121_v3_tuned

Saída em <model_dir>/<save_model>_scores/:
  {val,test}_scores.npy   (N, 3) saída sigmoide do ramo fused, ordem Good/Usable/Reject
  {val,test}_labels.npy   (N,)   rótulo inteiro 0/1/2
  {val,test}_images.csv   nomes das imagens, na mesma ordem
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms

from dataloader.EyeQ_loader import DatasetGenerator
from networks.densenet_mcf import dense121_mcs

DATA_ROOT = '../EyeQ_preprocess/'
MANIFESTS = {
    'val': '../data/Label_EyeQ_patient_stratified_validation.csv',
    'test': '../data/Label_EyeQ_patient_stratified_test.csv',
}


@torch.no_grad()
def dump(model, dataset, args, device, out_dir, split):
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                         num_workers=4, pin_memory=True)
    scores, labels = [], []
    for imgs_a, imgs_b, imgs_c, y in loader:
        _, _, _, _, fused = model(imgs_a.to(device, non_blocking=True),
                                  imgs_b.to(device, non_blocking=True),
                                  imgs_c.to(device, non_blocking=True))
        scores.append(fused.float().cpu())
        labels.append(y.argmax(dim=1))  # o loader devolve one-hot

    np.save(os.path.join(out_dir, f'{split}_scores.npy'), torch.cat(scores).numpy())
    np.save(os.path.join(out_dir, f'{split}_labels.npy'), torch.cat(labels).numpy())
    pd.Series(dataset.csv_image_names, name='image').to_csv(
        os.path.join(out_dir, f'{split}_images.csv'), index=False)
    print(f'{split}: {len(dataset)} imagens salvas em {out_dir}')


def main():
    parser = argparse.ArgumentParser(description='EyeQ_dump_scores')
    parser.add_argument('--model_dir', type=str, default='./result/')
    parser.add_argument('--save_model', type=str, default='DenseNet121_v3_tuned')
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--crop_size', type=int, default=224)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--out_dir', type=str, default=None,
                        help='default: <model_dir>/<save_model>_scores')
    args = parser.parse_args()

    for path in MANIFESTS.values():
        if not os.path.exists(path):
            raise FileNotFoundError(
                'Missing patient-stratified manifest. Run '
                'EyeQ_preprocess/create_patient_stratified_splits.py first: ' + path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = args.out_dir or os.path.join(args.model_dir, args.save_model + '_scores')
    os.makedirs(out_dir, exist_ok=True)

    model = dense121_mcs(n_class=args.n_classes, pretrained=False)
    checkpoint = torch.load(os.path.join(args.model_dir, args.save_model + '.tar'),
                            map_location=device)
    protocol = checkpoint.get('data_protocol')
    if protocol is not None and protocol != 'patient_stratified':
        raise ValueError(f'Checkpoint protocol is {protocol}; prior adjustment needs '
                         'patient_stratified (validation split required)')
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device).eval()

    # mesmo pré-processamento determinístico do teste (sem data augmentation)
    transform1 = transforms.Compose([
        transforms.Resize(args.crop_size),
        transforms.CenterCrop(args.crop_size),
    ])
    transform2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    for split, manifest in MANIFESTS.items():
        dataset = DatasetGenerator(data_dir=DATA_ROOT, list_file=manifest,
                                   transform1=transform1, transform2=transform2,
                                   n_class=args.n_classes, set_name=split)
        dump(model, dataset, args, device, out_dir, split)


if __name__ == '__main__':
    main()
