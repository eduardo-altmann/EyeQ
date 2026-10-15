"""
Resume uma execução de cross-validation (submit_single_experiment.sh --folds):
lê <execução>/fold_<k>/training/*_metrics.txt e escreve <execução>/cv_summary.csv
com as métricas no test de cada fold e a média ± desvio padrão entre os folds.

Uso:
  python aggregate_folds.py ~/eyeq/results/single/<experimento>/<ID>
"""
import argparse
import glob
import os
import re

import pandas as pd

METRICS = ['Accuracy', 'Precision', 'Sensitivity', 'F1', 'AUC', 'Kappa',
           'Best_Val_Score', 'Best_Epoch']


def read_metrics(path):
    values = {}
    with open(path) as f:
        for line in f:
            if line.startswith('Per-class breakdown'):
                break  # o resto é por classe
            key, sep, value = line.partition(':')
            key = key.strip()
            if sep and key in METRICS:
                values[key] = float(value)
    return values


def main():
    parser = argparse.ArgumentParser(description='Aggregate EyeQ cross-validation folds')
    parser.add_argument('run_dir', help='pasta da execução, que contém fold_<k>/')
    parser.add_argument('--out', default=None, help='default: <run_dir>/cv_summary.csv')
    args = parser.parse_args()

    rows = []
    fold_dirs = glob.glob(os.path.join(args.run_dir, 'fold_*'))
    for fold_dir in sorted(fold_dirs, key=lambda d: int(re.sub(r'\D', '', os.path.basename(d)))):
        fold = int(os.path.basename(fold_dir).split('_')[1])
        metric_files = glob.glob(os.path.join(fold_dir, 'training', '*_metrics.txt'))
        if len(metric_files) != 1:
            print(f'fold {fold}: esperado 1 *_metrics.txt, achados {len(metric_files)}; ignorado')
            continue
        rows.append({'fold': fold, **read_metrics(metric_files[0])})

    if not rows:
        raise SystemExit(f'Nenhum fold com métricas em {args.run_dir}')

    table = pd.DataFrame(rows).set_index('fold')
    summary = pd.concat([table, table.agg(['mean', 'std'])])
    out = args.out or os.path.join(args.run_dir, 'cv_summary.csv')
    summary.to_csv(out, float_format='%.4f')

    print(summary.round(4).to_string())
    print(f'\n{len(table)} folds; resumo salvo em {out}')


if __name__ == '__main__':
    main()
