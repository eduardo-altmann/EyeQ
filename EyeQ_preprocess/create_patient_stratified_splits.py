import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


SPLIT_NAMES = ('train', 'validation', 'test')


def patient_id(image_name):
    stem = os.path.splitext(os.path.basename(str(image_name)))[0]
    for suffix in ('_left', '_right'):
        if stem.endswith(suffix):
            return stem[:-len(suffix)]
    return stem


def _load_source(csv_path, source):
    frame = pd.read_csv(csv_path)
    required = {'image', 'quality', 'DR_grade'}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f'{csv_path} is missing columns: {sorted(missing)}')

    frame = frame[['image', 'quality', 'DR_grade']].copy()
    frame['source'] = source
    frame['patient_id'] = frame['image'].map(patient_id)
    return frame


def _distribution_error(frame, target_counts):
    actual = frame['stratum'].value_counts().reindex(target_counts.index, fill_value=0)
    return float(np.abs(actual - target_counts).div(target_counts.clip(lower=1)).sum())


def create_patient_stratified_splits(train_csv, test_csv, output_dir, seed=0):
    combined = pd.concat(
        [_load_source(train_csv, 'train'), _load_source(test_csv, 'test')],
        ignore_index=True,
    )

    duplicate_images = combined['image'].duplicated(keep=False)
    if duplicate_images.any():
        duplicates = sorted(combined.loc[duplicate_images, 'image'].unique())
        raise ValueError(f'Duplicate image names across sources: {duplicates[:5]}')

    patient_sources = combined.groupby('patient_id')['source'].nunique()
    if (patient_sources > 1).any():
        patients = sorted(patient_sources[patient_sources > 1].index)
        raise ValueError(f'Patients overlap across official sources: {patients[:5]}')

    combined['stratum'] = (
        combined['source'].astype(str)
        + '|Q' + combined['quality'].astype(str)
        + '|DR' + combined['DR_grade'].astype(str)
    )
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = [
        validation_idx
        for _, validation_idx in splitter.split(
            np.zeros(len(combined)), combined['stratum'], combined['patient_id']
        )
    ]

    target_counts = combined['stratum'].value_counts() * 0.1
    ranked_folds = sorted(
        range(len(folds)),
        key=lambda index: (
            _distribution_error(combined.iloc[folds[index]], target_counts),
            abs(len(folds[index]) - len(combined) * 0.1),
            index,
        ),
    )
    validation_fold, test_fold = ranked_folds[:2]
    assignments = np.full(len(combined), 'train', dtype=object)
    assignments[folds[validation_fold]] = 'validation'
    assignments[folds[test_fold]] = 'test'
    combined['split'] = assignments

    patient_split_counts = combined.groupby('patient_id')['split'].nunique()
    if (patient_split_counts > 1).any():
        raise RuntimeError('Patient overlap detected in generated patient-stratified splits')

    os.makedirs(output_dir, exist_ok=True)
    output_columns = ['image', 'quality', 'DR_grade', 'source']
    outputs = {}
    for split_name in SPLIT_NAMES:
        split_frame = combined[combined['split'] == split_name][output_columns].reset_index(drop=True)
        output_path = os.path.join(
            output_dir, f'Label_EyeQ_patient_stratified_{split_name}.csv'
        )
        split_frame.to_csv(output_path, index=False)
        outputs[split_name] = output_path

        quality_counts = split_frame['quality'].value_counts().sort_index().to_dict()
        print(f'{split_name}: images={len(split_frame)}, quality={quality_counts}')

    return outputs


def main():
    parser = argparse.ArgumentParser(
        description='Create deterministic patient-stratified EyeQ 80/10/10 manifests'
    )
    parser.add_argument('--train-csv', default='../data/Label_EyeQ_train.filtered.csv')
    parser.add_argument('--test-csv', default='../data/Label_EyeQ_test.filtered.csv')
    parser.add_argument('--output-dir', default='../data')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    create_patient_stratified_splits(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
