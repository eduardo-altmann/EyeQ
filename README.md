# EyeQ — Retinal Fundus Image Quality Assessment

The original project is not mine, you can check it out here: https://github.com/HzFu/EyeQ/

This repository reproduces the MCF-Net experiment from the original repository and adds:

- **Multi-GPU (DDP) training** via `torchrun` (`MCF_Net/Main_EyeQuality_tuned_parallel.py`).
- A **hyperparameter optimization (HPO) pipeline** built on Optuna (`MCF_Net/optuna_driver.py`).
- A more **robust evaluation protocol** (held-out validation split, ordinal-aware selection metric)
  designed to make the tuning results reproducible and comparable to the baseline paper.

The model is MCF-Net: three DenseNet121 branches over the RGB, HSV and LAB color spaces whose
outputs are fused; quality is graded into three ordinal classes — **Good < Usable < Reject**.

## Repository layout

- `EyeQ_preprocess/` — masks, crops, centers and resizes the raw fundus images, and filters the
  label CSVs (`data/Label_EyeQ_*.filtered.csv`) so they only reference images that exist.
- `MCF_Net/` — training and inference:
  - `Main_EyeQuality_tuned_parallel.py` — DDP training + single-trial runner for the HPO loop.
  - `optuna_driver.py` — Optuna study that launches one `torchrun` subprocess per trial.
  - `Main_EyeQuality_test.py` — standalone evaluation on the test set.
- `slurm/` — cluster submission scripts (`submit_experiment.sh`, `train_job.sbatch`).

## Hyperparameter optimization

The HPO pipeline was designed to avoid the two most common failure modes in tuning studies:
**test-set leakage** (selecting hyperparameters on the same data used to report final numbers) and
**objective/metric mismatch** (optimizing a loss that does not reflect the ordinal grading task).

### Search space

Sampled per trial in `optuna_driver.py`:

| Hyperparameter  | Range / choices              | Scale       |
| --------------- | ---------------------------- | ----------- |
| `lr`            | 1e-4 – 1e-2                  | log-uniform |
| `batch_size`    | {4, 8, 16}                   | categorical |
| `weight_decay`  | 1e-6 – 1e-3                  | log-uniform |
| `lr_scheduler`  | {cosine, step}               | categorical |
| `warmup_epochs` | 2 – 6                        | integer     |
| `momentum`      | 0.8 – 0.99                   | uniform     |

### Selection metric (objective)

By default the study **maximizes the quadratic-weighted Cohen's kappa** on a held-out validation
split, because kappa respects the ordinal ordering of the classes (misgrading Good→Reject is
penalized more than Good→Usable). Alternatives are selectable via `--selection_metric`:
`kappa` (default), `macro_f1`, `auc`, or `loss` (the training script negates the loss so the study
still maximizes).

### Evaluation protocol (no leakage)

- The training script builds a **stratified train/validation split from the training set only**
  (`--val_split`, default 0.1, seeded so the split is identical across trials).
- All per-epoch model selection and the Optuna objective use the **validation split**.
- The **test set is touched exactly once**, in the final (non-`--tuning`) run, to report the
  published metrics (Accuracy, F1, AUC, quadratic-kappa, sensitivity, precision).

### Sampler, pruning and warm-start

- **Sampler:** `TPESampler` (seeded) with `n_startup_trials` random trials before it starts modelling.
- **Pruner:** `MedianPruner` — under-performing trials are stopped early (after a warmup of
  `--pruner_warmup_epochs`). The driver reads a per-epoch JSONL progress file, calls
  `trial.report()` / `trial.should_prune()`, and kills the whole `torchrun` process group on prune.
- **Warm-start:** the known-good hand-tuned configuration (`lr=1e-3, batch_size=4, weight_decay=1e-4,
  cosine, warmup=5, momentum=0.9`) is enqueued as the first trial, so the study can never end up worse
  than the baseline.
- **Seeding:** a fixed `--seed` is passed to every trial so differences reflect hyperparameters, not RNG.

### Running the HPO study

From `MCF_Net/`:

```bash
# Launch an Optuna study (4 GPUs, 6 trials, 25 epochs per trial)
python optuna_driver.py --n_gpus 4 --n_trials 6 --epochs 25 \
    --selection_metric kappa --seed 0

# Inspect live (optional)
optuna-dashboard sqlite:///optuna_study.db
```

Each trial internally runs:

```bash
torchrun --nproc_per_node=4 Main_EyeQuality_tuned_parallel.py --tuning \
    --save_model trial_<n> --progress_file ./result/trial_<n>_progress.jsonl \
    --lr ... --batch-size ... --weight_decay ... --lr_scheduler ... \
    --momentum ... --warmup_epochs ... --epochs 25
```

On the SLURM cluster the whole run (preprocess → dashboards → study) is submitted with
`slurm/submit_experiment.sh`, which auto-increments the run id and calls `slurm/train_job.sbatch`.

### Final training with the best configuration

Re-run `Main_EyeQuality_tuned_parallel.py` **without** `--tuning`, passing the best hyperparameters
found by the study, to train the final model and produce the test-set metrics. For reproducible
reporting, run the best configuration over several seeds and report mean ± std.

## References

### Domain — fundus image quality assessment

- Li H., Li H., Ou M., Yu X., Zhang X., Niu K., Fu H., Liu J. *Fundus Image Quality Assessment and
  Enhancement: a Systematic Review.* arXiv:2501.11520, 2025.
- Lee Q.Z., Wong O.J.H., Che Omar N.A.B., et al. *FundaQ-8: A Clinically-Inspired Scoring Framework
  for Automated Fundus Image Quality Assessment.* arXiv:2506.20303, 2025.
- Gong Z., Deng Z., Gan R., et al., Ma L. *Acquire Precise and Comparable Fundus Image Quality Score:
  FTHNet and FQS Dataset.* arXiv:2411.12273, 2024.
- Wang P., Morano J., Mares V., Bogunović H. *FunPiQ: A New Benchmark for Pixel-Level Quality
  Assessment in Fundus Images.* MICCAI 2026, arXiv:2606.25915.
- Wang P., Morano J., Wan Q., Bogunović H. *EFIQA: Explainable Fundus Image Quality Assessment via
  Anatomical Priors.* MIDL 2026 (PMLR 315:2248–2264), arXiv:2606.20108.
- Fu H., Wang B., Shen J., Cui S., Xu Y., Liu J., Shao L. *Evaluation of Retinal Image Quality
  Assessment Networks in Different Color-Spaces (MCF-Net / EyeQ).* MICCAI 2019.

### Retinal foundation models

- Zhou Y., Chia M.A., … Keane P.A. *A foundation model for generalizable disease detection from
  retinal images (RETFound).* Nature, 2023. doi:10.1038/s41586-023-06555-x.
- Engelmann J., Bernabeu M.O. *Training a high-performance retinal foundation model with half-the-data
  and 400× less compute (RETFound-Green).* Nature Communications, 2025. doi:10.1038/s41467-025-62123-z.

### Hyperparameter optimization methodology

- Bischl B., Binder M., Lang M., et al. *Hyperparameter optimization: Foundations, algorithms, best
  practices, and open challenges.* WIREs Data Mining and Knowledge Discovery 13(2):e1484, 2023.
- Rodrigues C., et al. *When Is an LLM Worth It for Hyperparameter Optimization? A Budget-Matched,
  Multi-Seed Study.* arXiv:2606.21641, 2026.
- Akiba T., Sano S., Yanase T., Ohta T., Koyama M. *Optuna: A Next-generation Hyperparameter
  Optimization Framework.* KDD 2019.
- Li L., et al. *A System for Massively Parallel Hyperparameter Tuning (ASHA).* MLSys 2020.