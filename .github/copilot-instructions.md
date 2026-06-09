# Copilot instructions for EyeQ

## Commands
- Preprocess raw fundus images and regenerate filtered label CSVs (run from `EyeQ_preprocess/`):  
  `python EyeQ_process_main.py`
- Train (single GPU/CPU, run from `MCF_Net/`):  
  `torchrun --nproc_per_node=1 Main_EyeQuality_tuned_parallel.py`
- Multi-GPU training (run from `MCF_Net/`):  
  `torchrun --nproc_per_node=NUM_GPUS Main_EyeQuality_tuned_parallel.py`
- Single evaluation run on the test set (run from `MCF_Net/`):  
  `python Main_EyeQuality_test.py --model_dir ./result --save_model DenseNet121_v3_tuned`

## High-level architecture
- **Preprocessing stage (`EyeQ_preprocess/`)**: `EyeQ_process_main.py` loads JPEGs from `original_img/{train,test}`, uses `fundus_prep.py` to mask/crop/center the fundus region, resizes to 800×800, writes PNGs into `train/` and `test/`, and writes metrics to `metrics/`. It then filters label CSVs into `data/Label_EyeQ_* .filtered.csv` via `filter_missing_labels.py`.
- **Training/inference stage (`MCF_Net/`)**: `DatasetGenerator` reads PNGs and labels, creates three image variants (RGB, HSV, LAB) and returns them per sample. `networks/densenet_mcf.py` runs three DenseNet121 branches and fuses their outputs. Training loops are in `utils/trainer.py`, evaluation/metrics in `utils/metric.py`, and scripts in `Main_EyeQuality*.py` orchestrate training and inference. Outputs are saved under `MCF_Net/result/` as `.tar` checkpoints and `.csv` predictions/metrics.

## Key conventions
- Run scripts from their own directories (`EyeQ_preprocess/` or `MCF_Net/`) because paths are relative (e.g., `../EyeQ_preprocess/`, `../data/Label_EyeQ_*.csv`).
- Label CSVs use columns `image` and `quality`; image names are mapped to PNGs with the same basename (e.g., `xxx.jpeg` → `xxx.png`).
- Use `Label_EyeQ_*.filtered.csv` when you need to guarantee only existing preprocessed images are referenced (the test script defaults to the filtered CSV).
- The model uses three color-space inputs (RGB/HSV/LAB) and combines five loss terms weighted by `loss_w` (see training scripts).
