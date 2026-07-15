# DCC-MER

Official implementation scaffold for **DCC-MER: Diffusion-Inspired Calibrated Consensus Learning for Multimodal Emotion Recognition**.

DCC-MER is a multimodal emotion recognition framework for learning reliable cross-modal consensus. The implementation contains the three main modules used in the paper:

- **TPLR**
- **PCRP**
- **RCCR**

This `github/` folder is prepared as a release-ready GitHub package. When publishing the code, place this README, `environment.yml`, `requirements.txt`, and the `scripts/` directory at the same repository level as `dcc_main.py`, `src/`, `dcc_best_config.py`, and `dcc_multiseed_tools/`.

## Repository Layout

Expected layout after copying these files into the code repository:

```text
DCC-MER/
  README.md
  environment.yml
  requirements.txt
  dcc_main.py
  dcc_best_config.py
  summarize_dcc_all_5runs.py
  collect_dcc_8ablation_results.py
  src/
  dcc_multiseed_tools/
    run_mechanism_controls.py
    summarize_mechanism_controls.py
    run_robustness_retrain.py
    summarize_robustness_retrain.py
    run_diagnostics_retrain.py
    analyze_retrained_rccr.py
    analyze_retrained_tplr.py
    summarize_retrain_diagnostics.py
  scripts/
    train_dcc_mer.sh
    evaluate_saved_metrics.py
    run_all_5seeds.sh
    run_8ablation.sh
    run_robustness.sh
    run_mechanism_controls.sh
    run_diagnostics.sh
    summarize_results.sh
```

## Environment

The experiments in the revised manuscript were run with Python 3.10 and PyTorch 2.7.1.

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate dcc-mer
```

Or install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA, install the PyTorch build that matches your local driver and CUDA runtime. The provided environment file uses the CUDA-enabled PyTorch channel. For CPU-only debugging, replace the PyTorch installation with the official CPU build.

## Datasets

DCC-MER expects preprocessed multimodal features. The code supports:

- `SIMS`
- `SIMS-v2`
- `MOSI`
- `MOSEI`

Set `DATA_PATH` to the parent directory that contains the processed dataset folders:

```bash
export DATA_PATH=/path/to/MSA_Datasets
```

Expected processed files:

```text
$DATA_PATH/
  CH-SIMS/
    Processed/
      unaligned_39.pkl
  CH-SIMS-v2/
    CH-SIMS-v2(s)/
      Processed/
        unaligned.pkl
  CMU-MOSI/
    Processed/
      unaligned_50.pkl
  CMU-MOSEI/
    Processed/
      unaligned_50.pkl
```

The datasets are not redistributed in this repository. Please obtain them from the original dataset providers and follow their licenses.

## Quick Start

Run the two-stage DCC-MER training and evaluation pipeline on one dataset:

```bash
bash scripts/train_dcc_mer.sh SIMS "$DATA_PATH" runs/quickstart/SIMS/seed_3328683074 0 3328683074
```

Arguments:

```text
bash scripts/train_dcc_mer.sh <dataset> <data_path> <run_dir> <gpu_id> <seed>
```

Example for MOSI:

```bash
bash scripts/train_dcc_mer.sh MOSI "$DATA_PATH" runs/quickstart/MOSI/seed_3328683074 0 3328683074
```

The script runs:

1. Stage 1 consensus pretraining with TPLR.
2. Stage 2 finetuning with TPLR, PCRP, and RCCR.

Main outputs are saved under `run_dir`:

```text
run_dir/
  logs/
    stage1.log
    stage2.log
  models/
  pseudo_labels/
  stage1_summary.json
  metrics.json
```

## Direct Commands

The single-run script is equivalent to the following commands.

Stage 1:

```bash
python -u dcc_main.py \
  --dataset SIMS \
  --data_path "$DATA_PATH" \
  --model_path runs/quickstart/SIMS/seed_3328683074/models \
  --run_dir runs/quickstart/SIMS/seed_3328683074 \
  --use_best \
  --is_pseudo \
  --use_tplr \
  --seed 3328683074
```

Stage 2:

```bash
python -u dcc_main.py \
  --dataset SIMS \
  --data_path "$DATA_PATH" \
  --model_path runs/quickstart/SIMS/seed_3328683074/models \
  --run_dir runs/quickstart/SIMS/seed_3328683074 \
  --use_best \
  --is_pseudo \
  --finetune \
  --pretrained_model \
  --use_tplr \
  --use_pcrp \
  --use_rccr \
  --seed 3328683074
```

Use `--use_best` to load the dataset-specific hyperparameters from `dcc_best_config.py`.

## Reproduce Main Multi-Seed Results

Run five seeds on one or more datasets:

```bash
bash scripts/run_all_5seeds.sh "$DATA_PATH" runs/dcc_all_5runs 0 SIMS,MOSI,MOSEI
```

Default seeds:

```text
3328683074、1974074723、1686464603
```

Override the seed list:

```bash
SEEDS=3328683074,1974074723,1686464603 \
bash scripts/run_all_5seeds.sh "$DATA_PATH" runs/dcc_all_3runs 0 SIMS,MOSI
```

After training, summarize the results:

```bash
bash scripts/summarize_results.sh 5run runs/dcc_all_5runs
```

## Ablation Study

Run the eight module-combination ablations:

```bash
bash scripts/run_8ablation.sh "$DATA_PATH" runs/dcc_8ablation 0 SIMS,MOSI
```

The combinations include:

```text
BASE
TPLR
PCRP
RCCR
TPLR_PCRP
TPLR_RCCR
PCRP_RCCR
DCC_FULL
```

Summarize ablations:

```bash
bash scripts/summarize_results.sh ablation runs/dcc_8ablation
```

## Mechanism Controls

Run the mechanism-control experiments used to analyze module interactions beyond the eight ablation combinations:

```bash
bash scripts/run_mechanism_controls.sh "$DATA_PATH" runs/mechanism_controls 0 SIMS,MOSI
```

Summarize:

```bash
bash scripts/summarize_results.sh mechanism runs/mechanism_controls
```

## Robustness Evaluation

Run missing-modality and noise robustness experiments:

```bash
bash scripts/run_robustness.sh "$DATA_PATH" runs/robustness_retrain 0 SIMS,MOSI
```

The robustness protocol uses condition-specific retraining/evaluation as implemented in `dcc_multiseed_tools/run_robustness_retrain.py`.

Summarize:

```bash
bash scripts/summarize_results.sh robustness runs/robustness_retrain
```

## RCCR, TPLR, and Cost Diagnostics

Run the retrained diagnostic experiments:

```bash
bash scripts/run_diagnostics.sh "$DATA_PATH" runs/retrain_diagnostics 0 SIMS,MOSI
```

This wrapper runs:

- `run_diagnostics_retrain.py`
- `analyze_retrained_rccr.py`
- `analyze_retrained_tplr.py`
- `summarize_retrain_diagnostics.py`

Summarize:

```bash
bash scripts/summarize_results.sh diagnostics runs/retrain_diagnostics
```

## Inspect Saved Metrics

Print saved `metrics.json` files:

```bash
python scripts/evaluate_saved_metrics.py runs/quickstart/SIMS/seed_3328683074
```

Scan a whole run root and save a CSV:

```bash
python scripts/evaluate_saved_metrics.py runs/dcc_all_5runs --csv runs/dcc_all_5runs/metrics_index.csv
```

## Common Options

All shell wrappers support these environment variables:

```bash
PYTHON=python
PROJECT_DIR=/path/to/DCC-MER
SEEDS=3328683074,1974074723,1686464603
CUDA_VISIBLE_DEVICES=0
```

The scripts also accept a GPU id argument and set `CUDA_VISIBLE_DEVICES` internally.

## Notes on Reproducibility

- Use the same processed feature files and the same random seeds when comparing with reported results.
- Use `--use_best` for the tuned dataset-specific hyperparameters.
- Each run should have an isolated `run_dir` so that checkpoints and pseudo labels do not overlap.
- Report mean and standard deviation over multiple seeds. The summary scripts output table-ready statistics from saved run artifacts.

