# R.O.A.D. Barbados Historic Handwriting Challenge

Line-level historical handwriting OCR (Zindi). Final leaderboard score is:

```
0.5 * weighted WER + 0.5 * weighted CER
```

where, for each metric, per-line edit distance `edit_distance_i` is weighted by
`W_i = L_i^0.5` (`L_i` = reference word count for WER, reference char count for CER):

```
Global score = sum(edit_distance_i * W_i) / sum(W_i)
```

(Metric implementation lives in `src/metrics/` — see that folder once it's built out.)

## Repo layout

```
road-barbados-htr/
  data/
    Train.csv           # 4098 rows
    Test.csv            # 1374 rows
    SampleSubmission.csv
    images/              # 5472 .jpg line crops, flat, gitignored
  starters/               # original Zindi starter pack, untouched, for reference
    VLM/
    Kraken-OCR/
    Paddle-OCR/
  src/
    metrics/              # weighted WER/CER implementation
    data/                 # dataset loading, verification, splits
    kraken/                # Kraken-OCR training/inference (.venv-kraken)
    vlm/                    # Qwen/transformers VLM fine-tuning (.venv-vlm)
    ensemble/              # combining predictions across models
  submissions/            # every submission.csv we generate, timestamped
  experiments/             # one subfolder per run, with config + metrics logged
  requirements.txt          # shared deps (pandas, metrics, tooling)
  README.md
```

## Compute environment

- GPU: NVIDIA GeForce RTX 2050, 4096 MiB VRAM (driver CUDA 13.2, nvcc/toolkit 13.1)
- Python: 3.11.9

4 GB of VRAM is tight for VLM fine-tuning at full precision — plan on
4-bit/QLoRA (`bitsandbytes` is already in `src/vlm/requirements.txt`) and small
batch sizes with gradient accumulation.

## Environment setup

Kraken and the transformers/Qwen VLM stack pin conflicting torch/numpy
versions, so they live in separate virtual environments. Shared tooling
(pandas, metrics, ensembling) uses a plain top-level environment.

**Shared environment** (metrics, data loading, ensembling):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash; use .venv\Scripts\Activate.ps1 in PowerShell
pip install -r requirements.txt
```

**Kraken-OCR** (`.venv-kraken`, already created):

```bash
source .venv-kraken/Scripts/activate   # PowerShell: .venv-kraken\Scripts\Activate.ps1
pip install -r src/kraken/requirements.txt
```

**VLM / transformers** (`.venv-vlm`, already created):

```bash
source .venv-vlm/Scripts/activate      # PowerShell: .venv-vlm\Scripts\Activate.ps1
pip install -r src/vlm/requirements.txt
```

Package installation was not run yet as part of repo setup (large downloads,
GPU-specific torch builds) — run the `pip install` step above inside each venv
before training.

Verify the data at any time with:

```bash
python src/data/verify.py
```

## Results log

| Date | Tier | Model | Local weighted CER | Local weighted WER | Local final score | Public LB score |
|------|------|-------|---------------------|---------------------|--------------------|------------------|
|      |      |       |                     |                     |                    |                  |
