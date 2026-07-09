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
    kraken/                # Kraken-OCR training/inference (.venv-kraken):
                            #   prepare_data.py, train.py, infer.py (pageseg bug fixed),
                            #   lm_rescore.py (opt-in lexicon+bigram rescoring)
    vlm/                    # Qwen/transformers VLM fine-tuning (.venv-vlm)
    ensemble/              # combining predictions across models
  submissions/            # every submission.csv we generate, timestamped
  experiments/             # one subfolder per run, with config + metrics logged
  runners/
    narval/                # Slurm/DRAC HPC job scripts — see runners/narval/README.md
    colab/                  # rclone-backed Colab bootstrap notebook — see runners/colab/README.md
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

`.venv-vlm` currently has only what the tier0 zero-shot baseline needed
(`torch` — note: install from `--index-url https://download.pytorch.org/whl/cu124`,
plain `pip install torch` resolves to a CPU-only wheel on this machine —
`transformers`, `accelerate`, `bitsandbytes`, `pillow`, `tqdm`, `pyyaml`,
`pandas`), not the full `src/vlm/requirements.txt` (`peft`/`trl`/`datasets`/
`evaluate`/`av`/`scikit-learn` still need installing before fine-tuning).

Verify the data at any time with:

```bash
python src/data/verify.py
```

## Other compute environments

This local RTX 2050 box is one of three environments this repo trains on:

- **Narval** (Slurm HPC, no internet on compute nodes) — `runners/narval/`,
  see its README for the login-node setup → prefetch → `sbatch` → pull
  results workflow.
- **Colab** (code/data on Drive via `rclone`, ephemeral VM) — `runners/colab/`,
  see its README for the one-time `rclone.conf` setup and session bootstrap
  (`bootstrap.ipynb` for VLM, `train_kraken.ipynb` for Tier 1 Kraken training).

Both keep the same `src/kraken` / `src/vlm` script interfaces as this local
setup — only the environment bootstrapping differs.

## Tier 0: zero-shot baseline

`src/vlm/tier0_baseline_infer.py` runs the base (non-fine-tuned)
`Qwen/Qwen2-VL-2B-Instruct`, 4-bit quantized, against line-crop images with
the same "transcribe exactly as seen" prompt as the starter. It exists to
de-risk the submission pipeline (CSV format, ID coverage, scorer wiring)
before investing in real fine-tuning — not to score well. Fits in ~1.8GB of
this card's 4GB VRAM; ~3.6s/image.

```bash
source .venv-vlm/Scripts/activate
python -m src.vlm.tier0_baseline_infer --input_csv data/val_split.csv --output_csv <preds.csv>
python -m src.metrics.weighted_wer_cer --gt data/val_split.csv --pred <preds.csv>
python -m src.vlm.tier0_baseline_infer --input_csv data/Test.csv --output_csv submissions/tier0_baseline_<timestamp>.csv
python -m src.vlm.validate_submission --submission submissions/tier0_baseline_<timestamp>.csv --sample_submission data/SampleSubmission.csv
```

`validate_submission.py` checks the ID set matches `SampleSubmission.csv`
exactly, no duplicates, and no empty/NaN `Target` values before upload.
Empty model outputs are replaced with the placeholder `[illegible]` rather
than left blank (a blank scores the same maximum-edit-distance penalty
either way — the placeholder just guarantees a well-formed CSV).

## Tier 1: Kraken, fine-tuned from CATMuS

Fixes a real bug in `starters/Kraken-OCR/inference.py`: every image here is
already a single pre-cropped text line, but that script still ran
`kraken.pageseg.segment()` (full-page layout analysis) on each one before
recognition, which produces garbage/empty output on a line crop — there's
no page layout to find. `src/kraken/infer.py` never calls `pageseg`; it
builds a synthetic single-line `kraken.containers.Segmentation` covering
the whole image and calls `kraken.rpred.rpred` directly. Two more bugs
found and fixed along the way (both verified against the installed
`kraken==6.0.3` and pinned in `tests/test_kraken_infer.py`): the
segmentation must be `type='baselines'`, not `'bbox'` (CATMuS is
baseline-trained; bbox-type input runs but produces garbled non-Latin
output), and polygon/baseline coordinates must be strictly `< (width,
height)` — using `width`/`height` themselves as the boundary's far corner
raises `"Line polygon outside of image bounds"`.

- `src/kraken/prepare_data.py` — converts `train_split.csv`/`val_split.csv`
  into Kraken's `path`-format training data (image + `.gt.txt` sidecar +
  manifest).
- `src/kraken/train.py` — wraps `ketos train` (subprocess), fine-tuning
  from the CATMuS pretrained model (Zenodo DOI `10.5281/zenodo.10592716`)
  with `--resize new` on the first run. Validates via `ketos train`'s
  explicit `-e/--evaluation-files` flag against `val_split.csv` — the same
  stratified split every tier uses, not a random re-split — so CER/WER are
  comparable across tiers.
- `src/kraken/infer.py` — the bug-fixed direct line recognizer described
  above. `--lm_rescore` opts into `src/kraken/lm_rescore.py`, a word-level
  lexicon + bigram rescoring pass over already-decoded output (only
  touches out-of-vocabulary words). **Not yet validated against a trained
  model** — a true lattice-level LM rescore would need kraken's internal
  per-timestep decode output, which `rpred()` doesn't expose without
  reaching into version-fragile internals; the word-level pass was the
  feasible alternative given no trained checkpoint existed yet to validate
  against either approach.

**Training runs on Colab, not locally** (`runners/colab/train_kraken.ipynb`
— self-contained, see `runners/colab/README.md` for upload/run
instructions). Locally, the bug fix was sanity-checked against the
zero-shot (not fine-tuned) CATMuS model on 5 `val_split` images — confirmed
non-empty, well-formed output with no crashes
(`experiments/tier1_kraken/infer_bugfix_sanity_check_zeroshot_preds.csv`);
quality is poor as expected for an unfine-tuned model on unseen handwriting.
Real CER/WER/final scores land in `experiments/tier1_kraken/metrics.json`
and the Results log below once training completes.

## Results log

| Date | Tier | Model | Local weighted CER | Local weighted WER | Local final score | Public LB score |
|------|------|-------|---------------------|---------------------|--------------------|------------------|
| 2026-07-09 | Tier 0 (zero-shot) | Qwen2-VL-2B-Instruct, 4-bit, no fine-tuning | 0.4329 | 0.6916 | 0.5622 | 0.35 |
| pending | Tier 1 (Kraken) | Kraken fine-tuned from CATMuS (`10.5281/zenodo.10592716`) | pending Colab training | pending | pending | not submitted yet |
