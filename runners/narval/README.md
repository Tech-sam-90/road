# Narval runner

Workflow for training/inference on Narval (Digital Research Alliance of
Canada, Slurm-scheduled). Compute nodes have **no internet** — every network
call (pip installs not in the wheelhouse, HF model downloads, the CATMuS
Kraken model) happens on a login node ahead of time and is read from local
cache afterwards.

## 1. One-time setup (login node)

```bash
ssh <you>@narval.alliancecan.ca
git clone <your-repo-url> ~/road-barbados-htr-checkout   # or rsync it over
cd ~/road-barbados-htr-checkout/runners/narval
./setup_env.sh
```

`setup_env.sh`:
- picks the newest available `StdEnv` / `python` / `cuda` / `cudnn` modules
  by actually running `module avail` on the node you're on — it doesn't
  trust a hardcoded version list, since what's installed changes over time
  and we have no way to check from outside the cluster (see the module
  versions caveat in the top-level report for this session)
- if your checkout isn't already under `$PROJECT`, copies it there (never
  `$HOME` — small quota, not meant for code + multi-GB venvs)
- creates three virtualenvs under `$PROJECT/road-barbados-htr/venvs/`:
  `shared/`, `kraken/`, `vlm/` — kraken and the transformers/Qwen stack pin
  conflicting torch versions, same reason they're split locally
- for every package in each `requirements.txt`, tries
  `pip install --no-index <pkg>` against the Alliance wheelhouse first
  (hardware-optimized builds, works later with no network), falls back to a
  normal `pip install` (needs this login node's internet) otherwise
- prints a summary of what came from the wheelhouse vs. PyPI vs. failed

Re-run it any time `requirements.txt` changes.

## 2. One-time model prefetch (login node)

```bash
./prefetch_models.sh
# or, for a smaller backbone if your allocation's GPUs are memory-limited:
QWEN_MODEL_ID=Qwen/Qwen2.5-VL-3B-Instruct ./prefetch_models.sh
```

Downloads the Qwen2.5-VL backbone (`huggingface_hub.snapshot_download`) and
the CATMuS Kraken pretrained model (`kraken get`, then copied to a fixed
path) into `$PROJECT/road-barbados-htr/model_cache/`. Writes
`.env.narval` (gitignored — see `.env.narval.example` for the shape) with
`HF_HOME`, `TRANSFORMERS_CACHE`, `HF_HUB_OFFLINE=1`, `XDG_DATA_HOME`, and the
resolved model paths. The `submit_*.sb` scripts source this automatically —
compute nodes never touch the network.

## 3. Submit jobs

```bash
sbatch submit_kraken_train.sb
sbatch submit_vlm_train.sb
sbatch submit_vlm_infer.sb   # after a VLM checkpoint exists
```

Edit `--account=<FILL_IN_YOUR_ALLOCATION>` in each `.sb` file to your
allocation first (`sshare -U` or `sacctmgr show associations user=$USER`
lists what you have access to).

Resource requests (adjust as needed — see comments in each file):

| Job | GPU | CPUs | Mem | Time |
|---|---|---|---|---|
| `submit_kraken_train.sb` | 1 | 6 | 32G | 6h |
| `submit_vlm_train.sb` | 1 | 8 | 64G | 2d (max 7d) |
| `submit_vlm_infer.sb` | 1 | 4 | 32G | 3h |

Check the queue:

```bash
squeue -u $USER      # or: sq   (Alliance alias for the same)
```

## 4. Checkpointing / resuming

Jobs queue, can be preempted, and are capped at 7 days — training must
survive being killed mid-run. All three `submit_*.sb` scripts:

- write checkpoints to `$SCRATCH_REPO/experiments/<tier>/checkpoints`
  (fast, but purged after inactivity — never the durable copy)
- on start, look for an existing checkpoint there and resume from it
  (`--load_model <path> --load_hyper_parameters` for kraken;
  `--resume_from_checkpoint <path>` for the VLM, the standard HF Trainer
  flag) — falling back to the prefetched CATMuS/Qwen weights if nothing's
  there yet
- mirror `$SCRATCH_REPO/experiments/<tier>/checkpoints` back to
  `$PROJECT_REPO/experiments/<tier>/checkpoints` (durable) every 30 minutes
  during VLM training, and once at the end for both — plus immediately on a
  `SIGUSR1` sent ~2 minutes before the Slurm time limit hits
  (`--signal=B:USR1@120`), so a job that runs out of time doesn't lose
  progress since the last periodic sync

To resume a killed/requeued job, just `sbatch` the same script again — it
finds the latest checkpoint under `$SCRATCH_REPO` (or, if scratch got
purged, whatever's newest under `$PROJECT_REPO`; copy it back to
`$SCRATCH_REPO/experiments/<tier>/checkpoints` first if so) and continues.

## 5. Pull results

Everything durable lands under `$PROJECT/road-barbados-htr/`:
- `experiments/<tier>/checkpoints/` — model checkpoints
- `experiments/<tier>/logs/` — training logs (kraken)
- `submissions/` — timestamped `submission_*.csv` from inference runs

`rsync` or `scp` those back to your machine, or point the next Colab/local
session at the same Drive-synced copy if you're mirroring `$PROJECT` there.
