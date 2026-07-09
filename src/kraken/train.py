"""
Fine-tunes the CATMuS pretrained Kraken model on our line-crop data via
`ketos train`, shelling out to the CLI (like the starter does) rather than
kraken's internal training API, since the CLI is the more stable public
surface across kraken releases.

Trains on --train_csv, validates on --val_csv explicitly via ketos's
-e/--evaluation-files flag (confirmed against the installed kraken==6.0.3:
"File(s) with paths to evaluation data. Overrides the -p parameter") — so
every tier evaluates against the SAME stratified splits from
src/data/make_splits.py, not a random resplit, which is what makes
CER/WER comparable across tiers.

CLI contract matches runners/narval/submit_kraken_train.sb and
runners/colab/train_kraken.ipynb, which both invoke this script directly:
--config/--train_csv/--val_csv/--base_image_dir/--output/--log_dir/
--load_model/--load_hyper_parameters[/--resize/--epochs/--batch_size/--device].

Not run against real training locally — training happens on Colab (GPU
there is far more useful for this than a 4GB local card, and the user asked
to keep it there). This script is exercised here only via --dry_run, which
builds both manifests and prints the exact `ketos` command without invoking
it, to catch wiring bugs before a real run.
"""

import argparse
import os
import subprocess
import sys

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.kraken.prepare_data import build_manifest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))


def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    return {}


def build_ketos_command(
    train_manifest, val_manifest, output, log_dir, device, seed,
    load_model=None, load_hyper_parameters=False, resize=None,
    epochs=-1, batch_size=16, workers=None, ketos_bin="ketos",
):
    if workers is None:
        # ketos defaults to 1 dataloader worker, which leaves the GPU idle
        # waiting on CPU-bound image preprocessing (confirmed in a real
        # run: kraken's own warning flagged this exact bottleneck).
        workers = min(os.cpu_count() or 4, 8)

    cmd = [
        ketos_bin,
        "--device", device,
        "--seed", str(seed),
        "--workers", str(workers),
        "train",
        "-t", train_manifest,
        "-e", val_manifest,
        "-f", "path",
        "-o", output,
        "-N", str(epochs),
        "-B", str(batch_size),
        "-F", "1",  # report/checkpoint every epoch (ketos default anyway) — explicit so it's not silently changed later
        "--log-dir", log_dir,
    ]
    if load_model:
        cmd.extend(["-i", load_model])
        if load_hyper_parameters:
            cmd.append("--load-hyper-parameters")
        if resize:
            cmd.extend(["--resize", resize])
    return cmd


def _stream_subprocess(cmd):
    """Runs cmd, relaying its output to our own stdout line-by-line with an
    immediate flush after each line.

    subprocess.run(cmd, check=True) alone looked completely frozen in
    practice (confirmed: a real run sat silent long enough to get
    KeyboardInterrupt'd) — the classic cause is that ketos's own stdout
    isn't a real terminal when piped through a subprocess, so CPython
    switches from line-buffered to fully block-buffered output and doesn't
    flush until several KB accumulate or the process exits. PYTHONUNBUFFERED
    forces ketos's own process to flush immediately regardless of whether
    its stdout is a TTY; the manual flush=True below then makes sure WE
    relay it immediately too instead of adding another buffering layer on
    top.
    """
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, env=env,
    )
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
    process.stdout.close()
    returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def run_training(
    config_path=None, train_csv=None, val_csv=None, base_image_dir=None,
    output=None, log_dir=None, load_model=None, load_hyper_parameters=False,
    resize=None, epochs=None, batch_size=None, workers=None, device=None, seed=42,
    train_data_dir=None, val_data_dir=None, ketos_bin="ketos", dry_run=False,
):
    cfg = load_config(config_path)
    cfg_train = cfg.get("training", {})

    train_csv = train_csv or cfg_train.get("train_csv", os.path.join(REPO_ROOT, "data", "train_split.csv"))
    val_csv = val_csv or cfg_train.get("val_csv", os.path.join(REPO_ROOT, "data", "val_split.csv"))
    base_image_dir = base_image_dir or cfg_train.get("base_image_dir", os.path.join(REPO_ROOT, "data", "images"))
    output = output or cfg_train.get("output", os.path.join(REPO_ROOT, "experiments", "tier1_kraken", "checkpoints", "kraken_model"))
    log_dir = log_dir or cfg_train.get("log_dir", os.path.join(REPO_ROOT, "experiments", "tier1_kraken", "logs"))
    load_model = load_model if load_model is not None else cfg_train.get("load_model")
    epochs = epochs if epochs is not None else cfg_train.get("epochs", -1)
    batch_size = batch_size if batch_size is not None else cfg_train.get("batch_size", 16)
    workers = workers if workers is not None else cfg_train.get("workers")
    train_data_dir = train_data_dir or cfg_train.get(
        "train_data_dir", os.path.join(REPO_ROOT, "experiments", "tier1_kraken", "kraken_train_data"))
    val_data_dir = val_data_dir or cfg_train.get(
        "val_data_dir", os.path.join(REPO_ROOT, "experiments", "tier1_kraken", "kraken_val_data"))

    if resize is None:
        # Fine-tuning from a pretrained model with a different alphabet needs
        # the output layer resized to our data on the FIRST run only; a
        # resumed run (load_hyper_parameters=True) already has a codec
        # that matches, and resizing again would be wrong.
        resize = "new" if (load_model and not load_hyper_parameters) else None

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print(f"[prepare] building train manifest from {train_csv}")
    train_manifest = build_manifest(train_csv, base_image_dir, train_data_dir)
    print(f"[prepare] building val manifest from {val_csv}")
    val_manifest = build_manifest(val_csv, base_image_dir, val_data_dir)

    if device is None:
        try:
            import torch
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    cmd = build_ketos_command(
        train_manifest, val_manifest, output, log_dir, device, seed,
        load_model=load_model, load_hyper_parameters=load_hyper_parameters,
        resize=resize, epochs=epochs, batch_size=batch_size, workers=workers, ketos_bin=ketos_bin,
    )

    print("[train] command:", " ".join(cmd), flush=True)
    if dry_run:
        print("[train] --dry_run set, not invoking ketos")
        return cmd

    _stream_subprocess(cmd)
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Kraken on our line-crop data via ketos train.")
    parser.add_argument("--config", default=os.path.join(SCRIPT_DIR, "config.yaml"))
    parser.add_argument("--train_csv", default=None)
    parser.add_argument("--val_csv", default=None)
    parser.add_argument("--base_image_dir", default=None)
    parser.add_argument("--output", default=None, help="Output model prefix (ketos -o)")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--load_model", default=None, help="Pretrained/checkpoint .mlmodel to continue from (ketos -i)")
    parser.add_argument("--load_hyper_parameters", action="store_true")
    parser.add_argument("--resize", default=None, choices=["add", "union", "both", "new", "fail"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None,
                         help="Dataloader worker processes (ketos --workers). Default: min(cpu_count, 8) — "
                              "ketos itself defaults to 1, which bottlenecks the GPU on CPU-bound image preprocessing.")
    parser.add_argument("--device", default=None, help="cpu or cuda:0 — default: auto-detect")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry_run", action="store_true", help="Build manifests and print the ketos command without running it")
    args = parser.parse_args()

    run_training(
        config_path=args.config, train_csv=args.train_csv, val_csv=args.val_csv,
        base_image_dir=args.base_image_dir, output=args.output, log_dir=args.log_dir,
        load_model=args.load_model, load_hyper_parameters=args.load_hyper_parameters,
        resize=args.resize, epochs=args.epochs, batch_size=args.batch_size, workers=args.workers,
        device=args.device, seed=args.seed, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
