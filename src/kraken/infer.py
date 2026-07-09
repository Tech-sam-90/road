"""
Kraken OCR inference — direct line recognition (bug fix).

BUG in starters/Kraken-OCR/inference.py: every image in this dataset is
already a single pre-cropped text line, not a full page. That script still
called kraken.pageseg.segment() on each image before recognition — page
segmentation looks for multiple text-line regions inside a full page layout
(margins, columns, gaps between lines) and produces garbage or empty output
when fed a single line crop instead, since there's no page layout to find.
It only fell back to a single full-image line (create_fallback_segmentation)
when pageseg errored or returned nothing, which papers over the mismatch on
some images rather than fixing it for all of them.

This version never calls pageseg. It always builds the single full-image
line as a kraken.containers.Segmentation/BBoxLine (same object schema the
starter's own fallback used — confirmed against the installed kraken==6.0.3
API) and calls kraken.rpred.rpred directly against it.
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from tqdm import tqdm

from kraken.containers import BaselineLine, Segmentation
from kraken.lib.models import load_any
from kraken.rpred import rpred as kraken_rpred

try:
    from src.kraken.lm_rescore import build_lexicon_lm
except ImportError:
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.kraken.lm_rescore import build_lexicon_lm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

# Same term real transcribers use for unreadable text — matches the
# convention already established in src/vlm/tier0_baseline_infer.py.
EMPTY_PLACEHOLDER = "[illegible]"


def clean_output(text: str) -> str:
    text = str(text)
    for tag in ["assistant", "user", "<|assistant|>", "<|user|>"]:
        if tag in text:
            text = text.split(tag)[-1]
    return " ".join(text.split()).strip()


def otsu_threshold(image: Image.Image) -> int:
    array = np.asarray(image.convert("L"))
    hist, _ = np.histogram(array, bins=256, range=(0, 255))
    total = array.size
    sum_all = np.dot(np.arange(256), hist)
    sumB, wB, maximum, threshold = 0, 0, 0.0, 0
    for i in range(256):
        wB += hist[i]
        if wB == 0:
            continue
        wF = total - wB
        if wF == 0:
            break
        sumB += i * hist[i]
        mB, mF = sumB / wB, (sum_all - sumB) / wF
        between = wB * wF * (mB - mF) ** 2
        if between >= maximum:
            threshold, maximum = i, between
    return threshold


def binarize_image(image: Image.Image, threshold=None):
    gray = image.convert("L")
    if threshold is None:
        threshold = otsu_threshold(gray)
    binary = gray.point(lambda p: 255 if p > threshold else 0, mode="1")
    return binary, threshold


def direct_line_segmentation(image: Image.Image, text_direction: str = "horizontal-lr",
                              baseline_frac: float = 0.75) -> Segmentation:
    """The fix: treat the whole image as a single line, no pageseg call.

    CATMuS (like all modern kraken recognition models) is trained on
    baseline-type segmentation, not bbox — feeding it a BBoxLine runs but
    kraken warns "will likely result in severely degraded performance", and
    empirically it does (verified: garbage/gibberish output on real
    samples). So this builds a synthetic BaselineLine instead: since the
    image is already a tight single-line crop, the boundary is just the
    full image rectangle, and the baseline is a straight horizontal line
    near the bottom of the text (~75% down, leaving room for descenders)
    rather than the very bottom edge — kraken's line extraction uses the
    baseline position to orient/dewarp the crop, and an already-straight,
    already-cropped line needs no real dewarping, just a sane reference
    line.
    """
    # kraken requires all polygon/baseline coords to be strictly inside
    # [0, width) x [0, height) — using w/h themselves (not w-1/h-1) raises
    # "Line polygon outside of image bounds" (verified against
    # kraken.lib.segmentation's `pl.max(axis=0)[::-1] >= imshape` check).
    w, h = image.width - 1, image.height - 1
    y_baseline = int(h * baseline_frac)
    baseline = [(0, y_baseline), (w, y_baseline)]
    boundary = [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]
    line = BaselineLine(id="line_0", baseline=baseline, boundary=boundary)
    return Segmentation(
        type="baselines",
        imagename=None,
        text_direction=text_direction,
        script_detection=False,
        lines=[line],
    )


def infer_image(model, image_path, text_direction="horizontal-lr", pad=16, threshold=None,
                 bidi_reordering=True, binarize=True):
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        proc_img = binarize_image(img, threshold)[0] if binarize else img
        seg = direct_line_segmentation(proc_img, text_direction=text_direction)

        text_segments = []
        for rec in kraken_rpred(model, proc_img, seg, pad=pad, bidi_reordering=bidi_reordering):
            if getattr(rec, "prediction", None):
                text_segments.append(str(rec.prediction))

    return clean_output(" ".join(text_segments))


def run_inference(ids, model, image_dir, text_direction="horizontal-lr", pad=16, threshold=None,
                   bidi_reordering=True, binarize=True, lexicon_lm=None, limit=None):
    if limit:
        ids = ids[:limit]

    results = []
    fallback_count = 0
    for record_id in tqdm(ids, desc="kraken direct-line inference"):
        image_path = os.path.join(image_dir, f"{record_id}.jpg")
        pred = None
        if os.path.exists(image_path):
            try:
                pred = infer_image(
                    model, image_path, text_direction=text_direction, pad=pad,
                    threshold=threshold, bidi_reordering=bidi_reordering, binarize=binarize,
                )
            except Exception as exc:  # noqa: BLE001 - keep the run alive across a bad image
                print(f"[WARN] inference failed for {record_id}: {exc}")
                pred = None
        else:
            print(f"[WARN] no image found for ID {record_id}")

        if not pred:
            pred = EMPTY_PLACEHOLDER
            fallback_count += 1
        elif lexicon_lm is not None:
            pred = lexicon_lm.rescore_sentence(pred)

        results.append({"ID": record_id, "Target": pred})

    if fallback_count:
        print(f"[WARN] {fallback_count}/{len(ids)} predictions fell back to '{EMPTY_PLACEHOLDER}'")

    return pd.DataFrame(results, columns=["ID", "Target"])


def main():
    parser = argparse.ArgumentParser(description="Kraken direct line-recognition inference (no pageseg).")
    parser.add_argument("--model_path", required=True, help="Path to a .mlmodel file")
    parser.add_argument("--input_csv", required=True, help="CSV with an ID column (Test.csv or val_split.csv)")
    parser.add_argument("--base_image_dir", default=os.path.join(REPO_ROOT, "data", "images"))
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--device", default=None, help="cpu or cuda:0 — default: auto-detect")
    parser.add_argument("--pad", type=int, default=16)
    parser.add_argument("--text_direction", default="horizontal-lr")
    parser.add_argument("--threshold", type=int, default=None, help="Explicit Otsu binarization threshold override")
    parser.add_argument("--no_binarize", action="store_true", help="Skip binarization, feed the model raw RGB")
    parser.add_argument("--bidi_reordering", type=lambda x: str(x).lower() in ("1", "true", "yes"), default=True)
    parser.add_argument("--lm_rescore", action="store_true", help="Apply lexicon+bigram rescoring (see src/kraken/lm_rescore.py)")
    parser.add_argument("--lm_train_csv", default=os.path.join(REPO_ROOT, "data", "train_split.csv"))
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N rows (smoke testing)")
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[model] loading {args.model_path} on {device}")
    model = load_any(args.model_path, device=device)

    lexicon_lm = None
    if args.lm_rescore:
        print(f"[lm] building lexicon from {args.lm_train_csv}")
        lexicon_lm = build_lexicon_lm(args.lm_train_csv)

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    ids = df["ID"].astype(str).tolist()

    out_df = run_inference(
        ids, model, args.base_image_dir,
        text_direction=args.text_direction, pad=args.pad, threshold=args.threshold,
        bidi_reordering=args.bidi_reordering, binarize=not args.no_binarize,
        lexicon_lm=lexicon_lm, limit=args.limit,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"[done] wrote {len(out_df)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
