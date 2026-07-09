"""
Tier 0 zero-shot baseline: run the base (non-fine-tuned) Qwen2-VL-2B-Instruct
model directly against line-crop images. This exists to de-risk the
submission pipeline (CSV format, ID coverage, scorer wiring) before we
invest time in real fine-tuning — it is not expected to score well.

Adapted from starters/VLM/inference.py's clean_output/load_image/predict
logic (kept byte-for-byte where it doesn't need to change), minus the PEFT
adapter loading step since there's no fine-tuned checkpoint yet.
"""

import argparse
import os

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(REPO_ROOT, "data")
IMAGE_DIR = os.path.join(DATA_DIR, "images")

MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"

# Written into the Target cell whenever the model produces nothing usable
# (empty generation, or after stripping the model regurgitates only
# whitespace/role tags) so the CSV never has a blank/NaN cell — Zindi's
# scoring already treats a blank the same as "" (full edit distance
# against the reference), so this doesn't change scoring, it just
# guarantees the submission is well-formed. A term real transcribers use
# for unreadable text, rather than an arbitrary filler.
EMPTY_PLACEHOLDER = "[illegible]"

# Unchanged from starters/VLM/{trainer,inference}.py.
OCR_PROMPT = (
    "Transcribe EXACTLY as seen. Preserve line breaks, punctuation, and spacing. "
    "Do not add explanations. Return verbatim transcription. No paraphrasing. No normalization."
)


def clean_output(text: str) -> str:
    text = str(text)
    for tag in ["assistant", "user", "<|assistant|>", "<|user|>"]:
        if tag in text:
            text = text.split(tag)[-1]
    return " ".join(text.split()).strip()


def load_image(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    aspect = w / h
    if aspect > 10:
        img.thumbnail((2048, 384))
    elif aspect > 5:
        img.thumbnail((2048, 512))
    else:
        img.thumbnail((1536, 768))
    return img


def load_model(model_id=MODEL_ID):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[model] loading {model_id} on {device}")

    quant_config = None
    if device == "cuda":
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()

    # Cap vision tokens — these are thin line crops, not full pages, so we
    # don't need Qwen2-VL's default max_pixels headroom, and capping it
    # keeps peak VRAM predictable on a 4GB card.
    processor = AutoProcessor.from_pretrained(
        model_id,
        trust_remote_code=True,
        min_pixels=256 * 28 * 28,
        max_pixels=1024 * 28 * 28,
    )
    return model, processor


def predict_one(model, processor, image, max_new_tokens=64, num_beams=1):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": OCR_PROMPT},
                {"type": "image", "image": image},
            ],
        }
    ]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
        )

    decoded = processor.batch_decode(output, skip_special_tokens=True)[0]
    return clean_output(decoded)


def run_inference(ids, model, processor, image_dir=IMAGE_DIR, max_new_tokens=64, num_beams=1, limit=None):
    if limit:
        ids = ids[:limit]

    results = []
    fallback_count = 0
    for record_id in tqdm(ids, desc="tier0 zero-shot inference"):
        image_path = os.path.join(image_dir, f"{record_id}.jpg")
        pred = None
        if os.path.exists(image_path):
            try:
                image = load_image(image_path)
                pred = predict_one(model, processor, image, max_new_tokens=max_new_tokens, num_beams=num_beams)
            except Exception as exc:  # noqa: BLE001 - keep the run alive across a bad image
                print(f"[WARN] inference failed for {record_id}: {exc}")
                pred = None
        else:
            print(f"[WARN] no image found for ID {record_id}")

        if not pred:
            pred = EMPTY_PLACEHOLDER
            fallback_count += 1

        results.append({"ID": record_id, "Target": pred})

    if fallback_count:
        print(f"[WARN] {fallback_count}/{len(ids)} predictions fell back to '{EMPTY_PLACEHOLDER}'")

    return pd.DataFrame(results, columns=["ID", "Target"])


def main():
    parser = argparse.ArgumentParser(description="Tier 0 zero-shot Qwen2-VL baseline inference")
    parser.add_argument("--input_csv", required=True, help="CSV with an ID column (Test.csv or val_split.csv)")
    parser.add_argument("--output_csv", required=True, help="Where to write ID,Target predictions")
    parser.add_argument("--model_id", default=MODEL_ID)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N rows (smoke testing)")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, encoding="utf-8-sig")
    ids = df["ID"].astype(str).tolist()

    model, processor = load_model(args.model_id)
    out_df = run_inference(
        ids, model, processor,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        limit=args.limit,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8")
    print(f"[done] wrote {len(out_df)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
