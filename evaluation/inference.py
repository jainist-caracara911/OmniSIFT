#!/usr/bin/env python
"""Run OmniSIFT inference on bundled evaluation metadata."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

CHOICES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
MAX_FRAMES = 256
VIDEO_MAX_PIXELS = 256 * 28 * 28


def load_records(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    if text[0] in "[{":
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("annotations", "data", "items", "examples"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
            return [data]
        if data is not None:
            raise TypeError(f"Unsupported JSON root type in {path}: {type(data)!r}")

    records = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def resolve_media_path(media_path: str, media_root: str = "") -> str:
    path = Path(media_path)
    if path.is_absolute() or not media_root:
        return str(path)
    return str(Path(media_root) / path)


def first_conversation_value(item: dict[str, Any], role: str) -> str:
    for turn in item.get("conversations", []):
        if turn.get("from") == role:
            return str(turn.get("value", ""))
    return ""


def normalize_item(item: dict[str, Any], benchmark: str, video_base_dir: str) -> dict[str, Any]:
    prompt = item.get("prompt") or item.get("question") or item.get("Question") or first_conversation_value(item, "human")
    answer = item.get("answer") or item.get("Answer") or first_conversation_value(item, "gpt")

    if benchmark == "dailyomni" and "video_id" in item and "video_path" not in item:
        video_id = str(item["video_id"])
        video_path = f"{video_id}/{video_id}_video.mp4"
    elif "video_path" in item:
        video_path = str(item["video_path"])
    elif "videos" in item:
        video_path = str(item["videos"])
    elif "video" in item:
        video_path = str(item["video"])
    else:
        raise KeyError(f"No video field found in item keys: {sorted(item.keys())}")

    prompt = str(prompt).replace("<image>\n", "").replace("<video>\n", "")
    return {
        "raw": item,
        "prompt": prompt,
        "answer": str(answer) if answer is not None else "",
        "video_path": resolve_media_path(video_path, video_base_dir),
    }


def parse_choice(text: str) -> str:
    if not text:
        return ""
    cleaned = text.strip().upper()
    for token in CHOICES:
        if cleaned == token or cleaned.startswith(f"{token}.") or cleaned.startswith(f"{token}:"):
            return token
    for token in CHOICES:
        if f" {token} " in f" {cleaned} ":
            return token
    return cleaned[:1]


def parse_reference_choice(text: str) -> str:
    cleaned = text.strip().upper()
    if cleaned in CHOICES:
        return cleaned
    if 1 < len(cleaned) <= 3 and cleaned[0] in CHOICES and cleaned[1:] in {".", ":", ")"}:
        return cleaned[0]
    return ""


def build_messages(video_path: str, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": video_path,
                    "max_frames": MAX_FRAMES,
                    "max_pixels": VIDEO_MAX_PIXELS,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def generate(model, processor, video_path: str, prompt: str, max_new_tokens: int) -> str:
    import torch
    from qwen_omni_utils import process_mm_info

    messages = build_messages(video_path, prompt)
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=True)

    if hasattr(model, "thinker") and videos is not None and len(videos) > 0:
        model.thinker.nframes = videos[0].shape[0]

    inputs = processor(
        text=text,
        audio=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        use_audio_in_video=True,
    )
    inputs = inputs.to(model.device).to(model.dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            use_audio_in_video=True,
            return_audio=False,
            do_sample=False,
            max_new_tokens=max_new_tokens,
        )

    output_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
    return processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()


def default_data_path(benchmark: str) -> Path:
    return Path(__file__).resolve().parent / benchmark / "data.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSIFT benchmark inference")
    parser.add_argument("--benchmark", required=True, help="Benchmark subdirectory under evaluation/.")
    parser.add_argument("--data_path", type=str, default=None, help="Defaults to evaluation/<benchmark>/data.jsonl.")
    parser.add_argument("--video_base_dir", type=str, default="", help="Root directory for relative media paths.")
    parser.add_argument("--model_path", type=str, default="dingyue1011/OmniSIFT-7B")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rho_audio", type=float, default=0.3)
    parser.add_argument("--rho_video", type=float, default=0.7)
    args = parser.parse_args()

    data_path = Path(args.data_path) if args.data_path else default_data_path(args.benchmark)
    records = load_records(data_path)
    if args.limit is not None:
        records = records[: args.limit]

    from omnisift import Qwen2_5OmniForConditionalGeneration
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model_path)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    if hasattr(model, "thinker"):
        model.thinker.compression_config = {"rho_audio": args.rho_audio, "rho_video": args.rho_video}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as fout:
        for idx, item in enumerate(records):
            sample = normalize_item(item, args.benchmark, args.video_base_dir)
            prediction = generate(model, processor, sample["video_path"], sample["prompt"], args.max_new_tokens)
            answer = sample["answer"]
            answer_choice = parse_reference_choice(answer)
            result = dict(sample["raw"])
            result.update(
                {
                    "video_path": sample["video_path"],
                    "prompt": sample["prompt"],
                    "prediction": prediction,
                    "answer": answer,
                    "judge": parse_choice(prediction) == answer_choice if answer_choice else None,
                }
            )
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")
            fout.flush()
            print(f"[{idx + 1}/{len(records)}] {sample['video_path']}")


if __name__ == "__main__":
    main()
