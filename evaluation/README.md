# Evaluation

This directory provides the benchmark metadata and a unified inference entrypoint for evaluating OmniSIFT on audio-video benchmarks.

Only lightweight metadata is included in this repository. Benchmark videos are not included; download each dataset from its official source and pass the local video directory with `--video_base_dir`.

## Benchmarks

| Benchmark | Metadata | Expected media paths |
| --- | --- | --- |
| DailyOmni | `dailyomni/data.jsonl` | `<video_id>/<video_id>_video.mp4` |
| WorldSense | `worldsense/data.jsonl` | `videos_and_audios/...` |
| VideoMME | `videomme/data.jsonl` | `videos/data/...` |
| OmniVideoBench | `omnivideobench/data.jsonl` | `videos/...` |
| Video-SALMONN | `video_salmonn/data.jsonl` | `video/...` |

## Usage

Run inference with the shared entrypoint:

```bash
python evaluation/inference.py \
  --benchmark worldsense \
  --video_base_dir /path/to/WorldSense \
  --model_path dingyue1011/OmniSIFT-7B \
  --output_path outputs/worldsense.jsonl
```

Arguments:

- `--benchmark`: one of `dailyomni`, `worldsense`, `videomme`, `omnivideobench`, or `video_salmonn`.
- `--video_base_dir`: local directory that contains the benchmark videos.
- `--model_path`: Hugging Face model id or local checkpoint path.
- `--output_path`: JSONL file where predictions will be written.
- `--data_path`: optional metadata path. If omitted, the script uses `evaluation/<benchmark>/data.jsonl`.

The output JSONL keeps the original metadata fields and appends `prediction`, `answer`, and `judge`. `judge` is only computed for multiple-choice references; open-ended references use `null`.

For Video-SALMONN, use the official evaluation implementation for final scoring; this script only formats the metadata and generates OmniSIFT predictions.

The current inference script hard-codes video preprocessing to `max_frames=256` and `max_pixels=256*28*28`.

## Example

```bash
python evaluation/inference.py \
  --benchmark videomme \
  --video_base_dir /path/to/Video-MME/origin_data \
  --output_path outputs/videomme.jsonl
```

For large evaluations, use `--limit` for a quick smoke test before running the full benchmark.
