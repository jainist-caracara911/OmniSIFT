# OmniSIFT

OmniSIFT is a lightweight research implementation of audio-video token compression for Qwen2.5-Omni. It keeps the repository focused on the inference-time compression code: the modified Qwen2.5-Omni modeling file, the similarity-based pruning unit, and the Qwen Omni media preprocessing utilities.

## Repository Layout

```text
OmniSIFT/
├── omnisift/
│   ├── compression_units.py          # Similarity-based audio/video pruning
│   └── modeling_qwen2_5_omni.py      # Qwen2.5-Omni model with compression hooks
├── qwen-omni-utils/                  # Vendored media preprocessing utilities
└── evaluation/                       # Placeholder for evaluation scripts
```

## Installation

```bash
conda create -n omnisift python=3.10 -y
conda activate omnisift
pip install --upgrade pip

pip install -e .
pip install -e qwen-omni-utils
```

For GPU inference, install the PyTorch build that matches your CUDA runtime. FlashAttention is optional but recommended when your environment supports it:

```bash
pip install flash-attn --no-build-isolation
```

## Quick Start

```python
import torch
from transformers import AutoProcessor
from qwen_omni_utils import process_mm_info
from omnisift import Qwen2_5OmniForConditionalGeneration

model_path = "Qwen/Qwen2.5-Omni-7B"

processor = AutoProcessor.from_pretrained(model_path)
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
)

# Optional: tune compression ratios.
model.thinker.compression_config = {
    "rho_audio": 0.3,
    "rho_video": 0.7,
}

messages = [
    {
        "role": "user",
        "content": [
            {"type": "video", "video": "file:///path/to/video.mp4"},
            {"type": "audio", "audio": "file:///path/to/audio.wav"},
            {"type": "text", "text": "Describe the audio and video."},
        ],
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
audios, images, videos = process_mm_info(messages, use_audio_in_video=False)
inputs = processor(text=text, images=images, videos=videos, audio=audios, padding=True, return_tensors="pt")
inputs = inputs.to(model.device)

generated_ids, generated_audio = model.generate(**inputs)
response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
print(response[0])
```

## Compression Parameters

`rho_audio` controls the fraction of audio tokens removed within each chunk.
`rho_video` controls the fraction of video tokens removed from the selected spatial/temporal positions.

Lower values preserve more tokens. Higher values are faster but may reduce answer quality.

## Open Source Notes

This repository intentionally excludes model weights, datasets, experiment logs, generated caches, and local outputs. Download model weights from their upstream providers according to their licenses.

The modified Qwen2.5-Omni modeling file keeps its upstream Apache-2.0 license header. See [NOTICE](NOTICE) for attribution details.

## Citation

If this code is used for OmniZip/OmniSIFT research, please cite the corresponding paper or project release when it is available.

