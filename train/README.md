# Training OmniSIFT

This directory contains the training entrypoint for OmniSIFT. Training is built on top of an external [ms-swift](https://github.com/modelscope/ms-swift) checkout, which should be installed separately. The ms-swift source tree is not included in this repository.

## Files

- `train.sh`: a parameterized DeepSpeed launch script for full-parameter SFT.
- `README.md`: setup notes for connecting OmniSIFT with an external ms-swift checkout.

Training data, model checkpoints, DeepSpeed configs, and media files are not distributed with this repository. Provide their local paths through environment variables when launching the script.

## ms-swift Integration

OmniSIFT reuses the ms-swift SFT pipeline, but the Qwen2.5-Omni model class must be replaced with the OmniSIFT implementation. In your external ms-swift checkout, edit:

```text
swift/llm/model/model/qwen.py
```

Inside `get_model_tokenizer_qwen2_5_omni`, import `Qwen2_5OmniForConditionalGeneration` from `omnisift` and set it as `automodel_class`:

```python
def get_model_tokenizer_qwen2_5_omni(model_dir, *args, **kwargs):
    from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniConfig
    from omnisift import Qwen2_5OmniForConditionalGeneration
    from qwen_omni_utils import vision_process

    kwargs["automodel_class"] = kwargs["automodel_class"] or Qwen2_5OmniForConditionalGeneration
    processor = Qwen2_5OmniProcessor.from_pretrained(model_dir, trust_remote_code=True)
    kwargs["tokenizer"] = processor.tokenizer
```

Keep the rest of the original ms-swift function unchanged unless your local ms-swift version requires additional adjustments. `train.sh` adds both this repository and `MS_SWIFT_ROOT` to `PYTHONPATH`, so the `omnisift` import can resolve without copying ms-swift into this repository.

## Run

Set the required paths and launch from the OmniSIFT repository root:

```bash
MS_SWIFT_ROOT=/path/to/ms-swift \
MODEL_PATH=/path/to/qwen2.5-omni-or-initialized-omnisift \
DATASET_JSONL=/path/to/train.jsonl \
DS_CONFIG=/path/to/zero2.json \
bash train/train.sh
```

Optional settings:

```bash
OMNISIFT_ROOT=/path/to/OmniSIFT
NPROC_PER_NODE=8
HOSTFILE=/path/to/hostfile
OUTPUT_DIR=output
EXP_NAME=omnisift_sft
GLOBAL_BATCH_SIZE=128
MAX_LENGTH=32768
LEARNING_RATE=1e-5
NUM_TRAIN_EPOCHS=1
SPLIT_DATASET_RATIO=0
DEEPSPEED_ENV_FILE=/path/to/.deepspeed_env
```
