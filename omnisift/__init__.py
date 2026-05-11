__all__ = [
    "Qwen2_5OmniForConditionalGeneration",
    "similarity_pruning",
]


def __getattr__(name):
    if name == "Qwen2_5OmniForConditionalGeneration":
        from .modeling_qwen2_5_omni import Qwen2_5OmniForConditionalGeneration

        return Qwen2_5OmniForConditionalGeneration
    if name == "similarity_pruning":
        from .compression_units import similarity_pruning

        return similarity_pruning
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
