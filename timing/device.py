"""The torch device every model runs on: CUDA, else Apple GPU (MPS).

TIMING_DEVICE overrides the choice (cuda / mps / cpu). CPU is never picked on
its own: the models are far too slow there, so it has to be asked for.
"""
import os


def device():
    import torch
    chosen = os.environ.get("TIMING_DEVICE", "").strip().lower()
    if chosen:
        return chosen
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def require_gpu(what):
    """Return the device, or refuse like the CUDA-only build did when there is no GPU."""
    chosen = device()
    if chosen == "cpu" and not os.environ.get("TIMING_DEVICE"):
        raise RuntimeError(f"{what}需要 GPU（NVIDIA CUDA 或 Apple 芯片）。")
    return chosen


def empty_cache():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def reset_peak_memory():
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gib():
    """Peak tensor memory of this process; MPS only reports what is allocated now."""
    import torch
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 2**30
    if torch.backends.mps.is_available():
        return torch.mps.driver_allocated_memory() / 2**30
    return None
