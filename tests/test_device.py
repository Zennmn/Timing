import pytest
import torch

from timing import device as device_module
from timing.device import device, require_gpu


@pytest.fixture
def no_override(monkeypatch):
    monkeypatch.delenv("TIMING_DEVICE", raising=False)


def test_cuda_wins_over_mps(monkeypatch, no_override):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert device() == "cuda"


def test_apple_gpu_is_used_without_cuda(monkeypatch, no_override):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert require_gpu("打轴") == "mps"


def test_no_gpu_refuses_instead_of_crawling_on_cpu(monkeypatch, no_override):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="主唱分离需要 GPU"):
        require_gpu("主唱分离")


def test_cpu_runs_only_when_asked_for(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setenv("TIMING_DEVICE", "cpu")
    assert require_gpu("主唱分离") == "cpu"


def test_override_picks_the_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setenv("TIMING_DEVICE", " MPS ")
    assert device_module.device() == "mps"
