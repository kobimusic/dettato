import torch

from dettato import pipeline


def test_default_device_prefers_cuda_then_mps_then_cpu(monkeypatch):
    for cuda, mps, want in ((True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")):
        monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps)
        assert pipeline.default_device() == want
