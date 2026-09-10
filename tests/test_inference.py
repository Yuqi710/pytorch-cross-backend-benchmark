"""Execution-contract tests for reference inference on PyTorch devices.

These tests validate each backend independently. Cross-backend numerical
consistency is intentionally deferred and documented in the experiment log.
"""

import pytest
import torch

from inference_bench.models import load_model
import inference_bench.inference as inference


def test_reference_inference_runs_on_cpu():
    # Keep model initialization and input generation reproducible.
    torch.manual_seed(0)
    model = load_model("mobilenet_v3_small", pretrained=False)
    inputs = torch.rand(1, 3, 224, 224)

    outputs = inference.run_reference_inference(
        model = model,
        inputs = inputs,
        device_name = "cpu",
    )

    assert outputs.shape == (1, 1000)
    # The reference inference normalizes backend outputs onto the CPU.
    assert outputs.device.type == "cpu"
    assert outputs.requires_grad is False
    assert torch.isfinite(outputs).all().item()


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_reference_inference_runs_on_mps():
    # Aviod network-dependent pretrained weights in hardware tests.
    torch.manual_seed(0)
    model = load_model("mobilenet_v3_small", pretrained=False)
    inputs = torch.rand(1, 3, 224, 224)

    outputs = inference.run_reference_inference(
        model = model,
        inputs = inputs,
        device_name = "mps",
    )

    assert outputs.shape == (1, 1000)
    # Computation runs on MPS; the result is copied back for comparison.
    assert outputs.device.type == "cpu"
    assert outputs.requires_grad is False
    assert torch.isfinite(outputs).all().item()