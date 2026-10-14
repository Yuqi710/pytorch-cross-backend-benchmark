"""Integration tests for latency sampling with real inference backends."""

import copy

import pytest
import torch
from torch import nn

from inference_bench.backends import OrtCpuBackend, TorchBackend, run_once
from inference_bench.benchmark import BenchmarkConfig, collect_latency_samples
from inference_bench.onnx_export import export_model_to_onnx


@pytest.fixture(scope="module")
def case(tmp_path_factory):
    """Create a shared model, input, reference output, and ONNX artifact.

    A small randomly initialized model keeps the test fast and avoids
    pretrained-weight downloads. Module scope allows all backends cases
    to reuse the same weights, input values, and exported model.
    """
    torch.manual_seed(2026)
    # Shape flow:
    # [1, 3, 8, 8] -> [1, 3, 1, 1] -> [1, 3] -> [1, 3].
    model = nn.Sequential(
        nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        nn.Flatten(start_dim=1, end_dim=-1),
        nn.Linear(in_features=3, out_features=3, bias=True),
    ).eval()
    x = torch.randn((1, 3, 8, 8))
    with torch.inference_mode():
        expected = model(x).clone()
    path = tmp_path_factory.mktemp("benchmark") / "small.onnx"
    export_model_to_onnx(copy.deepcopy(model), x, path, opset_version=18)
    return model, x, expected, path


@pytest.mark.parametrize("name", ["pytorch_cpu", "pytorch_mps", "ort_cpu"])
def test_sampler_executes_real_backends_and_preserves_correctness(name, case):
    """Verify sample validity and post-sampling numerical correctness.

    This is a short integration check, not a stable performance measurement.
    Numerical validation uses a fresh inference call after sampling; it does
    not inspect every output produced within the measured intervals.
    """
    model, x, expected, path = case
    if name == "pytorch_mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    if name == "ort_cpu":
        backend = OrtCpuBackend(path)
    else:
        device = "mps" if name == "pytorch_mps" else "cpu"
        backend = TorchBackend(model, device=device, input_shape=(1, 3, 8, 8))

    # Exercise real preparation, execution, and synchronizaiton with a small
    # number of calls suitable for integration testing.
    samples = collect_latency_samples(
        backend, x, BenchmarkConfig(warmup=2, iterations=5)
    )
    # Warm-up calls must not contribute to the measured sample count.
    assert len(samples) == 5
    assert all(type(value) is int and value > 0 for value in samples)

    # Execute one more after sampling and compare the CPU-normalized output
    # with the independently computed PyTorch reference.
    torch.testing.assert_close(
        run_once(backend, x), expected, rtol=1e-4, atol=1e-4
    )
