"""Contract tests for native inference backend adapters."""

import copy

import numpy as np
import pytest
import torch
from torchvision.models import mobilenet_v3_small

from inference_bench import backends
from inference_bench.onnx_export import export_model_to_onnx


INPUT_SHAPE = (1, 3, 224, 224)


@pytest.fixture(scope="module")
def model_case(tmp_path_factory):
    """Shared model weights, input, reference output, and an ONNX artifact."""
    torch.manual_seed(2026)

    # Leave the caller's model in training mode to detect in-place changes
    # made by an adapter. Random weights avoid downloads during testing.
    source = mobilenet_v3_small(weights=None).cpu().train()
    reference_model = copy.deepcopy(source).eval()

    # Decouple input generation from RNG consumption during model creation.
    generator = torch.Generator().manual_seed(2027)
    x = torch.randn(
        INPUT_SHAPE,
        generator=generator,
        dtype=torch.float32,
    )

    # Compute the reference without using any adapter under test.
    with torch.inference_mode():
        expected = reference_model(x)

    path = tmp_path_factory.mktemp("day4") / "model.onnx"

    # Isolate the reference model from exporter-side state changes.
    export_model_to_onnx(
        model=copy.deepcopy(reference_model),
        example_input=x,
        output_path=path,
        opset_version=18,
    )

    return source, x, expected, path


@pytest.fixture(params=["pytorch_cpu", "pytorch_mps", "ort_cpu"])
def backend(request, model_case):
    """Create a fresh adapter for each test case and backend parameter."""
    source, _, _, path = model_case

    if request.param == "pytorch_mps":
        if not torch.backends.mps.is_available():
            pytest.skip("MPS is unavailable")
        return backends.TorchBackend(source, device="mps")

    if request.param == "pytorch_cpu":
        return backends.TorchBackend(source, device="cpu")

    return backends.OrtCpuBackend(path)


def test_backend_preserves_native_execution_and_input_snapshot(
    backend,
    model_case,
):
    """Check native execution and results after caller-side input mutation."""
    source, x, expected, _ = model_case

    # Protect shared fixture input and exercise gradient-enabled input.
    caller_input = x.clone().requires_grad_(True)
    prepared = backend.prepare_input(caller_input)

    # Complete any input transfer before modifying the caller's tensor.
    backend.synchronize()

    # Prepared input is expected to retain the original values.
    with torch.no_grad():
        caller_input.zero_()

    # Exercise repeated execution with the same prepared input.
    for _ in range(2):
        native_output = backend.run(prepared)
        backend.synchronize()

        # Native data remains in the backend's format until to_cpu().
        if backend.name == "ort_cpu":
            assert isinstance(prepared, np.ndarray)
            assert isinstance(native_output, np.ndarray)
            assert backend.metadata()["providers"] == [
                "CPUExecutionProvider"
            ]
        else:
            device = "mps" if backend.name == "pytorch_mps" else "cpu"
            assert prepared.device.type == device
            assert native_output.device.type == device
            assert native_output.requires_grad is False

        result = backend.to_cpu(native_output)

        assert result.device.type == "cpu"
        assert result.dtype == torch.float32
        assert tuple(result.shape) == (1, 1000)
        assert bool(torch.isfinite(result).all())

        # Compare the full output against direct PyTorch CPU inference.
        torch.testing.assert_close(
            result,
            expected,
            rtol=1e-4,
            atol=1e-4,
        )

    # Preserve the caller's training flag and parameter placement.
    assert source.training is True
    assert all(p.device.type == "cpu" for p in source.parameters())


@pytest.mark.parametrize(
    ("bad_input", "error_type"),
    [
        (torch.zeros(INPUT_SHAPE, dtype=torch.float64), TypeError),
        (torch.zeros((2, 3, 224, 224)), ValueError),
    ],
    ids=["wrong_dtype", "wrong_shape"],
)
def test_backend_rejects_invalid_input(
    backend,
    bad_input,
    error_type,
):
    """Reject inputs that violate the FP32, fixed-shape contract."""
    with pytest.raises(error_type):
        backend.prepare_input(bad_input)


def test_run_once_works_with_each_backend(backend, model_case):
    """Verify the shared entry point against the CPU reference."""
    _, x, expected, _ = model_case

    # The parametrized fixture supplies each supported backend.
    result = backends.run_once(backend, x)

    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )


def test_run_once_obeys_lifecycle_order(model_case):
    """Verify lifecycle ordering while executing a real CPU model."""
    source, x, expected, _ = model_case
    events = []

    class RecordingBackend(backends.TorchBackend):
        "Recording lifecycle calls and delegate to the real implementation."

        def prepare_input(self, x):
            events.append("prepare")
            return super().prepare_input(x)

        def run(self, prepared):
            events.append("run")
            return super().run(prepared)

        def synchronize(self):
            events.append("sync")
            return super().synchronize()

        def to_cpu(self, output):
            events.append("to_cpu")
            return super().to_cpu(output)

    backend = RecordingBackend(source, device="cpu")
    result = backends.run_once(backend, x)

    # Check all order and count. CPU synchronization is a no-op;
    # this verifies orchestration, not MPS synchronization behavior.
    assert events == [
        "prepare",
        "sync",
        "run",
        "sync",
        "to_cpu",
    ]

    # Recording lifecycle calls must preserve the inference result.
    torch.testing.assert_close(
        result,
        expected,
        rtol=1e-4,
        atol=1e-4,
    )