from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torchvision.models import mobilenet_v3_small
import onnxruntime as ort

from inference_bench import ort_backend
from inference_bench.onnx_export import export_model_to_onnx

from inference_bench import validation


@dataclass(frozen=True)
class ExportedModelFixture:
    """Artifacts shared by the ONNX Runtime integration tests."""

    model_path: Path
    input_tensor: torch.Tensor
    pytorch_output: torch.Tensor


@pytest.fixture(scope="module")
def exported_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> ExportedModelFixture:
    """Export one deterministic model instance for this test module."""

    # Seed the global PyTorch RNG before initializing the model so that
    # its randomly initialized weights are reproducible.
    torch.manual_seed(2026)

    # Avoid pretrained weights so the test remains independent of
    # network access and the Local torchvision model cache.
    model = mobilenet_v3_small(weights=None).cpu().eval()

    # Use a dedicated generator so the test input does not depend on
    # how many random values were consumed during model construction.
    generator = torch.Generator(device="cpu")
    generator.manual_seed(2027)
    input_tensor = torch.randn(
        (1, 3, 224, 224),
        generator=generator,
        dtype=torch.float32,
    )

    # Produce the reference output before exporting the exact same
    # in-memory model instance to ONNX.
    with torch.inference_mode():
        pytorch_output = model(input_tensor).detach().cpu()

    # A module-scoped temporary directory allows all tests in this
    # module to reuse one exported model without polluting the repository.
    model_path = (
        tmp_path_factory.mktemp("day3_ort")
        / "mobilenet_v3_small.onnx"
    )

    export_model_to_onnx(
        model=model,
        example_input=input_tensor,
        output_path=model_path,
        input_name="input",
        output_name="logits",
        opset_version=18,
    )

    return ExportedModelFixture(
        model_path=model_path,
        input_tensor=input_tensor,
        pytorch_output=pytorch_output,
    )


def test_create_cpu_session_uses_cpu_execution_provider_and_runtime_thread_defaults(
    exported_model: ExportedModelFixture,
) -> None:
    """Preserve the established CPU-only session policy and thread defaults."""

    session = ort_backend.create_cpu_session(
        exported_model.model_path
    )
    options = session.get_session_options()

    # Restricting the active provider list prevents execution from
    # being delegated to another provider such as Core ML.
    assert session.get_providers() == [
        "CPUExecutionProvider"
    ]

    # Verify the public model interface observed by ONNX Runtime.
    assert [value.name for value in session.get_inputs()] == [
        "input"
    ]
    assert [value.name for value in session.get_outputs()] == [
        "logits"
    ]

    # Zero is ONNX Runtime's sentinel for its runtime-selected thread
    # policy. It does not mean that the session executes with zero threads.
    assert options.intra_op_num_threads == 0
    assert options.inter_op_num_threads == 0

    # Graph nodes remain sequential while individual operators may use
    # the intra-op thread pool.
    assert options.execution_mode == (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )
    assert options.graph_optimization_level == (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )


def test_ort_cpu_output_matches_pytorch_cpu(
        exported_model: ExportedModelFixture,
) -> None:
    """Verify that the ORT CPU output is numerically consistent with the PyTorch CPU output."""

    # Create an inference session explicity restrcited to the
    # CPU Execution Provider.
    session = ort_backend.create_cpu_session(
        exported_model.model_path
    )

    # Execute the exported ONNX model using the exact input tensor
    # previously used for the PyTorch reference inference.
    ort_output = ort_backend.run_ort_inference(
        session,
        exported_model.input_tensor,
    )

    # Validate the expected model interface and reject invalid
    # numerical output before performing an error comparison.
    assert tuple(ort_output.shape) == (1, 1000)
    assert bool(torch.isfinite(ort_output).all())

    # Compute both numerical-error metrics and task-level top-1
    # agreement against the PyTorch reference output.
    comparison = validation.compare_classification_outputs(
        exported_model.pytorch_output,
        ort_output,
        rtol=1e-4,
        atol=1e-4,
    )

    # Confirm that both backends produce the same classification
    # decision for the current input.
    assert comparison.top1_match is True

    # Perform an element-wise comparison with detailed diagnostics
    # if the configured tolerance is exceeded.
    torch.testing.assert_close(
        ort_output,
        exported_model.pytorch_output,
        rtol=1e-4,
        atol=1e-4,
    )

    # Confirm that the persisted comparison summary agrees with
    # the strict element-wise assertion above.
    assert comparison.allclose is True


@pytest.mark.parametrize(
    "value",
    [0, -1, True, 1.5],
    ids=[
        "zero",
        "negative",
        "boolean",
        "non_integer",
    ],
)
def test_create_cpu_session_rejects_invalid_intra_op_thread_count(
    exported_model: ExportedModelFixture,
    value: object,
) -> None:
    """Reject values that cannot represent a positive thread count."""

    with pytest.raises(
        ValueError,
        match=(
            "intra_op_num_threads must be a positive integer or None"
        ),
    ):
        ort_backend.create_cpu_session(
            exported_model.model_path,
            intra_op_num_threads=value,
        )


def test_create_cpu_session_applies_explicit_intra_op_only(
    exported_model: ExportedModelFixture,
) -> None:
    """Apply the requested intra-op count without changing inter-op policy."""

    session = ort_backend.create_cpu_session(
        exported_model.model_path,
        intra_op_num_threads=2,
    )
    options = session.get_session_options()

    # The experiment changes only the selected independent variable.
    assert options.intra_op_num_threads == 2

    # Inter-op remains delegated to the ONNX Runtime default policy.
    assert options.inter_op_num_threads == 0

    # Existing execution and optimization policies remain unchanged.
    assert options.execution_mode == (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )
    assert options.graph_optimization_level == (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    assert session.get_providers() == [
        "CPUExecutionProvider"
    ]
