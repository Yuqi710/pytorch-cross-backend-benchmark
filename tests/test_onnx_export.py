from pathlib import Path

import onnx
import pytest
import torch
from torchvision.models import mobilenet_v3_small

from inference_bench import onnx_export


# Use a fixed NCHW input shape for the Day 2 static-shape export contract.
INPUT_SHAPE = (1, 3, 224, 224)


def _fixed_shape(value_info: onnx.ValueInfoProto) -> tuple[int, ...]:
    """Extract concrete tensor dimensions from ONNX value information."""

    return tuple(
        int(dimension.dim_value)
        for dimension in value_info.type.tensor_type.shape.dim
    )


@pytest.fixture(scope="module")
def exported_model_path(tmp_path_factory: pytest.TempdirFactory) -> Path:
    """Export one ONNX model shared by all tests in this module."""

    output_path = (
        tmp_path_factory.mktemp("onnx")
        / "mobilenet_v3_small_fp32_opset18.onnx"
    )

    # Avoid network access and cached pretrained weights. This test validates
    # export behavior and graph structure rather than model accuracy.
    model = mobilenet_v3_small(weights=None).eval()

    # The example tensor defines the fixed input shape captured during export.
    example_input = torch.zeros(INPUT_SHAPE, dtype=torch.float32)

    return onnx_export.export_model_to_onnx(
        model=model,
        example_input=example_input,
        output_path=output_path,
        input_name="input",
        output_name="logits",
        opset_version=18,
    )


def test_export_creates_valid_fixed_shape_onnx_model(
        exported_model_path: Path,
) -> None:
    """Verify the exported file and its public ONNX graph contract."""

    # Confirm that the exporter created a non-empty model artifact.
    assert exported_model_path.is_file()
    assert exported_model_path.stat().st_size > 0

    # Deserialize the model and validate it against the ONNX specification.
    model = onnx.load(str(exported_model_path))
    onnx.checker.check_model(model)

    # Collect the public graph inputs and outputs by name and shape.
    graph_inputs = {
        value.name : _fixed_shape(value)
        for value in model.graph.input
    }

    graph_outputs = {
        value.name : _fixed_shape(value)
        for value in model.graph.output
    }

    # Normalize the empty default-domain name to "ai.onnx" for readability.
    opset_version = {
        opset.domain or "ai.onnx": int(opset.version)
        for opset in model.opset_import
    }

    # Verify the fixed-shape model interface and target opset.
    assert graph_inputs == {"input": INPUT_SHAPE}
    assert graph_outputs == {"logits": (1, 1000)}
    assert opset_version["ai.onnx"] == 18

    # Confirm that the graph contains computation nodes and embedded weights.
    assert len(model.graph.node) > 0
    assert len(model.graph.initializer) > 0


def test_inspect_onnx_model_reports_graph_structure(
        exported_model_path: Path,
) -> None:
    """Verify that ONNX graph inspection reports the expected model contract."""

    # Inspect the exported model and collect its interface, storage, and
    # operator-level graph metadata.
    summary = onnx_export.inspect_onnx_model(exported_model_path)

    # Verify the ONNX operator-set version and the fixed-shape input/output
    # interface established by the export contract.
    assert summary.opset_version == 18
    assert summary.input_shapes == {"input": (1, 3, 224, 224)}
    assert summary.output_shapes == {"logits": (1, 1000)}

    # Confirm that the exported artifact contains a non-empty computation
    # graph, serialized model parameters, and file data.
    assert summary.node_count > 0
    assert summary.initializer_count > 0
    assert summary.file_size_bytes > 0

    # MobileNetV3 is convolution-based, so its ONNX gragh must contain at
    # least one Conv node.
    assert summary.operator_counts.get("Conv", 0) > 0

    # Every graph node must be represented exactly once in the per-operator
    # node counts.
    assert sum(summary.operator_counts.values()) == summary.node_count

@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        ([1, 3, 224, 224], (1, 3, 224, 224)),
        (["batch", 3, 224, 224], ("batch", 3, 224, 224)),
        ([None, 3, 224, 224], (None, 3, 224, 224)),
        ([0, 3], (0, 3)),
        (["batch", 3, "height", "width"],
         ("batch", 3, "height", "width")),
    ],
)


def test_value_info_shape_preserves_dimensions(shape, expected):
    """Preserve concrete, symbolic, and unknown ONNX dimension metadata."""
    value = onnx.helper.make_tensor_value_info(
        "input", onnx.TensorProto.FLOAT, shape
    )
    assert onnx_export._value_info_shape(value) == expected
