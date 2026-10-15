"""Utilities for exporting PyTorch models to ONNX and inspecting ONNX graphs."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import onnx
import torch
from onnx import shape_inference
from torch import nn


# An ONNX dimension can be:
# - int: a statically known dimension;
# - str: a symbolic dimension, such as "batch_size";
# - None: an unknown dimension that could be inferred.
ShapeDimension = int | str | None
TensorShape = tuple[ShapeDimension, ...]


@dataclass(frozen=True)
class OnnxGraphSummary:
    """Structural metadata collected from a validate ONNX model.

    Attributes:
        ir_version:
            Version of the ONNX intermediate representation used by the model.
        opset_version:
            Version of the default ONNX operator set.
        producer_name:
            Name of the framework or tool that produce the model.
        producer_version:
            Version of the model producer.
        node_count:
            Number of computational nodes in the ONNX graph.
        initializer_count:
            Number of stored parameter and constant tensors.
        file_size_bytes:
            Size of the serialized ONNX model in bytes.
        input_shapes:
            Mapping from user-visible input names to tensor shapes.
        output_shaped:
            Mapping from output names to tensor shapes.
        operator_counts:
            Number of occurrences of each ONNX operator type.
    """

    ir_version: int
    opset_version: int
    producer_name: str
    producer_version: str
    node_count: int
    initializer_count: int
    file_size_bytes: int
    input_shapes: dict[str, TensorShape]
    output_shapes: dict[str, TensorShape]
    operator_counts: dict[str, int]


def export_model_to_onnx(
    model: nn.Module,
    example_input: torch.Tensor,
    output_path: str | Path,
    *,
    input_name: str = "input",
    output_name: str = "logits",
    opset_version: int = 18,
) -> Path:
    """Export a single-input PyTorch model to a fixed-shape ONNX model.

    The model is switched to evaluation mode and moved to the CPU before
    export. The example input is detached from its existing autograd graph
    and moved to the CPU. Its shape is used as the static input shape of the
    exported ONNX model.

    Args:
        model: PyTorch module to export. Its device and training state are
            modified by this fuction,
        example_input: Example tensor used to capture the model graph and
            specialize its input shape.
        output_path: Destination path of the generated ONNX model.
        input_name: Name assigned to the model's user input.
        output_name: Name assigned to the model's user output.
        opset_version: Target version of the default ONNX operator set.

    Returns:
        The path of the generated ONNX model.

    Raises:
        RuntimeError: If the Dynamo exporter does not return an ONNXProgram.

    Note:
        This function does not enforce a specific floating-point dtype and
        does not perform numerical verification with ONNX Runtime.
    """

    destination = Path(output_path)
    # Create the destination directory if it does not already exist.
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Export on the CPU in evaluation mode to obtain deterministic inference
    # behavior from modules such as BatchNorm and Dropout.
    export_model = model.cpu().eval()
    # Remove the example input from its existing autograd graph and place it
    # on the same device as the model.
    export_input = example_input.detach().cpu()

    # Capture the PyTorch program with the torch.exported-based Dynamo exporter
    # and translate it into and optimized ONNX program.
    onnx_program = torch.onnx.export(
        export_model,
        (export_input,),
        f=None,
        input_names=[input_name],
        output_names=[output_name],
        opset_version=opset_version,
        dynamo=True,
        external_data=False,
        optimize=True,
        verify=False,
        report=False,
    )

    # Dynamo exporter is expected to return a serializable ONNXProgram.
    if onnx_program is None:
        raise RuntimeError("Dynamo ONNX exporter did not return an ONNXProgram")

    # Store the graph and its initializers in a single ONNX file. Model
    # parameters are treated as fixed initializers rather than user inputs.
    onnx_program.save(
        destination,
        include_initializers=True,
        keep_initializers_as_inputs=False,
        external_data=False,
    )

    return destination


def inspect_onnx_model(model_path: str | Path) -> OnnxGraphSummary:
    """Validate an ONNX model and summarize its graph structure.
    The function first validates the serialized model, then performs ONNX
    shape inference and validates the inferred model again. It reports only
    user-visible graph inputs, excluding any initializers that may also appear
    in the graph input list.

    Args:
        model_path: Path to the ONNX model to inspect.

    Returns:
        Structual metadata for the validated ONNX graph.

    Raises:
        FileNotFoundError:
            If 'model_path' does not point to an existing file.
        ValueError:
            If the model does not contain exactly one default ONNX opset.
        onnx.checker.ValidationError:
            If either the original or shape-inferred model is invalid.
    """

    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(path)

    model = onnx.load(str(path), load_external_data=True)

    # Model checking validates the protobuf structure, graph connections,
    # operator schemas, attributes, and other ONNX constrains. It does not
    # execute the model or establish numerical equivalence with PyTorch.
    onnx.checker.check_model(model)

    # Shape inference returns a new ModelProto with additional type and shape
    # information where inference is supported. strict_mode=False allows
    # best-effort inference when same intermediate shapes cannot be resolved.
    inferred_model = shape_inference.infer_shapes(
        model,
        check_type=True,
        strict_mode=False,
        data_prop=False,
    )

    # Validate the inferred representation to ensure that inferrence pass
    # did not produce an invalid graph.
    onnx.checker.check_model(inferred_model)

    opset_version = _default_onnx_opset(model)

    # Initializers normally represent model parameters and constants. The set
    # is retained here so that older or externally produced ONNX models do not
    # incorrectly report their weights as user-provided runtime inputs.
    initializer_names = {
        initializer.name
        for initializer in inferred_model.graph.initializer
    }

    input_shapes = {
        value.name: _value_info_shape(value)
        for value in inferred_model.graph.input
        if value.name not in initializer_names
    }
    output_shapes = {
        value.name: _value_info_shape(value)
        for value in inferred_model.graph.output
    }

    # Count nodes by ONNX operator type. Sorting makes JSON output and test
    # results deterministic across repeated runs.
    operator_counts = dict(
        sorted(
            Counter(
                node.op_type
                for node in inferred_model.graph.node
            ).items()
        )
    )

    return OnnxGraphSummary(
        ir_version=int(model.ir_version),
        opset_version=opset_version,
        producer_name=model.producer_name,
        producer_version=model.producer_version,
        node_count=len(inferred_model.graph.node),
        initializer_count=len(
            inferred_model.graph.initializer
        ),
        file_size_bytes=path.stat().st_size,
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        operator_counts=operator_counts,
    )


def _default_onnx_opset(model: onnx.ModelProto) -> int:
    """Return the version of the model's default ONNX operator set.

    ONNX models may import operator sets from multiple domains. This helper
    selects only the standard ONNX domain, represented by either an empty
    string or "ai.onnx".
    """

    versions = [
        int(opset.version)
        for opset in model.opset_import
        if opset.domain in ("", "ai.onnx")
    ]

    if len(versions) != 1:
        raise ValueError(
            f"Expected exactly one default ONNX opset, found {versions}"
        )

    return versions[0]


def _value_info_shape(
    value_info: onnx.ValueInfoProto,
) -> TensorShape:
    """Extract a Python representation of an ONNX tensor shape.

    Args:
        value_info: ONNX value descriptor containing tensor type and shape metadata.

    Returns:
        Tensor dimensions represented as integers, symbolic strings, or
        'None' for unknown dimensions.
    """

    dimensions: list[ShapeDimension] = []

    for dimension in value_info.type.tensor_type.shape.dim:
        # HasField must be used here because zero is a valid static dimension
        # and must not be mistaken for a missing value.
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)

    return tuple(dimensions)