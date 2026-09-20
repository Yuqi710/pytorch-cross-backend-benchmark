from pathlib import Path

import onnxruntime as ort

import torch


def create_cpu_session(
    model_path: str | Path,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session backed exclusively by the CPU EP.

    Args:
        model_path: Path to the serialized ONNX model.

    Returns:
        An initialized ONNX Runtime inference session whose only active
        execution provider is CPUExecutionProvider.
    """

    session_options = ort.SessionOptions()

    # Apply all graph optimizations supported by the active CPU
    # execution provider during session initialization.
    session_options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    # Schedule graph nodes sequentially. Individual operator kernels
    # may still use multiple intra-op threads.
    session_options.execution_mode = (
        ort.ExecutionMode.ORT_SEQUENTIAL
    )

    # Explicitly restrict the session to the CPU EP so that other
    # providers available in the environment cannot execute any nodes.
    return ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )


def run_ort_inference(
    session: ort.InferenceSession,
    input_tensor: torch.Tensor,
) -> torch.Tensor:
    """Run a single-input, single-output model through ORT.

    The caller is responsible for providing a tensor whose shape and
    data type match the input contract declared by the ONNX model.

    Args:
        session: An initialized ONNX Runtime inference session.
        input_tensor: The PyTorch tensor supplied to the model input.

    Returns:
        A CPU PyTorch tensor containing the model's first output.
    """

    # Resolve the model interface from the loaded ONNX graph instead
    # of relying on hard-coded input and output names.
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Convert the input to a contiguous CPU NumPy array because the
    # ORT CPU Execution Provider consume host-memory input data.
    input_array = (
        input_tensor
        .detach()
        .cpu()
        .contiguous()
        .numpy()
    )

    # Request the model's first output and bind the NumPy array to
    # the corresponding ONNX input name.
    output_array = session.run(
        [output_name],
        {input_name: input_array},
    )[0]

    # Copy the ORT output before wrapping it as a Torch tensor
    # to avoid sharing memory with the retured Numpy array.
    return torch.from_numpy(output_array.copy())