from pathlib import Path

import onnxruntime as ort

import torch


def _validate_intra_op_num_threads(
    value: int | None,
) -> None:
    """Validate an optional explicit intra-op thread count.

    None preserves ONNX Runtime's default thread-selection policy.
    An explicit value must be a positive built-in integer. Booleans
    are rejected even though bool is a subclass of int.
    """
    if value is not None and (
        type(value) is not int or value < 1
    ):
        raise ValueError(
            "intra_op_num_threads must be a positive integer or None"
        )


def create_cpu_session(
    model_path: str | Path,
    *,
    intra_op_num_threads: int | None = None,
) -> ort.InferenceSession:
    """Create an ONNX Runtime session backed exclusively by the CPU EP.

    Args:
        model_path: Path to the serialized ONNX model.
        intra_op_num_threads:
            Explicit intra-op thread count for the ONNX Runtime
            session. None preserves the runtime-default policy.

    Returns:
        An initialized ONNX Runtime inference session whose only active
        execution provider is CPUExecutionProvider.

    The session retains sequential graph-node scheduling and the
    runtime-default inter-op configuration. The configured intra-op
    value does not represent a measured active-thread count.
    """
    _validate_intra_op_num_threads(intra_op_num_threads)

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

    if intra_op_num_threads is not None:
        session_options.intra_op_num_threads = (
            intra_op_num_threads
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