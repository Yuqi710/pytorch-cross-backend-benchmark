"""Backend adapters with explicit inference lifecycle boundaries.

The adapters separate initialization, input preparation, execution,
synchronization, and output conversion. This allows callers to define
measurement boundaries without embedding backend-specific data handling
in the benchmark loop.

Public inputs are fixed-shape FP32 CPU tensors. Each backend uses its
native input and output representations during execution and provides
an explicit conversion to an independent CPU tensor.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, TypeVar

import numpy as np
import torch
from torch import nn

# Keep type-only ORT dependencies out of the shared adapter import path.
if TYPE_CHECKING:
    from inference_bench.ort_backend import GraphOptimizationLevelName


# Connect the native data types across the lifecycle methods:
# prepare_input() -> PreparedInput -> run() -> NativeOutput -> to_cpu().
PreparedInput = TypeVar("PreparedInput")
NativeOutput = TypeVar("NativeOutput")

# Default image input shape in NCHW order.
DEFAULT_INPUT_SHAPE = (1, 3, 224, 224)


class InferenceBackend(Protocol[PreparedInput, NativeOutput]):
    """Define the interface for single-input, single-output inference.

    Implementations may satisfy this protocol structually; explicit
    inheritance is not required. Type annotations describe the contract
    for static analysis and do not enforece it at runtime.

    Callers should prepare inputs before execution, synchronize at the
    required completion boundaries, and convert outputs separately.
    """

    name: str

    def prepare_input(self, x: torch.Tensor) -> PreparedInput:
        """Validate a CPU input and create a backend-native snapshot."""
        ...

    def run(self, prepared: PreparedInput) -> NativeOutput:
        """Execute inference using an input prepared by this backend.

        Return the native output without converting it to the common
        CPU representation. Accelerator execution may still be pending
        when this method returns.
        """
        ...

    def synchronize(self) -> None:
        """Wait for the pending execution when required by the backend."""
        ...

    def to_cpu(self, output: NativeOutput) -> torch.Tensor:
        """Convert a native output to an independent CPU tensor."""
        ...

    def metadata(self) -> dict[str, object]:
        """Return backend identity and runtime configuration metadata."""
        ...


def _validate_input(
    x: torch.Tensor,
    expected_shape: tuple[int, ...],
) -> None:
    """Validate the shared CPU, FP32, and fixed-shape input contract.

    Args:
        x: Input tensor supplied by the caller.
        expected_shape: Exact shape accepted by the backend.

    Raises:
        ValueError: If the input is not on CPU or has an unexpected shape.
        TypeError: If the input dtype is not torch.float32.

    This check does not validate numerical values or model-specific
    preprocessing such as normalization and channel ordering.
    """

    if x.device.type != "cpu":
        raise ValueError("prepared_input expects a CPU Tensor")

    if x.dtype != torch.float32:
        raise TypeError(
            f"Expected torch.float32, received {x.dtype}"
        )

    if tuple(x.shape) != expected_shape:
        raise ValueError(
            f"Expected shape {expected_shape}, received {tuple(x.shape)}"
        )


class TorchBackend:
    """Run an independently owned PyTorch model on CPU or MPS.

    Model copying, evaluation-mode selection, and device placement occur
    during initialization. The supplied model must support deepcopy,
    as the torchvision models used in this project do.
    """

    def __init__(
            self,
            model: nn.Module,
            *,
            device: str,
            input_shape: tuple[int, ...] = DEFAULT_INPUT_SHAPE,
    ) -> None:
        """Create an FP32 inference backend without modifying the source.

        Args:
            model: Source model whose parameters and buffers are copied.
            device: Target device, either "cpu" or "mps".
            input_shape: Exact shape accepted by prepare_input().

        Raises:
            ValueError: If the requested device is unsupported.
            RuntimeError: If MPS is requested but unavailable.
        """
        if device not in ("cpu", "mps"):
            raise ValueError(f"Unsupported PyTorch device: {device}")

        if device == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS is unavailable")

        self.name = f"pytorch_{device}"
        self._device = torch.device(device)
        self._input_shape = tuple(input_shape)

        # eval() and Module.to() modify their target module in place.
        # Copy first to isolate this backend from the caller and from
        # other backends constructed from the same source model.
        self._model = (
            copy.deepcopy(model)
            .eval()
            .to(device=self._device, dtype=torch.float32)
        )

    def prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        """Create a contiguous input snapshot on the target device."""
        _validate_input(x, self._input_shape)

        # detach() removes autograd tracking but does not copy storage.
        # copy=True guarantees an independent snapshot even on CPU.
        # contiguous() normalizes storage layout without permuting axes.
        return (
            x.detach()
            .to(device=self._device, copy=True)
            .contiguous()
        )

    @torch.inference_mode()
    def run(self, prepared: torch.Tensor) -> torch.Tensor:
        """Execute the model and retain the output on its native device.

        The input must come from this backend's prepare_input() method.
        Device transfer and explicit completion synchronization are
        handled by separate lifecycle methods.
        """
        return self._model(prepared)

    def synchronize(self) -> None:
        """Wait for MPS work to complete; no extra wait is needed on CPU."""
        if self._device.type == "mps":
            torch.mps.synchronize()

    def to_cpu(self, output: torch.Tensor) -> torch.Tensor:
        """Return a detached, contiguous CPU copy of the native output.

        A copy is made even the output is already on CPU, so the
        returned tensor does not alias the native output's storage.
        """
        return output.detach().to("cpu", copy=True).contiguous()

    def metadata(self) -> dict[str, object]:
        """Report backend identity and current PyTorch configuration."""
        return {
            "name": self.name,
            "runtime": "pytorch",
            "runtime_version": torch.__version__,
            "device": str(self._device),
            "precision": "fp32",
            "input_shape": list(self._input_shape),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),

            # Report the environment value for reproducibility.
            # This field does not establish whether fallback occurred.
            "mps_fallback_env": os.environ.get(
                "PYTORCH_ENABLE_MPS_FALLBACK"
            ),
        }


class OrtCpuBackend:
    """Run a fixed-shape FP32 ONNX model using only the CPU provider.

    Session creation and model-interface inspection occur once during
    initialization. Prepared inputs and native outputs are Numpy arrays.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        intra_op_num_threads: int | None = None,
        graph_optimization_level: GraphOptimizationLevelName = "all",
    ) -> None:
        """Load an ONNX model and validate the supported session contract.

        Args:
            model_path: Path to the exported ONNX model.
            intra_op_num_threads:
                Explicit intra-op thread count for the ONNX Runtime
                session. None preserves the runtime-default policy.
            graph_optimization_level:
                Graph-optimization level used to create the underlying
                ORT session.

        Raises:
            RuntimeError: If the session is not restricted to the CPU EP.
            ValueError: If the model does not have one input, one output,
                and a nonempty input shape of positive fixed dimensions.
            TypeError: If the input or output is not an FP32 tensor.

        Model-loading and compatibility errors from ONNX Runtime are
        propagated to the caller.

        The benchmark adapter creates a regular, non-profiled session.
        Dedicated profiling code manages profiling sessions directly so
        that trace collection cannot affect formal latency measurements.
        """
        # Load the native runtime only when an ORT backend is constructed.
        from inference_bench.ort_backend import create_cpu_session

        self.name = "ort_cpu"

        # Preserve the stable experiment-facing label separately from the
        # runtime enum representation reported by SessionOptions.
        self._graph_optimization_level_name = (
            graph_optimization_level
        )

        # Reuse the project's CPU-only session creation policy.
        self._session = create_cpu_session(
            model_path,
            intra_op_num_threads=intra_op_num_threads,
            graph_optimization_level=graph_optimization_level,
        )

        # Inspect this session's registered providers.
        if self._session.get_providers() != ["CPUExecutionProvider"]:
            raise RuntimeError("Expected a CPU-only ORT session")

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()

        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError("Expected one input and one output")

        if inputs[0].type != "tensor(float)":
            raise TypeError("Expected an FP32 ONNX input")

        if outputs[0].type != "tensor(float)":
            raise TypeError("Expected an FP32 ONNX output")

        # Symbolic, unknown, empty, and nonpositive input dimensions are
        # outside the fixed-shape contract of this adapter.
        shape = inputs[0].shape
        if not shape or any(
            not isinstance(d, int) or d <= 0 for d in shape
        ):
            raise ValueError("Day4 requires a fixed input shape")

        # Cache the model's declared interface instead of assuming names.
        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        self._input_shape = tuple(shape)

    def prepare_input(self, x: torch.Tensor) -> np.ndarray:
        """Create a contiguous Numpy snapshot of the validated CPU input."""
        _validate_input(x, self._input_shape)

        # numpy() can share storage with the detached CPU tensor.
        # The final copy isolates the prepared input from caller changes.
        return x.detach().contiguous().numpy().copy()

    def run(self, prepared: np.ndarray) -> np.ndarray:
        """Execute the existing session and return its sole native output.

        Input preparation and conversion to the common CPU tensor format
        are deliberately handled by separate lifecycle methods.
        """
        return self._session.run(
            [self._output_name],
            {self._input_name: prepared},
        )[0]

    def synchronize(self) -> None:
        """Provide a no-op completion boundary for synchronous CPU runs."""
        # CPU-only session.run() returns after producing its output.
        # No additional accelerator synchronization is required here.
        pass

    def to_cpu(self, output: np.ndarray) -> torch.Tensor:
        """Return a CPU tensor independent of the native output array."""
        # from_numpy() shares storage with the newly created copy,
        # not with the original ORT output array.
        return torch.from_numpy(output.copy())

    def metadata(self) -> dict[str, object]:
        """Report session configuration without inferring worker counts."""
        import onnxruntime as ort

        options = self._session.get_session_options()

        return {
            "name": self.name,
            "runtime": "onnxruntime",
            "runtime_version": ort.__version__,
            "device": "cpu",
            "precision": "fp32",
            "input_shape": list(self._input_shape),
            "input_name": self._input_name,
            "output_name": self._output_name,
            "providers": self._session.get_providers(),
            "graph_optimization_level_name": (
                self._graph_optimization_level_name
            ),
            "graph_optimization_level": str(
                options.graph_optimization_level
            ),
            "execution_mode": str(options.execution_mode),

            # These are configured values, not measured worker counts.
            # Zero delegates thread-count selection to ONNX Runtime.
            "intra_op_num_threads": options.intra_op_num_threads,
            "inter_op_num_threads": options.inter_op_num_threads,
        }


def run_once(
    backend: InferenceBackend[PreparedInput, NativeOutput],
    x: torch.Tensor,
) -> torch.Tensor:
    """Execute one complete inference lifecycle for correctness checks.

    This entry point includes input preparation, synchronization, and
    output conversion. It is intended for correctness checks rarher than
    isolated execution-latency measurement.
    """
    prepared = backend.prepare_input(x)

    # Wait for previously submitted device work, including input
    # preparation, before starting inference.
    backend.synchronize()

    native_output = backend.run(prepared)

    # Wait for inference to complete before converting its output.
    backend.synchronize()

    return backend.to_cpu(native_output)
