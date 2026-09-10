"""PyTorch device capabilities reporting and request resolution."""

from dataclasses import dataclass
import platform

import torch


@dataclass(frozen=True)
class DeviceCapabilities:
    """PyTorch-relevant capabilities of the current runtime.

    frozen=True prevents filed reassignment after construction, protecting
    benchmark metadata from accidental modification.
    """

    architecture: str
    python_version: str
    torch_version: str
    mps_built: bool
    mps_available: bool


def detect_capabilities() -> DeviceCapabilities:
    """Return PyTorch-relevant capabilities of the current runtime."""

    return DeviceCapabilities(
        architecture=platform.machine(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        # Built support and runtime availability are intentionally recorded separately
        # because an MPS-enabled build may run on an unsupported host.
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
    )


def resolve_device(name: str) -> torch.device:
    """Resolve a user- or configuration-provided device name to a PyTorch device object.

    This function validates the device request but does not execute a
    workload on the underlying hardware.

    Args:
        name: Requested device name, e.g. "cpu" or "mps". Case and whitespace are ignored.

    Returns:
        A PyTorch device object corresponding to the requested device.

    Raises:
        RuntimeError: If MPS is requested but unavailable.
        ValueError: If the requested device name is unsupported.
    """

    normalized_name = name.strip().lower()

    if normalized_name == "cpu":
        return torch.device("cpu")

    if normalized_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return torch.device("mps")

    raise ValueError(f"Unsupported device: {name}")