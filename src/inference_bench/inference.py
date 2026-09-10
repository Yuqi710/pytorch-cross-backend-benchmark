"""Reference PyTorch inference used for backend correctness validation."""

import torch
from torch import nn

from inference_bench.device import resolve_device


def run_reference_inference(
    model: nn.Module,
    inputs: torch.Tensor,
    device_name: str,
) -> torch.Tensor:
    """"Run inference on a PyTorch device and return a detached CPU tensor."""

    device = resolve_device(device_name)

    # Keep the model and its input on the same execution device.
    model = model.to(device)
    device_inputs = inputs.to(device)

    with torch.inference_mode():
        outputs = model(device_inputs)

    # Wait for asynchronous MPS work before consuming the result.
    if device.type == "mps":
        torch.mps.synchronize()

    # Remove any possible gradient relationship and normalize backend
    # outputs onto CPU for correctness comparisons.
    return outputs.detach().cpu()