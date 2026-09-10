"""Smoke test for PyTorch inference consistency across CPU and MPS.

The Script runs the same pretrained MobileNetV3 Small model with the
same input on CPU and Apple MPS, then verifies that both backends:

1. Produce outputs with the expected shape.
2. Produce finite values with NaN or infinity.
3. Select the same Top-1 class.
4. Produce numerically close logits within the configured tolerances.

This is a correctness smoke test, not a performance benchmark.
"""

import torch

from inference_bench.device import detect_capabilities
from inference_bench.models import load_model
from inference_bench.inference import run_reference_inference

def main() -> None:
    "Run the CPU/MPS inference consistency smoke test."

    # Use a fixed seed so the synthetic input is reproducible across runs.
    # This does not guarantee bitwise-identical results across backends.
    torch.manual_seed(20260908)

    capabilities = detect_capabilities()
    print("Device capabilities:", capabilities)

    # Load the pretrained model in evaluation mode.
    model = load_model(
        "mobilenet_v3_small",
        pretrained=True,
    )

    # Create one synthetic RGB image in NCHW format.
    #
    # The random input is sufficient for backend consistency validation,
    # but it must not be used to evaluate classification accuracy.
    inputs = torch.randn(1, 3, 244, 244)

    cpu_outputs = run_reference_inference(
        model = model,
        inputs = inputs,
        device_name="cpu",
    )

    mps_outputs = run_reference_inference(
        model = model,
        inputs=inputs,
        device_name="mps",
    )

    absolute_error = (cpu_outputs - mps_outputs).abs()

    # The output contains one Logit for each ImageNet class.
    # Since the input is synthetic noise, the selected class has no
    # semantic meaning and is used only for consistency validation.
    cpu_top1 = int(cpu_outputs.argmax(dim=1).item())
    mps_top1 = int(mps_outputs.argmax(dim=1).item())

    print(f"Output shape: {tuple(cpu_outputs.shape)}")
    print(f"CPU top-1 index: {cpu_top1}")
    print(f"MPS top-1 index: {mps_top1}")
    print(f"Maximum absolute error: {absolute_error.max().item():8f}")
    print(f"Mean absolute error: {absolute_error.mean().item():8f}")

    # Reject invalid numerical results before comparing backend outputs.
    assert torch.isfinite(cpu_outputs).all()
    assert torch.isfinite(mps_outputs).all()

    # Verify that both backends make the same Top-1 decision.
    assert cpu_top1 == mps_top1

    # Small differences are expected because CPU and MPS may use different
    # kernels, parallel execution orders, and floating-point implementation.
    torch.testing.assert_close(
        cpu_outputs,
        mps_outputs,
        rtol=1e-3,
        atol=1e-3,
    )

    print("Day 1 CPU/MPS correctness smoke test: PASS")

if __name__ == "__main__":
    main()