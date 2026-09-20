"""Validate numerical consistency between PyTorch CPU and ORT CPU."""

import json
from dataclasses import asdict
from pathlib import Path

import onnxruntime as ort
import torch

from inference_bench.models import load_model
from inference_bench.onnx_export import export_model_to_onnx
from inference_bench.ort_backend import (
    create_cpu_session,
    run_ort_inference,
)
from inference_bench.validation import(
    compare_classification_outputs,
)


# Define the reproducible model, input, export, and validation configuration.
MODEL_NAME = "mobilenet_v3_small"
WEIGHTS_SOURCE = "MobileNet_V3_Small_Weights.DEFAULT"
INPUT_SHAPE = (1, 3, 224, 224)
OPSET_VERSION = 18
MODEL_SEED = 2026
INPUT_SEED = 2027
RTOL = 1e-4
ATOL = 1e-4

# Store generated model binaries separately from committed result metadata.
MODEL_PATH = Path(
    "artifacts/models/mobilenet_v3_small_fp32_opset18.onnx"
)
RESULT_PATH = Path(
    "artifacts/results/day3_ort_correctness.json"
)


def main() -> None:
    """Run the PyTorch-to-ORT numerical-consistency experiment."""

    # Seed model construction so randomly initialized models remain
    # reproducible. For pretrained weights, this also keeps the model
    # construction procedure explicit and deterministic where applicable.
    torch.manual_seed(MODEL_SEED)
    model = load_model(
        MODEL_NAME,
        pretrained=True,
    ).cpu().eval()

    # Use a decided generator so input generation is independent of
    # any random numbers consumed while constrcuting the model.
    input_generator = torch.Generator(device="cpu")
    input_generator.manual_seed(INPUT_SEED)
    input_tensor = torch.randn(
        INPUT_SHAPE,
        generator=input_generator,
        dtype=torch.float32,
    )

    # Produce the reference output with PyTorch CPU inference.
    with torch.inference_mode():
        pytorch_output = model(input_tensor).detach().cpu()

    # Export the same in-memory model instance and input contract used
    # for the PyTorch reference inference.
    export_model_to_onnx(
        model=model,
        example_input=input_tensor,
        output_path=MODEL_PATH,
        input_name="input",
        output_name="logits",
        opset_version=OPSET_VERSION,
    )

    # Load the exported ONNX model into an ORT session restricted to the
    # CPU Execution provider, then execute it with the identical input.
    session = create_cpu_session(MODEL_PATH)
    ort_output = run_ort_inference(
        session,
        input_tensor,
    )

    # Calculate shape, finite-value, top-1, and numerical-error metrics.
    comparison = compare_classification_outputs(
        pytorch_output,
        ort_output,
        rtol=RTOL,
        atol=ATOL,
    )

    # Raise a detailed element-wise diagnostic if the outputs exceed
    # the configured absolute or relative tolerance.
    torch.testing.assert_close(
        ort_output,
        pytorch_output,
        rtol=RTOL,
        atol=ATOL,
    )

    # Validate task-level agreement independent of tensor closeness.
    # Near-tied logits may produce different Top-1 indices even when
    # their element-wise numerical differences are small.
    if not comparison.top1_match:
        raise AssertionError(
            "Top-1 mismatch: "
            f"PyTorch={comparison.reference_top1_index}, "
            f"ORT={comparison.candidate_top1_index}"
        )

    # Read the model interface as observed by ONNX Runtime rather than
    # duplicating the expected names, shapes, and types in the report.
    session_input = session.get_inputs()[0]
    session_output = session.get_outputs()[0]

    # Assemble machine-readable model, runtime, interface, and
    # numerical-comparison metadata.
    result = {
        "model": {
            "name": MODEL_NAME,
            "weights": WEIGHTS_SOURCE,
            "precision": "fp32",
            "model_seed": MODEL_SEED,
            "input_seed": INPUT_SEED,
            "onnx_opset": OPSET_VERSION,
            "onnx_path": str(MODEL_PATH),
        },
        "runtime": {
            "python_backend": "CPython",
            "torch_version": torch.__version__,
            "onnxruntime_version": ort.__version__,
            "available_execution_providers": (
                ort.get_available_providers()
            ),
            "active_execution_providers": (
                session.get_providers()
            ),
        },
        "session_interface": {
            "input_name": session_input.name,
            "input_shape": session_input.shape,
            "input_type": session_input.type,
            "output_name": session_output.name,
            "output_shape": session_output.shape,
            "output_type": session_output.type,
        },
        "comparison": asdict(comparison),
    }

    # Create the result directory if required and persist the report
    # only after all correctness checks have passed.
    RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # Print the same report to the terminal for immediate inspection.
    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )
    print(
        "\nDay 3 PyTorch CPU / "
        "ONNX Runtime CPU correctness test: PASS"
    )
    print(
        f"Result saved to: {RESULT_PATH}"
    )


if __name__ == "__main__":
    # Prevent the experiment from running when this module is imported.
    main()