"""Export MobileNetV3 Small to ONNX and persist graph metadata."""

import json
from dataclasses import asdict
from pathlib import Path

import torch

from inference_bench.models import load_model
from inference_bench.onnx_export import (
    export_model_to_onnx,
    inspect_onnx_model,
)


# Reproducible export configuration.
MODEL_NAME = "mobilenet_v3_small"
INPUT_SHAPE = (1, 3, 224, 224)
OPSET_VERSION = 18
INPUT_RANDOM_SEED = 2026
WEIGHTS_SOURCE = "MobileNet_V3_Small_Weights.DEFAULT"

# Generated ONNX binaries are stored separately from lightweight,
# version-controlled experiment reports.
MODEL_PATH = Path(
    "artifacts/models/mobilenet_v3_small_fp32_opset18.onnx"
)
SUMMARY_PATH = Path(
    "artifacts/results/day2_onnx_graph_summary.json"
)


def main() -> None:
    """Export the reference model and write its ONNX graph summary.

    The functio  performs the complete Day 2 export workflow:

    1. Initialize reproducible PyTorch random state.
    2. Load the reference model in CPU evaluation mode.
    3. Create a fixed-shape FP32 example input.
    4. Export the model to an optimized ONNX graph.
    5. Validate and inspect the serialized model.
    6. Persist the experiment configuration and graph metadata as JSON.
    """

    # Set the seed before model construction so that both randomly initialized
    # weights and the example input are reproducible when pretrained weights
    # are not used.
    torch.manual_seed(INPUT_RANDOM_SEED)

    # Reuse the Day 1 model factory to keep model construction consistent
    # across PyTorch correctness tests and ONNX export.
    model = load_model(
        MODEL_NAME,
        pretrained=True,
    ).cpu().eval()

    # The example tensor defines the exported model's FP32 input contract and
    # fixed NCHW: batch, channels, height, width.
    example_input = torch.randn(
        INPUT_SHAPE,
        dtype=torch.float32,
    )

    # Export a single-input, single-output ONNX model using the explicity
    # selected default ONNX operator-set version.
    export_model_to_onnx(
        model=model,
        example_input=example_input,
        output_path=MODEL_PATH,
        input_name="input",
        output_name="logits",
        opset_version=OPSET_VERSION,
    )

    # Reload the serialized artifact, validate it, infer available tensor
    # shapes, and collect graph-level metadata.
    summary = inspect_onnx_model(MODEL_PATH)

    # Combine experiment configuration with metadata extracted from the actual
    # serialized model. Path and tuple values are converted into JSON-friendly
    # representations.
    result = {
        "model_name": MODEL_NAME,
        "weights": WEIGHTS_SOURCE,
        "pretrained": True,
        "input_random_seed": INPUT_RANDOM_SEED,
        "model_path": str(MODEL_PATH),
        "framework": f"PyTorch {torch.__version__}",
        "format": "ONNX",
        "precision": "fp32",
        "fixed_input_shape": list(INPUT_SHAPE),
        "dynamic_shapes": False,
        **asdict(summary),
    }

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nONNX model saved to: {MODEL_PATH}")
    print(f"\nGraph summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    # Run the workflow only when this module is executed as a script.
    main()