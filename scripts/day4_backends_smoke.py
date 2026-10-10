"""Validate unified inference backends against a direct PyTorch CPU reference.

This smoke test uses one FP32 model and one deterministic synthetic input.
It records numerical consistency and backend metadata without measuring
latency or throughput.

A new report is written only after every executed backend passes.
"""

import copy
import json
import platform
from dataclasses import asdict
from pathlib import Path

import torch

from inference_bench.backends import (
    OrtCpuBackend,
    TorchBackend,
    run_once,
)
from inference_bench.models import load_model
from inference_bench.onnx_export import export_model_to_onnx
from inference_bench.validation import compare_classification_outputs


# Keep the input contract and tolerances identical across all backends.
INPUT_SHAPE = (1, 3, 224, 224)
RTOL = 1e-4
ATOL = 1e-4

# Paths are relative to the project root.
MODEL_PATH = Path(
    "artifacts/models/day4_mobilenet_v3_small_fp32.onnx"
)
RESULT_PATH = Path(
    "artifacts/results/day4_backend_correctness.json"
)


def main() -> None:
    # Seed model constrcution and establish the CPU reference model.
    torch.manual_seed(2026)
    source = load_model("mobilenet_v3_small").cpu().eval()

    # Use a dedicated generator so input values do not depend on random
    # numbers consumed during model construction.
    generator = torch.Generator().manual_seed(2027)
    x = torch.randn(
        INPUT_SHAPE,
        generator=generator,
        dtype=torch.float32,
    )

    # Bypass the backend adapters to obtain an independent reference.
    # Clone the output so the reference owns its storage.
    with torch.inference_mode():
        reference = source(x).detach().clone()

    # Export identical weights from a separate model instance.
    # The exporter may change device placement or evaluation mode.
    export_model_to_onnx(
        model=copy.deepcopy(source),
        example_input=x,
        output_path=MODEL_PATH,
        opset_version=18,
    )

    # Defer backend initialization until each backend is evaluated.
    factories = [
        ("pytorch_cpu", lambda: TorchBackend(source, device="cpu")),
        ("ort_cpu", lambda:OrtCpuBackend(MODEL_PATH)),
    ]

    # Report unavailable hardware explicity rather than treating it as
    # a successful execution.
    skipped = []
    if torch.backends.mps.is_available():
        factories.append(
            ("pytorch_mps", lambda: TorchBackend(source, device="mps"))
        )
    else:
        skipped.append({
            "name": "pytorch_mps",
            "reason": "MPS is unavailable",
        })

    records = []

    for name, build_backend in factories:
        backend = build_backend()

        # Exercise the same lifecycle for every backend:
        # prepare input, synchronize, execute, synchronize, convert to CPU.
        output = run_once(backend, x)

        # Validate shapes and finite values, then collect error metrics
        # and classification agreement.
        comparison = compare_classification_outputs(
            reference,
            output,
            rtol=RTOL,
            atol=ATOL,
        )

        # Use PyTorch's assertion for detailed element-wise diagnostics.
        # Numerical closeness and Top-1 agreement are separate checks.
        torch.testing.assert_close(
            output,
            reference,
            rtol=RTOL,
            atol=ATOL,
        )
        if not comparison.top1_match:
            raise AssertionError(f"{name}: Top-1 mismatch")

        # Record success only after all correctness check pass.
        records.append({
            "backend": backend.metadata(),
            "comparison": asdict(comparison),
            "status": "passed",
        })

        print(
            f"{name}: PASS | "
            f"top1={comparison.candidate_top1_index} | "
            f"max_abs={comparison.max_absolute_error:.9g}"
        )

        # Release this reference before constructing the next backend.
        # Runtime-managed caches may retain allocated.
        del backend

    # Keep manually specified model metadata aligned with load_model().
    result = {
        "schema_version": 1,
        "experiment": "day4_unified_backend_correctness",
        "environment": {
            "python": platform.python_version(),
            "architecture": platform.machine(),
            "system": platform.system(),
            "macos_version": platform.mac_ver()[0],
        },
        "model": {
            "name": "mobilenet_v3_small",
            "weights": "MobileNet_V3_Small_Weights.DEFAULT",
            "precision": "fp32",
            "input_shape": list(INPUT_SHAPE),
            "model_seed": 2026,
            "input_seed": 2027,
            "onnx_opset": 18,
            "onnx_path": str(MODEL_PATH),
        },
        "reference": "direct_pytorch_cpu",
        "results": records,
        "skipped": skipped,
    }

    # Write the report after all executed backends have passed.
    # Reject NaN and infinity to preserve standard JSON compatibility.
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )

    print(f"Executed backends: {len(records)}")
    print(f"Skipped backends: {len(skipped)}")
    print(f"Result saved to: {RESULT_PATH}")

if __name__ == "__main__":
    main()