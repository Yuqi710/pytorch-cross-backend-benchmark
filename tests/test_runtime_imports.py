"""Tests for keeping PyTorch execution separate from ONNX Runtime imports."""

import hashlib
import json
import shutil
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path
from textwrap import dedent

import pytest


ROOT = Path(__file__).resolve().parents[1]
# Run this setup before PyTorch code in each independent Python subprocess.
# Allow checks for whether ORT is installed, but reject actual ORT imports.
ORT_IMPORT_GUARD = dedent("""
    import builtins
    import os
    import sys

    # Success must come from avoiding ORT imports, not from disabling telemetry.
    os.environ.pop("ORT_DISABLE_TELEMETRY", None)

    # Keep normal imports working for all other packages.
    original_import = builtins.__import__

    def import_without_ort(name, *args, **kwargs):
        '''Reject ORT imports and pass other imports to the original function.'''

        if name == "onnxruntime" or name.startswith("onnxruntime."):
            raise AssertionError(f"Unexpected ORT import: {name}")
        return original_import(name, *args, **kwargs)

    # Apply the check to later import statements in this subprocess.
    builtins.__import__ = import_without_ort
""")


def run_python(code, *args):
    """Run test code in a fresh Python subprocess and require a clean exit."""

    # Modules loaded by other pytest tests must not affect this check.
    result = subprocess.run(
        [sys.executable, "-c", code, *map(str, args)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
    )
    # Writing a result before a native abort does not constitute a successful run.
    assert result.returncode == 0, result.stdout + result.stderr


def test_torch_backend_runs_without_importing_onnxruntime():
    """Check the shared adapter independently of the profiling entry point."""

    # Verify that TorchBackend can run without importing ONNX Runtime.
    run_python(ORT_IMPORT_GUARD + dedent("""
        import torch
        from inference_bench.backends import TorchBackend, run_once

        model = torch.nn.Linear(3, 2).eval()
        x = torch.randn(1, 3)
        backend = TorchBackend(model, device="cpu", input_shape=(1, 3))
        with torch.inference_mode():
            expected = model(x)
        torch.testing.assert_close(run_once(backend, x), expected)
        assert backend.metadata()["runtime"] == "pytorch"
        assert not any(name.startswith("onnxruntime") for name in sys.modules)
    """))


@pytest.mark.parametrize(
    ("backend", "threads"),
    [("pytorch_cpu", 1), ("pytorch_cpu", 6), ("ort_cpu", 1)],
)
def test_profile_cli_preserves_runtime_isolation(tmp_path, backend, threads):
    """Check runtime imports and saved profiling results in separate processes."""

    # Run a copy in a temporary project directory so model exports and result
    # files cannot overwrite existing experiment artifacts.
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    entrypoint = scripts / "day7_profile_cpu.py"
    shutil.copyfile(ROOT / "scripts/day7_profile_cpu.py", entrypoint)

    # The ORT case uses the telemetry setting to avoid the known shutdown crash.
    # PyTorch cases leave it unset and must not load ORT.
    setup = ORT_IMPORT_GUARD if backend == "pytorch_cpu" else dedent("""
        import os
        import sys
        os.environ["ORT_DISABLE_TELEMETRY"] = "1"
    """)
    run_python(setup + dedent("""
        import runpy
        from functools import partial
        from inference_bench import models

        # Use random weights so the test does not download pretrained weights.
        models.load_model = partial(models.load_model, pretrained=False)
        entrypoint, backend, threads = sys.argv[1:]
        sys.argv = [
            entrypoint, "--backend", backend,
            "--intra-op-threads", threads,
            "--warmup", "0", "--active-runs", "1",
            "--output-dir", "results", "--trace-dir", "traces",
        ]
        if backend == "ort_cpu":
            sys.argv += ["--optimized-model-path", "optimized.onnx"]
        runpy.run_path(entrypoint, run_name="__main__")
        # Only the ORT profiling process should have loaded ORT.
        loaded_ort = any(name.startswith("onnxruntime") for name in sys.modules)
        assert loaded_ort == (backend == "ort_cpu")
    """), entrypoint, backend, threads)

    paths = list((tmp_path / "results").glob("*.json"))
    assert len(paths) == 1
    result = json.loads(paths[0].read_text())
    assert result["status"] == "passed"
    assert result["backend"]["name"] == backend
    assert result["threading"]["intra_op"]["runtime_reported_value"] == threads
    # PyTorch results must retain installed ORT version metadata without loading it.
    assert result["environment"]["versions"]["onnxruntime"] == version("onnxruntime")
    for phase in ("before", "after"):
        assert result["correctness"][phase]["allclose"]
        assert result["correctness"][phase]["top1_match"]

    # Verify persisted trace identity as well as the summarized profiling events.
    trace = tmp_path / result["trace"]["path"]
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == result["trace"]["sha256"]
    if backend == "ort_cpu":
        assert result["backend"]["providers"] == ["CPUExecutionProvider"]
        assert result["backend"]["profiling_enabled"] is True
        assert result["profile_summary"]["active_run_count"] == 1
        assert result["profile_summary"]["node_event_count"] > 0
        assert (tmp_path / "optimized.onnx").is_file()
    else:
        assert result["graph_optimization_level"] is None
        assert result["profile_summary"]["operator_group_count"] > 0
