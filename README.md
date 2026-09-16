# PyTorch Cross-Backend Inference Benchmark

A reproducible inference benchmark project for evaluating numerical
consistency and performance across PyTorch and deployment backend.

The project currently focuses on Apple Silicon and separates model loading,
device transfer, inference execution, synchronization, result validation,
and performance measurement into explicit lifecycle stages.

> Current milestone: Day 2 - PyTorch CPU/MPS correctness baseline and
> PyTorch-to-ONNX export pipline.

# Project Goals

This project aims to build a reproducible workflow for:

 - Comparing PyTorch CPU, PyTorch MPS, and ONNX Runtime inference
 - Verifying numerical consistency before measuring performance
 - Separating initialization, warm-up, execution, synchronization, and transfer
 - Recording environment and model metadata with benchmark results
 - Analyzing exported graphs and backend operator behavior
 - Investing optimization opportunities such as graph optimization,
   precision conversion, quantization, and executor-provider selection

# Current Backend Status

| Backend | Correctness | Performance benchmark | Status |
|---|---:|---:|---|
| PyTorch CPU | Implemented | Planned | Available |
| PyTorch MPS | Implemented | Planned | Available |
| ONNX export | Structural validation | Not applicable | Available |
| ONNX Runtime CPU | Planned | Planned | Day 3 |
| Additional deployment backends | Planned | Planned | Future work |

Performance numbers are intentionally not reported yet. A formal benchmark
requires warm-up, synchronizaiotn, repeated sampling, and statistical
aggregation, whihc will be implemented in later milestones.

# Implemented Features

### Day 1 - PyTorch CPU and MPS Baseline

 - Runtime capability detection for CPU and MPS
 - Explicit device resolution and MPS availability handling
 - MobileNetV3 Small model loading
 - CPU and MPS inference paths
 - Finite-value and output-shape validation
 - Top-1 agreement checking
 - Maximum and mean absolute-error measurement
 - Unit and integration smoke test

### Day 2 - PyTorch to ONNX export

 - PyTorch Dynamo-based ONNX export
 - Fixed-shape FP32 export for MobileNetV3 Small
 - Explicit input and output naming
 - ONNX opset 18 export
 - Embedded model initializers
 - ONNX Checker validation
 - ONNX shape inference
 - Graph input, output, initializer, node, and operator statistics
 - Machine-readable JSON result generation

## Current Reproducible Results

### CPU/MPS Correctness Smoke Test

The same MobileNetV3 Small model, pretrained weights, and synthetic input
were used on both backends.

| Metric | Result |
|---|---:|
| Output shape | `[1, 1000]` |
| CPU Top-1 index | `549` |
| MPS Top-1 index | `549` |
| Maximum absolute error | `0.000007` |
| Mean absolute error | `0.000002` |
| Element-wise tolerance check | Passed |

This is a single-input correctness smoke test. It is not a model-accuracy
evaluation and does not represent a performance improvement.

### ONNX Export Summary

| Property | Result |
|---|---:|
| Model | MobileNetV3 Small |
| Weights | `MobileNet_V3_Small_Weights.DEFAULT` |
| Input random seed | `2026` |
| Precision | FP32 |
| Input | `input: [1, 3, 224, 224]` |
| Output | `logits: [1, 1000]` |
| Dynamic shapes | Disabled |
| ONNX IR version | 10 |
| ONNX opset | 18 |
| Graph nodes | 122 |
| Initializers | 110 |
| Model size | 10,476,133 bytes |
| ONNX Checker | Passed |
| Shape inference | Passed |

The exported graph contains the following operator distribution:

| Operator | Count |
|---|---:|
| Conv | 52 |
| HardSwish | 19 |
| Relu | 14 |
| ReduceMean | 10 |
| HardSigmoid | 9 |
| Mul | 9 |
| Add | 6 |
| Gemm | 2 |
| Reshape | 1 |

The complete machine-readable result is stored in
[`artifacts/results/day2_onnx_graph_summary.json`](artifacts/results/day2_onnx_graph_summary.json).

## Environment

The current reference environment is:

| Component | Version |
|---|---|
| Architecture | Apple Silicon ARM64 |
| Python | 3.11.16 |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| ONNX | 1.22.0 |
| ONNX Script | 0.7.2 |
| Test framework | pytest 9.1.1 |

MPS execution requires a compatible Apple Silicon Mac and a PyTorch build
with MPS support. The CPU path can also be used on other supported platforms.

## Installation

Create and activate a Python 3.11 virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the project and development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify the installation:

```bash
python -m pip check
pytest -v
```

## Usage

Run the PyTorch CPU/MPS correctness smoke test:

```bash
python scripts/day1_smoke.py
```

Export and inspect the fixed-shape ONNX model:

```bash
python scripts/day2_export.py
```

The export command generates:

```text
artifacts/models/mobilenet_v3_small_fp32_opset18.onnx
artifacts/results/day2_onnx_graph_summary.json
```

The generated ONNX binary is excluded from Git because it can be reproduced
from the source code. The compact JSON summary is committed as experiment
evidence.

## Repository Structure

```text
.
├── artifacts/
│   ├── models/                         # Generated model binaries, ignored by Git
│   └── results/                        # Reproducible JSON/CSV results
├── docs/
│   ├── experiment-log.md               # Executed experiments and observations
│   └── learning-notes.md               # Concepts and engineering notes
├── scripts/
│   ├── day1_smoke.py                    # CPU/MPS correctness smoke test
│   └── day2_export.py                   # ONNX export and graph inspection
├── src/
│   └── inference_bench/
│       ├── device.py                    # Device capability and resolution logic
│       ├── inference.py                 # Reference PyTorch inference
│       ├── models.py                    # Model construction
│       └── onnx_export.py               # ONNX export and graph inspection
├── tests/
│   ├── test_device.py
│   ├── test_inference.py
│   ├── test_models.py
│   └── test_onnx_export.py
└── pyproject.toml
```

## Reproducibility Principles

The project follows several rules to avoid misleading benchmark conclusions:

1. Verify numerical correctness before comparing speed.
2. Use the same model weights and input data across backends.
3. Keep model initialization outside the measured inference interval.
4. Warm up each backend before collecting latency samples.
5. Synchronize asynchronous devices at measurement boundaries.
6. Report distributions such as median and percentiles, not only averages.
7. Record software, hardware, model, precision, shape, and backend metadata.
8. Distinguish graph validation from runtime numerical validation.
9. Document limitations and avoid extrapolating from single-input tests.

## Documentation

- [Experiment Log](docs/experiment-log.md)
- [Learning Notes](docs/learning-notes.md)

## Roadmap

- [x] PyTorch CPU/MPS capability detection
- [x] CPU/MPS numerical-consistency smoke test
- [x] Dynamo-based fixed-shape ONNX export
- [x] ONNX graph validation and operator statistics
- [ ] ONNX Runtime CPU inference
- [ ] PyTorch/ONNX Runtime numerical-consistency validation
- [ ] Unified backend lifecycle interface
- [ ] Warm-up and repeated latency sampling
- [ ] Median, percentile, throughput, and variability reporting
- [ ] ResNet50 benchmark coverage
- [ ] Dynamic batch-size experiments
- [ ] Graph optimization and operator-level profiling
- [ ] FP16 and INT8 optimization experiments

## Current Limitations

- Only MobileNetV3 Small is covered.
- The current exported model uses a fixed batch size of one.
- Correctness results use a single synthetic input.
- ONNX Runtime execution has not yet been implemented.
- Formal latency and throughput measurements have not yet been collected.
- Model accuracy on an evaluation dataset has not been measured.