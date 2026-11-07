# PyTorch Cross-Backend Inference Benchmark

A reproducible inference benchmark project for evaluating numerical
consistency and performance across PyTorch and deployment backend.

The project currently focuses on Apple Silicon and separates model loading,
device transfer, inference execution, synchronization, result validation,
and performance measurement into explicit lifecycle stages.

> Current milestone: Day 7 - CPU graph-optimization comparison and
> operator profiling, with validated raw records and reproducible summaries.

# Project Goals

This project aims to build a reproducible workflow for:

 - Comparing PyTorch CPU, PyTorch MPS, and ONNX Runtime inference
 - Verifying numerical consistency before measuring performance
 - Separating initialization, warm-up, execution, synchronization, and transfer
 - Recording environment and model metadata with benchmark results
 - Analyzing exported graphs and backend operator behavior
 - Investigating optimization opportunities such as graph optimization,
   precision conversion, quantization, profiling, and executor-provider selection

# Current Backend Status

| Backend | Correctness | Performance benchmark | Status |
|---|---:|---:|---|
| PyTorch CPU | Implemented | Implemented | Baseline recorded |
| PyTorch MPS | Implemented | Implemented | Available; cross-process latency is state-sensitive |
| ONNX export | Structural validation | Not applicable | Available |
| ONNX Runtime CPU | Implemented | Implemented | Baseline and graph-level comparison recorded |
| ONNX Runtime CoreML | Not Started | Not Started | Future work |

Day 5 reports a prepared-input, concurrency-one latency baseline. The
results do not represent end-to-end application latency or measured
concurrent throughput.

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

### Day 3 - ONNX Runtime CPU Correctness

 - Explicit `CPUExecutionProvider` selection
 - ORT Session creation with graph optimization enabled
 - PyTorch Tensor to NumPy to ORT input path
 - ORT output conversion back to PyTorch Tensor
 - Shared-model PyTorch/ORT comparison
 - Shape and finite-value validation
 - Top-1 agreement checking
 - Maximum and mean absolute-error agreement
 - Configurable `rtol` and `atol`
 - Machine-readable correctness result generation

### Day 4 — Unified Backend Interface

 - Shared lifecycle: input preparation, execution, synchronization, and
   CPU output conversion
 - Independent PyTorch model copies and prepared input snapshots
 - Native Tensor/NumPy outputs preserved during execution
 - Common `run_once()` entry point for three backends
 - Backend configuration metadata and JSON experiment results

All three backends matched a direct PyTorch CPU reference within
`rtol=1e-4` and `atol=1e-4` for the tested FP32 input.

See [the Day 4 result](artifacts/results/day4_backend_correctness.json)
and [the experiment log](docs/experiment-log.md).

### Day 5 — Reproducible Latency Benchmark

 - Shared benchmark harness for PyTorch CPU, PyTorch MPS, and ONNX Runtime CPU
 - Correctness validation before and after latency sampling
 - Explicit warm-up outside the measured interval
 - Per-sample synchronization for asynchronous MPS execution
 - Raw integer-nanosecond sample preservation
 - Mean, P50, P90, P99, standard deviation, and coefficient-of-variation
   statistics
 - Derived serial service-rate reporting for batch-size-one inference
 - Three independent processes with rotated backend execution order
 - Runtime, hardware, thread-policy, source, model, and input metadata
 - Collision-safe machine-readable JSON result persistence

### Day 6 — CPU Intra-op Thread Sweep

- Explicit PyTorch CPU and ONNX Runtime CPU intra-op configurations:
  default, 1, 2, 4, and 6
- Unchanged runtime-default inter-op policies
- Three independent processes per backend and configuration
- Rotated execution order across rounds
- Matched intra-op comparison between the two CPU runtimes
- Process CPU-time to wall-time ratio reporting
- MPS host-thread sensitivity control
- Strict validation of source, hardware, runtime, correctness, thread
  metadata, raw samples, and persisted statistics

### Day 7 — CPU Graph Optimization and Operator Profiling

- Four ORT graph levels: disable, basic, extended, and all
- Explicit intra-op settings of 1 and 6, with three independent benchmark
  processes per configuration and rotated execution order
- Formal latency sessions separated from diagnostic profiling sessions
- PyTorch CPU shape and memory profiling at both thread settings
- ORT active-window operator summaries and four serialized graph summaries
- PyTorch execution isolated from ORT runtime imports
- Complete 24-record benchmark and 10-record profiling validation, including
  shared model, input, ONNX, environment, and producing-commit checks per campaign
- Immutable raw JSON plus five reproducible JSON/CSV summaries
- Three formal campaigns at the same implementation commit

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

### PyTorch CPU / ONNX Runtime CPU Consistency

The same model instance, weights, and input tensor were used to generate the
PyTorch reference output and exported ONNX model.

| Metric | Result |
|---|---:|
| Active Execution Provider | `CPUExecutionProvider` |
| Output shape | `[1, 1000]` |
| PyTorch Top-1 index | `574` |
| ONNX Runtime Top-1 index | `574` |
| Maximum absolute error | `1.1920929e-05` |
| Mean absolute error | `2.6078858e-06` |
| Relative tolerance | `1e-4` |
| Absolute tolerance | `1e-4` |
| Element-wise closeness | Passed |

The complete result is stored in
[`artifacts/results/day3_ort_correctness.json`](artifacts/results/day3_ort_correctness.json).

This is a correctness result, not a latency or throughput result.

### Synchronized Cross-Backend Latency Baseline

The benchmark used pretrained MobileNetV3 Small with a fixed synthetic FP32
input of shape `[1, 3, 224, 224]`, batch size 1, and request concurrency 1.

Each formal run used 50 warm-up calls followed by 300 measured calls. Three
independent processes were executed for each backend with rotated backend
order. The measured interval contains `backend.run()` and backend
synchronization. Model loading, ONNX export, session creation, input
preparation, output conversion, correctness validation, and result
serialization are excluded.

The formal campaign ran on AC power with the battery charged, Low Power Mode
disabled, an idle desktop, and `PYTORCH_ENABLE_MPS_FALLBACK=0`.

| Backend | Reported mean latency | P50 | P90 | P99 | Derived serial service rate |
|---|---:|---:|---:|---:|---:|
| PyTorch CPU | 48.769 ms | 48.690 ms | 49.374 ms | 51.664 ms | 20.50 images/s |
| ONNX Runtime CPU | 1.045 ms | 1.037 ms | 1.159 ms | 1.197 ms | 957.26 images/s |
| PyTorch MPS | 2.74–3.51 ms | State-dependent | State-dependent | State-dependent | 285–365 images/s |

CPU values are the median of the three per-process statistics. Under the
recorded runtime-default thread policies, ONNX Runtime CPU reduced mean
latency by approximately **46.68x** relative to PyTorch CPU for this model
and input. This is an observed backend/runtime result, not an equal-thread
kernel comparison.

MPS produced different stable latency states across independent processes.
Formal round means were `3.465`, `2.741`, and `2.758` ms. Two additional
runs with 1,000 warm-up iterations produced `3.506` and `3.090` ms.
Therefore, the project reports the observed controlled-AC envelope rather
than selecting the fastest individual run.

`Serial service rate` is calculated from mean batch-size-one latency. It is
not measured concurrent throughput.

Formal results are stored in
[`artifacts/results/day5/formal/ac_v2/`](artifacts/results/day5/formal/ac_v2/).
Selected MPS diagnostics are stored in
[`artifacts/results/day5/diagnostics/mps_state/`](artifacts/results/day5/diagnostics/mps_state/).
See the [experiment log](docs/experiment-log.md) for the complete analysis.

### CPU Intra-op Thread Sweep

Day 6 evaluated runtime-default and explicit intra-op configurations of
1, 2, 4, and 6. Each backend/configuration pair ran in three independent
processes. Representative latency is the median of the three process means.

PyTorch CPU was fastest with one explicit intra-op thread at `8.185 ms`.
The 2-, 4-, and 6-thread configurations were respectively `2.71x`, `3.66x`,
and `5.96x` slower than the single-thread result. Day 7 profiling now
localizes much of the instrumented increase to small-channel convolution
execution; a specific scheduling or kernel-level cause remains unproven.

ONNX Runtime CPU improved from `2.615 ms` at one thread to `1.050 ms` at
six threads, a `2.49x` speedup. Its runtime-default result was `1.051 ms`,
only `0.125%` slower than explicit six-thread execution. The default policy
used approximately `9.07` process CPU core equivalents, compared with
`5.95` for the explicit six-thread configuration.

At matched intra-op settings of 1, 2, 4, and 6, PyTorch CPU latency was
respectively `3.13x`, `13.42x`, `24.97x`, and `46.48x` higher than ONNX
Runtime CPU latency. These are runtime-level comparisons; inter-op defaults,
graph optimization, kernels, memory planning, and thread-pool behavior still
differ.

MPS representative means ranged from `2.986 ms` to `3.424 ms`, while
individual configurations showed `9.24%–22.47%` cross-process ranges.
The swept value controls PyTorch CPU intra-op behavior and is not an MPS
worker-thread count. No MPS thread optimum is inferred from this experiment.

Formal results are stored in
[`artifacts/results/day6/formal/ac_v1/`](artifacts/results/day6/formal/ac_v1/).

### CPU Graph Optimization and Operator Profiling

Day 7 measures ORT graph optimization and CPU operator behavior through
three formal campaigns at commit `22753d0`. The workload is MobileNetV3
Small, FP32, `[1, 3, 224, 224]`, and request concurrency 1.

Each graph/thread configuration has nine independent latency processes
across three campaigns. Each process uses 50 warm-up calls and 300 measured
calls. Representative latency is the median of all nine process means;
the range shows their minimum and maximum. Raw samples and slow processes
remain included.

| ORT graph level | t1 representative latency | t1 process-mean range | t6 representative latency | t6 process-mean range |
|---|---:|---:|---:|---:|
| disable | 2.652 ms | 2.626–2.694 ms | 0.937 ms | 0.927–1.187 ms |
| basic | 2.641 ms | 2.631–2.694 ms | 0.939 ms | 0.927–0.965 ms |
| extended | 2.624 ms | 2.597–2.645 ms | **0.902 ms** | 0.898–0.912 ms |
| all | **2.566 ms** | 2.549–2.581 ms | 1.043 ms | 1.035–1.047 ms |

The measurements establish a configuration-dependent optimization pattern:
`all` gives the lowest t1 latency, while `extended` gives the lowest t6
latency. Relative to same-thread disable, their aggregate improvements are
3.27% and 3.66%. At t6, all is 15.55% slower than extended; their process-mean
ranges remain separate. The per-campaign gap spans 14.24%–15.86%.

Basic and disable produce identical graph bytes and similar latency;
their sub-1% differences change direction across campaigns. Extended's
t1 improvement is smaller: 1.07% in the aggregate, with per-campaign
improvements of 0.21%–2.21%.

The original graph has 122 nodes. Disable/basic, extended, and all have
141, 118, and 114, with matching hashes at each level across campaigns.
The all graph adds two Transpose nodes and one `com.microsoft::NhwcFusedConv`.
Fewer nodes therefore do not guarantee lower inference latency.

Three PyTorch captures per thread setting identify a consistent
small-channel convolution hotspot. `aten::_slow_conv2d_forward` accounts
for a median 50.72% of self CPU time at t1 and 80.39% at t6. ORT all/t6
profiles attribute 7.59%–9.35% of Node duration to Transpose. These results
guide targeted kernel and layout diagnosis; profile durations are separate
from formal latency.

The campaigns use AC power at 100% battery. Execution completed with
`ORT_DISABLE_TELEMETRY` unset; PyTorch profiling uses the isolated import
path. The experiment log records power snapshots, process variability,
and the earlier ORT telemetry shutdown incident.

Raw records and per-campaign summaries are retained in
[ac_v1](artifacts/results/day7/formal/ac_v1/),
[ac_v2](artifacts/results/day7/formal/ac_v2/), and
[ac_v3](artifacts/results/day7/formal/ac_v3/).
See the [experiment log](docs/experiment-log.md) for aggregate statistics,
measurement details, and local archive paths.

## Environment

The current reference environment is:

| Component | Version |
|---|---|
| Hardware | Apple M5 Pro, 48 GB unified memory |
| Architecture | Apple Silicon ARM64 |
| Operating system | macOS 26.5.1 |
| Python | 3.11.16 |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| ONNX | 1.22.0 |
| ONNX Script | 0.7.2 |
| ONNX Runtime | 1.30.0 |
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

Run the PyTorch CPU / ONNX Runtime CPU correctness validation:

```bash
python scripts/day3_ort_correctness.py
```

Run correctness validation through the unified backend interface:

```bash
python scripts/day4_backends_smoke.py
```

Run one correctness-gated benchmark process:

```bash
python scripts/day5_benchmark.py \
  --backend pytorch_cpu \
  --warmup 50 \
  --iterations 300 \
  --round 1
```

Supported backends are pytorch_cpu, pytorch_mps, and ort_cpu.
Run the complete three-round baseline campaign:

```bash
export PYTORCH_ENABLE_MPS_FALLBACK=0
export DAY5_NOTES="AC power; battery charged; powermode=0; idle desktop"

bash scripts/day5_run_baseline.sh

unset DAY5_NOTES
unset PYTORCH_ENABLE_MPS_FALLBACK
```

Each process writes a machine-readable JSON record containing raw latency
samples, summary statistics, correctness results, runtime configuration,
environment metadata, and reproducibility fingerprints.

### Day6 CPU intra-op thread sweep

Run the lightweight 45-process matrix before collecting formal results:

```bash
DAY6_PYTHON_BIN=.venv/bin/python \
DAY6_SMOKE_OUTPUT_DIR=artifacts/results/day6/local_smoke/cpu_threads/raw \
bash scripts/day6_run_cpu_thread_sweep_lite.sh
```

The lite campaign uses two warm-up calls and one measured call per process.
It verifies matrix expansion, backend execution, correctness gates, thread
metadata, result persistence, and collision handling. Its output is local
smoke evidence and must not be interpreted as a latency result. The default
smoke directory must contain no existing JSON files.

Before the formal campaign, require a clean Git worktree, connect AC power,
and remove inherited thread-count overrides:

```bash
git status --short
unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS
unset VECLIB_MAXIMUM_THREADS
unset NUMEXPR_NUM_THREADS
```

Run the 45-process formal-protocol campaign into staging:

```bash
DAY6_PYTHON_BIN=.venv/bin/python \
DAY6_OUTPUT_DIR=artifacts/results/day6/staging/cpu_threads/raw \
bash scripts/day6_run_cpu_thread_sweep.sh
```

The staging campaign uses 50 warm-up calls and 300 measured calls for each
backend/configuration pair. It requires a fixed Git commit, a clean worktree,
unset thread environment variables, and an output directory containing no
existing JSON files.

After reviewing a completed staging campaign, copy the 45 accepted raw
records into a new versioned formal archive without modifying or overwriting
the staging evidence:

```bash
mkdir -p artifacts/results/day6/formal/ac_v1/raw

cp -n \
  artifacts/results/day6/staging/cpu_threads/raw/*.json \
  artifacts/results/day6/formal/ac_v1/raw/
```

If `formal/ac_v1` already contains records or summaries from an earlier
accepted campaign, create a new versioned directory such as `ac_v2` instead
of replacing them.

Validate and summarize the archived formal campaign:

```bash
python scripts/day6_summarize_cpu_threads.py \
  --input-dir artifacts/results/day6/formal/ac_v1/raw \
  --output-dir artifacts/results/day6/formal/ac_v1
```

The summarizer validates the complete experiment contract before creating
`summary.json` and `summary.csv`. It reads the archived raw records without
modifying them and refuses to overwrite existing summary files.

### Day7 graph optimization and profiling

Before collecting formal results, commit the implementation changes, require
a clean Git worktree, connect AC power, and remove inherited thread-count
overrides:

```bash
git status --short
pmset -g batt

unset OMP_NUM_THREADS
unset MKL_NUM_THREADS
unset OPENBLAS_NUM_THREADS
unset VECLIB_MAXIMUM_THREADS
unset NUMEXPR_NUM_THREADS

unset ORT_DISABLE_TELEMETRY

export DAY7_PYTHON_BIN=.venv/bin/python
DAY7_IMPLEMENTATION_COMMIT="$(git rev-parse HEAD)"
```

The retained Day7 campaigns use the ONNX Runtime default telemetry policy,
so `ORT_DISABLE_TELEMETRY` is left unset. PyTorch profiling uses an isolated
import path and does not initialize ONNX Runtime.

Before the formal campaign, smoke-test the graph benchmark, PyTorch profiler,
and ONNX Runtime profiler in separate ignored directories:

```bash
DAY7_SMOKE_ID="$(date -u +%Y%m%dT%H%M%SZ)"
DAY7_SMOKE_OUTPUT="artifacts/results/day7/staging/smoke/${DAY7_SMOKE_ID}"
DAY7_SMOKE_ARCHIVE="artifacts/results/day7/local_archive/smoke/${DAY7_SMOKE_ID}"

"$DAY7_PYTHON_BIN" scripts/day7_graph_optimization.py \
  --graph-optimization-level all \
  --intra-op-threads 1 \
  --round 1 \
  --warmup 2 \
  --iterations 5 \
  --output-dir "${DAY7_SMOKE_OUTPUT}/graph_optimization/raw" \
  --notes "Day7 smoke; excluded from formal results"

"$DAY7_PYTHON_BIN" scripts/day7_profile_cpu.py \
  --backend pytorch_cpu \
  --intra-op-threads 1 \
  --warmup 10 \
  --active-runs 10 \
  --output-dir "${DAY7_SMOKE_OUTPUT}/profiling/raw" \
  --trace-dir "${DAY7_SMOKE_ARCHIVE}/traces" \
  --notes "Day7 smoke; excluded from formal results"

"$DAY7_PYTHON_BIN" scripts/day7_profile_cpu.py \
  --backend ort_cpu \
  --graph-optimization-level all \
  --intra-op-threads 1 \
  --warmup 10 \
  --active-runs 10 \
  --output-dir "${DAY7_SMOKE_OUTPUT}/profiling/raw" \
  --trace-dir "${DAY7_SMOKE_ARCHIVE}/traces" \
  --optimized-model-path \
    "${DAY7_SMOKE_ARCHIVE}/optimized_models/all.onnx" \
  --notes "Day7 smoke; excluded from formal results"
```

These smoke runs verify the three execution paths, correctness gates, trace
generation, and optimized-model export. Their results are local smoke
evidence and must not be interpreted as formal latency or profiling results.

Use the next unused versioned batch name so existing staging records, traces,
optimized graphs, and formal results remain unchanged. For example:

```bash
DAY7_BATCH_ID="ac_v4"

export DAY7_GRAPH_OUTPUT_DIR="artifacts/results/day7/staging/${DAY7_BATCH_ID}/graph_optimization/raw"
export DAY7_PROFILE_OUTPUT_DIR="artifacts/results/day7/staging/${DAY7_BATCH_ID}/profiling/raw"
export DAY7_TRACE_DIR="artifacts/results/day7/local_archive/${DAY7_BATCH_ID}/traces"
export DAY7_OPTIMIZED_MODEL_DIR="artifacts/results/day7/local_archive/${DAY7_BATCH_ID}/optimized_models"
export DAY7_NOTES="Day7 ${DAY7_BATCH_ID}; AC power; ORT_DISABLE_TELEMETRY=unset (runtime default)"

git status --short
test "$(git rev-parse HEAD)" = "$DAY7_IMPLEMENTATION_COMMIT"
```

Confirm that `git status --short` produces no output, the recorded HEAD has
not changed, and the four destination directories contain no artifacts from
an earlier campaign.

Run the 24-process graph-optimization benchmark followed by the ten-process
profiling campaign at the same commit:

```bash
bash scripts/day7_run_graph_optimization.sh &&
test "$(git rev-parse HEAD)" = "$DAY7_IMPLEMENTATION_COMMIT" &&
bash scripts/day7_run_profiling.sh

test "$(git rev-parse HEAD)" = "$DAY7_IMPLEMENTATION_COMMIT"
```

The graph campaign evaluates four ONNX Runtime graph-optimization levels at
one and six intra-op threads. Each configuration has three independent
processes, with 50 warm-up calls and 300 measured calls per process.

The profiling campaign creates two PyTorch profiles and eight ONNX Runtime
profiles. Each profile uses ten external warm-up calls and ten active runs.
The campaign produces ten result records, ten traces, and four serialized
optimized ONNX graphs. Profiling durations include profiler overhead and are
not formal latency measurements.

After reviewing the completed staging campaign, copy the 34 raw JSON records
unchanged into a new formal archive:

```bash
DAY7_FORMAL_DIR="artifacts/results/day7/formal/${DAY7_BATCH_ID}"

mkdir -p artifacts/results/day7/formal
mkdir "$DAY7_FORMAL_DIR"
mkdir -p \
  "$DAY7_FORMAL_DIR/graph_optimization/raw" \
  "$DAY7_FORMAL_DIR/profiling/raw"

cp -n \
  "$DAY7_GRAPH_OUTPUT_DIR"/*.json \
  "$DAY7_FORMAL_DIR/graph_optimization/raw/"

cp -n \
  "$DAY7_PROFILE_OUTPUT_DIR"/*.json \
  "$DAY7_FORMAL_DIR/profiling/raw/"
```

If the selected formal batch directory already exists, choose a new version
instead of modifying or overwriting the earlier campaign.

Validate the archived graph and profiling matrices and generate the five
formal summaries:

```bash
"$DAY7_PYTHON_BIN" scripts/day7_summarize.py \
  --graph-input-dir "$DAY7_FORMAL_DIR/graph_optimization/raw" \
  --profile-input-dir "$DAY7_FORMAL_DIR/profiling/raw" \
  --output-root "$DAY7_FORMAL_DIR"
```

The summarizer validates the 24 benchmark records, ten profiling records,
workload identity, source identity, and graph/profile correspondence before
creating:

```text
graph_optimization/summary.json
graph_optimization/summary.csv
profiling/operator_summary.json
profiling/operator_summary.csv
profiling/graph_summary.json
```

Commit the 34 immutable raw JSON records, five summaries, and corresponding
analysis documentation. Traces and optimized ONNX binaries remain in the
ignored local archive; their paths and SHA-256 values are retained in the
profiling records. Validate and summarize each campaign independently before
combining process-level results across campaigns.

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
│   ├── day1_smoke.py                    # PyTorch CPU/MPS correctness smoke test
│   ├── day2_export.py                   # ONNX export and graph inspection
│   ├── day3_ort_correctness.py          # PyTorch/ORT CPU numerical-consistency experiment
│   ├── day4_backends_smoke.py           # Unified three-backend correctness smoke test
│   ├── day5_benchmark.py                # Single-process benchmark entry
│   ├── day5_run_baseline.sh             # Rotated three-round campaign
│   ├── day6_cpu_threads.py              # Single-process thread experiment
│   ├── day6_run_cpu_thread_sweep_lite.sh # Lightweight matrix smoke test
│   ├── day6_run_cpu_thread_sweep.sh      # Formal 45-process thread campaign
│   ├── day6_summarize_cpu_threads.py     # Result validation and JSON/CSV aggregation
│   ├── day7_graph_optimization.py       # Single-process ORT graph-level benchmark
│   ├── day7_profile_cpu.py              # Single-process CPU operator profile
│   ├── day7_run_graph_optimization.sh   # Formal 24-process latency matrix
│   ├── day7_run_profiling.sh            # Ten diagnostic profiles and graph exports
│   └── day7_summarize.py                # Campaign validation and five summaries
├── src/
│   └── inference_bench/
│       ├── backends.py                  # Backend protocol, adapters, and inference lifecycle
│       ├── benchmark.py                 # Sampling, statistics, and persistence
│       ├── device.py                    # Device capability detection and resolution
│       ├── graph_optimization.py        # Formal graph-level matrix validation
│       ├── inference.py                 # Reference PyTorch inference
│       ├── models.py                    # Model construction
│       ├── onnx_export.py               # ONNX export and graph inspection
│       ├── ort_backend.py               # ORT CPU session creation and inference adapter
│       ├── profiling.py                 # Trace and ONNX graph summaries
│       ├── profiling_campaign.py        # Profile matrix and cross-campaign validation
│       ├── result_validation.py         # Shared experiment-record validation
│       ├── thread_sweep.py              # CPU thread-sweep validation and analysis
│       └── validation.py                # Backend-independent output validation and error metrics
├── tests/
│   ├── test_backends.py
│   ├── test_benchmark_integration.py
│   ├── test_benchmark.py
│   ├── test_device.py
│   ├── test_graph_optimization.py
│   ├── test_inference.py
│   ├── test_models.py
│   ├── test_onnx_export.py
│   ├── test_ort_backend.py
│   ├── test_profiling.py
│   ├── test_profiling_campaign.py
│   ├── test_result_validation.py
│   ├── test_runtime_imports.py
│   ├── test_thread_sweep.py
│   └── test_validation.py
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

- [x] PyTorch CPU/MPS baseline and numerical validation
- [x] ONNX export and graph validation
- [x] ONNX Runtime CPU integration and cross-backend correctness validation
- [x] Unified backend inference lifecycle
- [x] Reproducible synchronized cross-backend latency benchmark
- [x] CPU intra-op thread control and sweep
- [x] CPU graph optimization and operator profiling
- [ ] Model and workload matrix expansion
- [ ] Precision, quantization, and deployment-backend optimization
- [ ] Concurrent serving and end-to-end pipeline benchmarking

## Current Limitations

- Only MobileNetV3 Small has been benchmarked.
- Results use one fixed synthetic FP32 input with batch size 1.
- The same prepared input is reused, so the measurements represent a
  hot-cache microbenchmark.
- Results cover one Apple Silicon system and should not be interpreted as
  general hardware rankings.
- Matched PyTorch CPU and ONNX Runtime CPU intra-op configurations have been
  measured while inter-op policies remain runtime defaults. This is not a
  complete equalization of runtime thread activity.
- The timed interval excludes preprocessing, input transfer, output
  conversion, postprocessing, model loading, and backend initialization.
- Derived serial service rate is not measured concurrent throughput.
- MPS latency is state-dependent across independent processes, and a fixed
  warm-up count does not guarantee cross-process convergence.
- CPU profiling identifies convolution hotspots and layout-related work,
  while causal scheduling, kernel, and memory explanations remain open.
- Instrumented durations include profiler overhead and are used for operator
  attribution; formal latency is measured independently in separate processes
  with profiling disabled.
- CoreML Execution Provider, FP16, and INT8 execution have not been evaluated.
- Model accuracy on a real evaluation dataset has not been measured.
