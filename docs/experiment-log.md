# Experiment Log

This document records executed experiments, observed results, validation
boundaries, and deferred work. Conceptual explanations are maintained
separately in [learning-notes.md](learning-notes.md).

## Day 1: PyTorch CPU and MPS Baseline

### Objective

Establish a minimal PyTorch inference path that can:

- Detect CPU and MPS runtime capabilities
- Resolve configured device names
- Execute the same model on CPU and MPS
- Compare the resulting tensors for numerical consistency
- Preserve a testable boundary between capability detection and execution

### Reference Environment

| Component | Value |
|---|---|
| Architecture | ARM64 |
| Python | 3.11.16 |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| MPS built | True |
| MPS available | True |

### Device Resolution Boundary

`resolve_device()` translates a user- or configuration-provided device name
into a `torch.device` descriptor. Its unit tests verify:

- Input normalization
- Supported-device mapping
- MPS availability handling
- Rejection of unknown device names

Creating a `torch.device` does not prove that a real workload can execute on
the hardware. Actual CPU and MPS tensor execution is therefore verified
separately through integration smoke tests.

### Backend Lifecycle Separation

The current reference inference function combines:

- Model transfer
- Input transfer
- Model execution
- Device synchronization
- Output transfer

Formal benchmarking will separate these operations into backend lifecycle
stages so that only the intended inference interval is measured.

### CPU/MPS Numerical-Consistency Smoke Test

CPU and MPS inference outputs were compared using identical pretrained
weights and the same synthetic input.

The smoke test verifies:

- Identical output shapes
- Finite outputs without NaN or infinity
- Top-1 agreement
- Maximum and mean absolute error
- Element-wise closeness using configured absolute and relative tolerances

Observed results:

| Metric | Result |
|---|---:|
| Output shape | `[1, 1000]` |
| CPU Top-1 index | `549` |
| MPS Top-1 index | `549` |
| Maximum absolute error | `0.000007` |
| Mean absolute error | `0.000002` |
| Correctness result | Passed |

This is a single-input correctness smoke test. It does not represent model
accuracy or backend performance.

### Test Refinement Plan

The current device-capability test verifies that the project abstraction
reports values consistent with the underlying Python and PyTorch runtime APIs.

When benchmark-result persistence is implemented, extend the test coverage
to include:

- Metadata schema validation
- Serialization
- Result read-back
- Required environment fields
- Rejection of incomplete benchmark records

### Day 1 Limitations

- Only one model is tested.
- Only one synthetic input is tested.
- No warm-up or repeated latency sampling is implemented.
- Model initialization and device transfer are not yet separated from all
  inference entry points.
- No ONNX Runtime backend is available.

---

## Day 2: PyTorch to ONNX Export and Graph Inspection

### Objective

Create a reproducible fixed-shape ONNX export pipeline for MobileNetV3 Small
and verify the exported graph before introducing an ONNX runtime backend.

The Day 2 pipeline covers:

1. PyTorch graph capture through the Dynamo exporter
2. Operator decomposition
3. Translation into ONNX operators
4. ONNX graph optimization
5. Model serialization
6. ONNX structural validation
7. Shape inference
8. Graph and operator statistics

Runtime execution and numerical comparison with ONNX Runtime are intentionally
deferred to Day 3.

### Dependency Versions

| Component | Version |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.14.0 |
| torchvision | 0.29.0 |
| ONNX | 1.22.0 |
| ONNX Script | 0.7.2 |
| pytest | 9.1.1 |

### Export Configuration

| Property | Configuration |
|---|---|
| Model | MobileNetV3 Small |
| Weights | `MobileNet_V3_Small_Weights.DEFAULT` |
| Input random seed | `2026` |
| Precision | FP32 |
| Export device | CPU |
| Model mode | Evaluation |
| Input name | `input` |
| Input shape | `[1, 3, 224, 224]` |
| Output name | `logits` |
| Output shape | `[1, 1000]` |
| Dynamic shapes | Disabled |
| ONNX opset | 18 |
| Graph capture | Dynamo exporter |
| Graph optimization | Enabled |
| External weight data | Disabled |
| Initializers as inputs | Disabled |
| ONNX Runtime verification | Disabled until Day 3 |

The export script reuses the Day 1 model-loading path so that model
construction and weight selection do not diverge between milestones.

### Export Pipeline Result

The exporter completed all reported stages successfully:

```text
Obtain model graph using torch.export.export(strict=False)
Run decompositions
Translate the graph into ONNX
Optimize the ONNX graph
```

The resulting model passed:

- `onnx.checker.check_model()`
- ONNX shape inference with type checking
- A second checker pass after shape inference

### Graph Summary

| Property | Result |
|---|---:|
| ONNX IR version | 10 |
| ONNX opset | 18 |
| Producer | PyTorch 2.14.0 |
| Node count | 122 |
| Initializer count | 110 |
| File size | 10,476,133 bytes |
| Input | `input: [1, 3, 224, 224]` |
| Output | `logits: [1, 1000]` |

The file size is approximately 10.0 MiB or 10.48 MB.

### Operator Distribution

| Operator | Count |
|---|---:|
| Add | 6 |
| Conv | 52 |
| Gemm | 2 |
| HardSigmoid | 9 |
| HardSwish | 19 |
| Mul | 9 |
| ReduceMean | 10 |
| Relu | 14 |
| Reshape | 1 |
| **Total** | **122** |

The operator-count total equals the graph node count.

### Graph Observations

The distribution is consistent with the expected MobileNetV3 structure:

- `Conv` includes spatial, depthwise, and pointwise convolution operations.
- `HardSwish` and `HardSigmoid` represent characteristic MobileNetV3
  activation and gating behavior.
- `ReduceMean` and `Mul` are associated primarily with squeeze-and-excitation
  aggregation and channel gating.
- `Add` represents residual connections.
- `Gemm` represents the classifier's fully connected operations.
- `Reshape` connects convolutional features with the classifier representation.

No `BatchNormalization` node appears in the final optimized graph. Based on
the evaluation-mode export and optimized graph, this suggests that BatchNorm
parameters were folded into adjacent operations. This is an inference from
the graph statistics and should be confirmed through graph visualization or
initializer inspection before making a stronger operator-fusion claim.

### Test Results

The complete project test suite produced:

```text
10 passed in 2.49s
```

This includes:

- Four device tests
- Two reference-inference tests
- Two model-loading tests
- Two ONNX export and inspection tests

The Day 1 CPU/MPS smoke test was also rerun after the Day 2 implementation and
continued to pass with the same observed maximum absolute error and Top-1
agreement.

### Artifact Policy

The generated model is stored locally at:

```text
artifacts/models/mobilenet_v3_small_fp32_opset18.onnx
```

The ONNX binary is excluded from Git because it can be regenerated from the
source code.

The compact graph summary is stored at:

```text
artifacts/results/day2_onnx_graph_summary.json
```

The JSON summary is committed to provide machine-readable experiment evidence.

### Validation Boundary

Passing ONNX Checker and shape inference establishes that:

- The serialized model is structurally valid.
- Nodes conform to the selected ONNX operator-set schema.
- Graph references are consistent.
- Input and output shapes can be represented and inferred.

It does not establish that:

- ONNX Runtime supports every node in the graph.
- The exported graph produces numerically equivalent output.
- Top-1 results match PyTorch.
- The exported graph is faster than PyTorch.
- The graph is optimally partitioned for a particular backend.

These properties require runtime execution and benchmarking.

### Day 2 Limitations

- Only MobileNetV3 Small is exported.
- Only FP32 is covered.
- Only a fixed `[1, 3, 224, 224]` input is covered.
- Numerical comparison with an ONNX runtime is not yet implemented.
- Operator statistics describe the final exported graph, not the original
  number of PyTorch modules.
- The absence of a node type alone does not prove a particular backend fusion.
- No latency or throughput measurement is reported.

---

## Day 3: ONNX Runtime CPU Execution and Numerical Consistency

### Objective

Execute the Day 2 ONNX model through ONNX Runtime's CPU Execution Provider
and compare its output with PyTorch CPU using the same model instance,
weights, and input tensor.

Performance measurement is intentionally excluded from this milestone.

### Runtime Environment

| Component | Version |
|---|---|
| Python | 3.11.16 |
| PyTorch | 2.14.0 |
| ONNX Runtime | 1.30.0 |
| Architecture | Apple Silicon ARM64 |
| Model | MobileNetV3 Small |
| Weights | `MobileNet_V3_Small_Weights.DEFAULT` |
| Precision | FP32 |
| ONNX opset | 18 |

### Execution Provider Isolation

The installed ONNX Runtime package reported:

```text
CoreMLExecutionProvider
AzureExecutionProvider
CPUExecutionProvider
```

The Day 3 Session was explicitly created with:

```python
providers=["CPUExecutionProvider"]
```

The active provider list was:

```text
CPUExecutionProvider
```

Available providers describe the capabilities compiled into the installed
runtime. Active providers describe the providers assigned to a particular
Session. The presence of CoreML or Azure in the available list does not mean
that they executed any nodes in this experiment.

### Session Configuration

The Session uses:

- `CPUExecutionProvider`
- `ORT_ENABLE_ALL` graph optimization
- Sequential execution mode
- Default ORT intra-op thread configuration
- Default ORT inter-op thread configuration

Thread-count and execution-mode experiments are deferred until the benchmark
harness can isolate them as controlled variables.

### Input and Output Contract

| Property | Value |
|---|---|
| Input name | `input` |
| Input shape | `[1, 3, 224, 224]` |
| Input type | `tensor(float)` |
| Output name | `logits` |
| Output shape | `[1, 1000]` |
| Output type | `tensor(float)` |

### Correctness Method

The script performs the following sequence:

1. Construct one PyTorch model instance.
2. Generate one deterministic input tensor.
3. Run PyTorch CPU inference.
4. Export that same model instance to ONNX.
5. Create an ORT Session restricted to the CPU EP.
6. Run ORT inference with the same input values.
7. Validate shape and finite values.
8. Compare Top-1 indices.
9. Calculate maximum and mean absolute error.
10. Apply element-wise relative and absolute tolerances.

Using the same in-memory model instance prevents model reconstruction or
random initialization from becoming a hidden comparison variable.

### Numerical-Consistency Result

| Metric | Result |
|---|---:|
| Reference shape | `[1, 1000]` |
| Candidate shape | `[1, 1000]` |
| PyTorch Top-1 index | `574` |
| ORT Top-1 index | `574` |
| Top-1 agreement | Passed |
| Maximum absolute error | `1.1920928955078125e-05` |
| Mean absolute error | `2.6078857899847208e-06` |
| Relative tolerance | `1e-4` |
| Absolute tolerance | `1e-4` |
| Element-wise closeness | Passed |

The observed maximum absolute error is approximately 11.9% of the configured
absolute tolerance. The output difference is therefore comfortably within
the selected FP32 correctness threshold for this input.

The Top-1 index differs from the Day 1 result because Day 1 and Day 3 use
different synthetic inputs. Cross-backend agreement must be evaluated within
the same experiment and input, not by comparing class indices across unrelated
inputs.

### Test Results

The complete test suite produced:

```text
16 passed in 3.38s
```

The six Day 3 test items cover:

- Error and Top-1 metric calculation
- Shape-mismatch rejection
- Non-finite reference rejection
- Non-finite candidate rejection
- CPU-only ORT Session creation
- Real PyTorch/ORT numerical consistency

### Artifact

The machine-readable result is stored at:

```text
artifacts/results/day3_ort_correctness.json
```

The result records:

- Model and weight identity
- Model and input seeds
- ONNX opset
- PyTorch and ORT versions
- Available and active providers
- Session input/output contract
- Error metrics
- Tolerances
- Top-1 and allclose results

### Validation Boundary

This experiment establishes that, for the tested model and input:

- ONNX Runtime can load the exported graph.
- All graph nodes can execute through the CPU EP.
- ORT produces the expected output shape and type.
- Outputs contain no NaN or infinity.
- PyTorch and ORT agree on Top-1.
- Tensor differences remain within the selected FP32 tolerances.

It does not establish that:

- ONNX Runtime is faster than PyTorch.
- ORT Session initialization is inexpensive.
- CoreML EP can execute the graph.
- All possible model inputs produce equivalent outputs.
- The selected tolerance is appropriate for every model or precision.
- Default ORT threading is optimal for this hardware.

### Day 3 Limitations

- Only one model, input shape, batch size, precision, and sample are tested.
- Only CPU Execution Provider execution is validated.
- Graph initialization and optimization time are not measured separately.
- NumPy conversion and output-copy costs are not benchmarked.
- No warm-up or latency distribution is collected.

---

## Day 4: Unified Backend Interface

### Objective

Expose PyTorch CPU, PyTorch MPS, and ONNX Runtime CPU through a common
inference lifecycle while preserving each backend's native execution data.

Separate backend initialization, input preparation, execution,
synchronization, and output conversion to support subsequent benchmarking.

### Interface

| Stage | Interface | Responsibility |
|---|---|---|
| Initialization | Constructor | Create an independent model or ORT Session |
| Input preparation | `prepare_input()` | Validate and prepare an owned input snapshot |
| Execution | `run()` | Execute using prepared input and return native output |
| Synchronization | `synchronize()` | Wait for outstanding backend work |
| Output conversion | `to_cpu()` | Return a CPU Tensor for validation |
| Metadata | `metadata()` | Report backend identity and configuration |

`run_once()` invokes:

```text
prepare_input → synchronize → run → synchronize → to_cpu
```

### Ownership and Native Data

Each PyTorch backend owns an independent copy of the source model.
Backend initialization does not mutate the caller's model mode or device.

Prepared inputs are independent snapshots. Subsequent modification of the
caller's input does not affect the prepared workload.

Native execution types are:

| Backend | Prepared input | Native output |
|---|---|---|
| PyTorch CPU | CPU Tensor | CPU Tensor |
| PyTorch MPS | MPS Tensor | MPS Tensor |
| ORT CPU | NumPy array | NumPy array |

### Experiment Configuration

- Architecture: ARM64
- macOS: 26.5.1
- Python: 3.11.16
- PyTorch: 2.14.0
- ONNX Runtime: 1.30.0
- Model: MobileNetV3 Small
- Weights: `MobileNet_V3_Small_Weights.DEFAULT`
- Precision: FP32
- Input shape: `[1, 3, 224, 224]`
- Model seed: 2026
- Input seed: 2027
- ONNX opset: 18
- Reference: direct PyTorch CPU inference
- Tolerances: `rtol=1e-4`, `atol=1e-4`

The reference was computed directly from the source model, outside the
new backend adapters.

### Observed Results

| Backend | Top-1 | Maximum absolute error | Mean absolute error | Allclose |
|---|---:|---:|---:|---|
| PyTorch CPU | 574 | 0 | 0 | Passed |
| ORT CPU | 574 | 1.1920928955078125e-05 | 2.6078857899847208e-06 | Passed |
| PyTorch MPS | 574 | 8.106231689453125e-06 | 1.3067797226540279e-06 | Passed |

All outputs had shape `[1, 1000]`. Three backends executed and none were
skipped.

The ORT errors matched the recorded Day 3 result for the same model and
input configuration.

### Runtime Configuration Observations

- PyTorch reported 6 intra-op threads and 18 inter-op threads.
- ORT reported 0 for both thread settings, indicating runtime defaults.
- ORT registered only `CPUExecutionProvider`.
- ORT used `ORT_ENABLE_ALL` and `ORT_SEQUENTIAL`.
- `PYTORCH_ENABLE_MPS_FALLBACK` was unset when metadata was collected.

Thread settings are configuration values, not measurements of active
worker threads. The two CPU runtimes have not been configured for an
equal-thread performance comparison.

MPS fallback environment metadata is not a per-operator execution trace.

### Test Evidence

The submitted test run reported:

```text
29 passed in 4.64s
```

The 13 new test items cover:

- Native execution, input ownership, and source-model preservation:
  3 backend cases
- Invalid dtype and shape rejection: 6 cases
- Shared `run_once()` execution: 3 backend cases
- Lifecycle call order: 1 case

All MPS cases executed rather than being skipped.

### Artifacts

- Generated model:
  `artifacts/models/day4_mobilenet_v3_small_fp32.onnx`
- Committed result:
  `artifacts/results/day4_backend_correctness.json`
- Result schema version: 1

### Limitations

- The real-model smoke test covers one synthetic input at batch size one.
- Only FP32 and a single-input, single-output model are covered.
- Protocol annotations do not perform runtime contract validation.
- The lifecycle-order test checks orchestration, not GPU timing accuracy.
- No per-operator MPS profiling was collected.
- No latency, throughput, or speedup was measured.
- Test-suite duration is not inference latency.

---

## Day 5: Synchronized Cross-Backend Latency Benchmark

### Objective

Build a reproducible latency harness over the Day 4 backend abstraction and
separate correctness, preparation, warm-up, timed execution, synchronization,
output conversion, statistics, and persistence into explicit lifecycle
stages.

### Implementation verification

The Day 5 implementation added:

- `BenchmarkConfig` validation
- synchronized nanosecond sampling with `perf_counter_ns`
- linear P50/P90/P99 calculation
- population standard deviation and coefficient of variation
- derived batch-1 serial service rate
- raw integer sample persistence
- exclusive JSON creation to prevent accidental overwrite
- real-backend integration coverage for PyTorch CPU, PyTorch MPS, and ONNX
  Runtime CPU

The complete project test run collected 55 tests and reported `55 passed`.

### Formal measurement scope

The timed region contains only:

1. `backend.run(prepared_input)`
2. `backend.synchronize()`

The following operations are outside the timed region:

- model loading and ONNX export
- backend initialization
- input validation and backend-native preparation
- warm-up
- pre-sample synchronization
- output transfer to CPU
- correctness validation
- explicit output-reference release
- JSON serialization

The benchmark uses MobileNetV3-Small, FP32, batch size 1, a fixed
`[1, 3, 224, 224]` synthetic input, request concurrency 1, 50 warm-up
iterations, and 300 recorded iterations. Three independent processes rotate
backend order. Each JSON file records the environment, runtime versions,
backend settings, Git commit, source hash, model/input hashes, correctness
metrics, raw samples, and summary statistics.

### Formal AC v2 campaign

The retained formal campaign used:

- Git commit `f2633de90fe12c71e8808ea376487ea8b2700972`
- AC power with the battery charged
- `powermode=0`
- an idle desktop
- `PYTORCH_ENABLE_MPS_FALLBACK=0`
- source SHA-256
  `bdd50a2e0b186669f7d9b10e13adbe2d64a202c132b8b94dede8ba85c6be257c`

All before/after correctness gates passed. PyTorch CPU matched the CPU
reference exactly. MPS maximum absolute error was
`8.106231689453125e-06`. ONNX Runtime CPU maximum absolute error was
`1.1920928955078125e-05`. Every backend preserved Top-1 index 574.

Per-process statistics were:

| Backend | Round | Mean | P50 | P90 | P99 | CV |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch CPU | 1 | 48.702 ms | 48.690 ms | 49.051 ms | 49.520 ms | 0.59% |
| PyTorch CPU | 2 | 48.769 ms | 48.616 ms | 49.374 ms | 51.664 ms | 1.33% |
| PyTorch CPU | 3 | 49.054 ms | 48.700 ms | 49.920 ms | 52.670 ms | 4.06% |
| ONNX Runtime CPU | 1 | 1.045 ms | 1.036 ms | 1.162 ms | 1.197 ms | 6.33% |
| ONNX Runtime CPU | 2 | 1.039 ms | 1.037 ms | 1.109 ms | 1.182 ms | 5.65% |
| ONNX Runtime CPU | 3 | 1.077 ms | 1.067 ms | 1.159 ms | 1.219 ms | 6.32% |
| PyTorch MPS | 1 | 3.465 ms | 3.438 ms | 3.732 ms | 4.261 ms | 7.06% |
| PyTorch MPS | 2 | 2.741 ms | 2.726 ms | 2.804 ms | 2.990 ms | 2.08% |
| PyTorch MPS | 3 | 2.758 ms | 2.743 ms | 2.825 ms | 3.028 ms | 2.29% |

PyTorch CPU round means had a 0.72% range relative to their mean. ONNX
Runtime CPU had a 3.65% range. MPS had a 24.22% range and therefore failed
the original cross-round stability criterion.

### MPS stability investigation

Several campaigns were retained locally while investigating the MPS
variation:

1. The initial battery campaign produced MPS round means of approximately
   `3.128`, `3.116`, and `3.296` ms.
2. An AC campaign performed while the battery was still charging produced
   `3.042`, `3.843`, and `4.207` ms. Raw samples in the latter rounds showed
   strong within-run upward trends, so the campaign was excluded.
3. With the battery charged and no macOS thermal/performance warning, three
   consecutive diagnostic processes produced `3.388`, `2.765`, and `2.770`
   ms. The last two differed by only about 0.18%.
4. An explicit preconditioning process followed by the formal v2 campaign
   did not eliminate the first-MPS-process difference: formal MPS means were
   `3.465`, `2.741`, and `2.758` ms.
5. Increasing in-process warm-up from 50 to 1,000 iterations produced two
   further means of `3.506` and `3.090` ms, a 12.59% cross-process
   difference.

These tests yielded two observations: continuous MPS execution did not
consistently increase latency, and a larger fixed in-process warm-up count
did not eliminate cross-process mean-latency differences. MPS/Metal device
or runtime state outside the individual process remains a working hypothesis.
Temperature, device frequency, and system scheduling have not been sufficiently
isolated to establish the root cause.

The result is therefore reported as an observed controlled-AC mean-latency
envelope of `2.74-3.51` ms. No run was removed merely because it was slower,
and the fastest run was not promoted as the single representative result.

### Result organization

- `artifacts/results/day5/formal/ac_v2/`: nine formal three-backend JSON files
- `artifacts/results/day5/diagnostics/mps_state/`: six selected MPS state
  diagnostics
- `artifacts/results/day5/local_archive/`: early exploratory files retained
  locally and excluded from Git

### Limitations and follow-up

- Runtime-default thread settings are recorded but not equalized.
- The derived serial service rate is not concurrent throughput.
- The fixed synthetic input and prepared-input scope exclude preprocessing and
  transfer overhead.
- Only MobileNetV3-Small, FP32, batch size 1, and one Apple Silicon machine
  were evaluated.
- A future benchmark revision can add convergence-based warm-up, backend
  profiling, thread sweeps, additional models/batch sizes, and end-to-end
  latency as separate experiments. These are not folded into the Day 5
  baseline.

---

## Day 6: CPU Intra-op Thread Sweep

### Objective

Measure the relationship between requested CPU intra-op parallelism,
prepared-input latency, process CPU consumption, and cross-process
repeatability. Keep inter-op configuration at each runtime's default so that
the experiment changes one explicit thread-control dimension.

MPS is retained as a host-setting sensitivity control. The swept value is a
PyTorch CPU intra-op setting and is not interpreted as an MPS worker-thread
count.

### Formal AC v1 campaign

The retained campaign used:

- Git commit `1bb0f0fb50acfb776bd4cf43812124a277248f62`
- source SHA-256
  `50c19fab3a646a8e0bca6a9c4e7b031aaa877d884105546532e91c7ad9fcde4c`
- Apple M5 Pro with 48 GB unified memory
- Python 3.11.16 and PyTorch 2.14.0
- AC power with the battery at 100%
- 50 warm-up iterations and 300 measured iterations
- three independent processes per backend/configuration
- no inherited thread-count environment variables
- `PYTORCH_ENABLE_MPS_FALLBACK=0`

The complete 45-record matrix passed source, environment, threading,
correctness, sample-count, statistics, and cross-record fingerprint
validation. The project test suite reported `87 passed`.

### CPU results

| Backend | Intra-op | Representative mean | Cross-process range | Process CPU core equivalents |
| --- | ---: | ---: | ---: | ---: |
| PyTorch CPU | default | 47.150 ms | 8.56% | 1.868 |
| PyTorch CPU | 1 | 8.185 ms | 1.19% | 1.001 |
| PyTorch CPU | 2 | 22.182 ms | 6.90% | 1.071 |
| PyTorch CPU | 4 | 29.947 ms | 4.31% | 1.315 |
| PyTorch CPU | 6 | 48.794 ms | 6.27% | 1.879 |
| ONNX Runtime CPU | default | 1.051 ms | 0.53% | 9.070 |
| ONNX Runtime CPU | 1 | 2.615 ms | 2.67% | 0.999 |
| ONNX Runtime CPU | 2 | 1.653 ms | 0.50% | 1.995 |
| ONNX Runtime CPU | 4 | 1.199 ms | 1.22% | 3.983 |
| ONNX Runtime CPU | 6 | 1.050 ms | 0.09% | 5.953 |

PyTorch CPU was fastest with one intra-op thread and became slower as the
requested thread count increased. The repeated observation is valid for this
model and environment, but the experiment does not identify its
operator-level cause.

ONNX Runtime CPU scaled through six explicit threads. Runtime-default and
explicit six-thread latency were effectively equal, while explicit
six-thread execution used a lower process CPU-time ratio.

`average_process_cpu_core_equivalents` is accumulated process CPU time divided
by sampling-block wall time. It estimates average process CPU consumption and
is not an active-thread count.

### Matched intra-op comparison

At requested intra-op counts of 1, 2, 4, and 6, the PyTorch CPU over ONNX
Runtime CPU latency ratios were `3.13x`, `13.42x`, `24.97x`, and `46.48x`.

Matching the requested intra-op count removes one configuration difference.
It does not equalize inter-op behavior, graph optimization, operator kernels,
memory planning, or internal runtime scheduling.

### MPS sensitivity control

MPS representative means ranged from `2.986 ms` to `3.424 ms`. Individual
configuration groups had cross-process ranges from `9.24%` to `22.47%`.
Because the within-configuration process variation overlaps the
between-configuration difference, no causal MPS thread-setting conclusion is
drawn.

### Result organization

- `artifacts/results/day6/formal/ac_v1/raw/`: 45 immutable process records
- `artifacts/results/day6/formal/ac_v1/summary.json`: validated hierarchical
  summary
- `artifacts/results/day6/formal/ac_v1/summary.csv`: 15-row analysis table

### Limitations and follow-up

- Inter-op policies remain runtime defaults.
- The process CPU-time ratio is not an active-thread measurement.
- The PyTorch CPU reverse-scaling cause requires operator-level profiling.
- MPS cross-process state remains uncontrolled.
- Convergence-based warm-up has not yet been implemented.
- Results still cover one model, input, precision, batch size, and machine.

---

## Day 7: CPU Graph Optimization and Operator Profiling

### Objective and experiment design

Measure the relationship between ORT graph-optimization level and CPU
threading, and locate the operator costs behind Day6's PyTorch reverse
scaling. Three formal campaigns use the same implementation and workload.
Together they provide nine latency processes per ORT graph/thread
configuration and three captures per profiling configuration.

Each campaign contains:

- 24 latency processes: four graph levels × two intra-op settings × three
  rounds, with configuration order rotated across rounds. Each process has
  50 warm-up calls and 300 measured calls.
- 10 profiles: PyTorch CPU at t1/t6, plus four ORT levels at t1/t6. Each
  capture has ten external warm-up calls and ten active calls.

Formal latency covers the prepared-input backend call and synchronization.
Loading, export, session initialization, input preparation, correctness
checks, and serialization remain outside the measured interval. Formal
sessions do not enable profiling or serialize optimized models.

PyTorch profiling records CPU activities, shapes, and memory with one wait
step, two profiler warm-up steps, and ten active steps. ORT records 21
model_run events: one correctness call, ten warm-up calls, and ten active
calls. Its summary selects the final ten; profiling ends before the final
correctness check. Each ORT t1 session saves its optimized graph.

### Environment and source identity

| Campaign | Date | Process-start window | Power snapshot |
|---|---|---|---|
| ac_v1 | 2026-10-30 | 00:32:09–00:34:21 | AC, 100%, charged |
| ac_v2 | 2026-10-30 | 11:51:53–11:54:01 | AC, 100%, charged |
| ac_v3 | 2026-11-02 | 10:55:21–10:57:13 | AC, 100%, finishing charge |

- Producing commit: `22753d099c451e15015190443e1ba4c859b30925`.
- Benchmark source SHA-256:
  `2a23b96ac41792123b407a6600716ae801d88c0a18b5048efa694222047b3d8d`.
- Profiling source SHA-256:
  `f8a4810185df5df435747480d57acb284d644413e8ba7f805559dc6995b2a3ab`.
- Apple M5 Pro, 48 GB unified memory, arm64, macOS 26.5.1, Python 3.11.16.
- PyTorch 2.14.0, torchvision 0.29.0, ONNX 1.22.0, ORT 1.30.0, NumPy 2.4.6.
- Clean source and all five external thread-count variables unset in every record.
- Pretrained MobileNetV3 Small, FP32, input `[1, 3, 224, 224]`, model seed 2026,
  input seed 2027, and request concurrency 1.
- PyTorch inter-op default 18; ORT CPUExecutionProvider, ORT_SEQUENTIAL,
  and inter-op default 0.

Model and input hashes match across all processes. The ONNX hash applies to
ORT records; PyTorch profiles do not consume ONNX.

| Artifact | SHA-256 |
|---|---|
| Model state | `eb1028c72b33c4a15ccc189cef35daae558385dfae44ab31374e84788eaad6ad` |
| Input | `db91944d34fdfa7e2c95ed5bb6c29c3fdd77f834a28e3a2d171e4393306d6d35` |
| Original ONNX | `825e083202937eb6e2cdb654de80475fd751b046960e25efbdc83c7902089cb4` |

Each source hash includes src plus its own entry script, so benchmark and
profiling hashes differ. Each producer's hash is identical across batches.

### Aggregation method

Each process first produces a mean from its 300 latency samples. For each
level/thread configuration, this report takes the median of all nine
process means, with equal weight per process. It also reports the minimum,
maximum, and `(max - min) / median * 100` process range. Relative changes
and thread speedups use these aggregate representative values.

The three campaign summaries retain their original three-process contract.
Their process means are combined for the table below; samples are not
pooled into one distribution, and campaign medians are not used as the
inputs to a second median. This preserves both process-level repetition
and the batch structure needed to inspect state variation.

### Configuration results

| Level | Intra-op | Median of nine process means | Process-mean range | Range / median | Change vs same-thread disable |
|---|---:|---:|---:|---:|---:|
| disable | 1 | 2.652 ms | 2.626–2.694 ms | 2.56% | +0.00% |
| basic | 1 | 2.641 ms | 2.631–2.694 ms | 2.38% | -0.41% |
| extended | 1 | 2.624 ms | 2.597–2.645 ms | 1.83% | -1.07% |
| all | 1 | 2.566 ms | 2.549–2.581 ms | 1.23% | -3.27% |
| disable | 6 | 0.937 ms | 0.927–1.187 ms | 27.67% | +0.00% |
| basic | 6 | 0.939 ms | 0.927–0.965 ms | 4.06% | +0.28% |
| extended | 6 | 0.902 ms | 0.898–0.912 ms | 1.52% | -3.66% |
| all | 6 | 1.043 ms | 1.035–1.047 ms | 1.17% | +11.32% |

The resulting configuration pattern is:

- At t1, all has the lowest representative latency, 2.566 ms, improving on
  disable by 3.27%. At t6, extended is lowest at 0.902 ms, a 3.66% improvement.
- All/t6 is 15.55% slower than extended/t6. Their nine-process ranges remain
  separate. The within-campaign gaps are 15.86%, 15.71%, and 14.24%, so the
  effect is consistently present while its magnitude varies with run state.
- Basic is close to disable: -0.41% at t1 and +0.28% at t6 in the aggregate.
  Its per-campaign differences change direction and the graph bytes are
  identical, giving no evidence of a useful graph transformation here.
- Extended/t1 gives a smaller 1.07% aggregate improvement. Its per-campaign
  improvements span 0.21%–2.21%; it should not be reported as a fixed 2% gain.
- Increasing intra-op from 1 to 6 improves latency at every graph level.
  Aggregate speedups are 2.83x, 2.81x, 2.91x, and 2.46x for disable, basic,
  extended, and all respectively.

These measurements describe MobileNetV3 Small on this machine at FP32,
batch size 1, and request concurrency 1. Graph level and thread count
must be selected together; node reduction alone does not predict latency.

### Process variability

Campaign medians remain visible as a check on temporal variation:

| Level | Intra-op | ac_v1 | ac_v2 | ac_v3 |
|---|---:|---:|---:|---:|
| disable | 1 | 2.672 ms | 2.635 ms | 2.652 ms |
| basic | 1 | 2.653 ms | 2.657 ms | 2.639 ms |
| extended | 1 | 2.613 ms | 2.630 ms | 2.615 ms |
| all | 1 | 2.566 ms | 2.560 ms | 2.578 ms |
| disable | 6 | 0.940 ms | 0.934 ms | 0.936 ms |
| basic | 6 | 0.939 ms | 0.941 ms | 0.939 ms |
| extended | 6 | 0.901 ms | 0.902 ms | 0.911 ms |
| all | 6 | 1.044 ms | 1.044 ms | 1.041 ms |

Extended/t6's campaign median rises from about 0.901–0.902 ms to 0.911 ms
in ac_v3. Its within-campaign process ranges remain below 0.46%, showing
that a tight single-campaign range does not capture all variation between
execution periods. Temperature, frequency, and scheduling were not isolated.

The ac_v1 disable/t6 round-1 mean is 1.186634 ms, with CV 68.36%, P99
2.681055 ms, and a 13.541375 ms maximum. Fifteen samples exceed twice its
P50. Consecutive 50-sample block means are 1.694, 1.479, 1.048, 0.952,
0.981, and 0.966 ms. This transient slowdown occurs once among the nine
processes and remains included in the aggregate and its 27.67% process range.

Ac_v3 basic/t6 round 3 also has a tail: mean 0.965051 ms, P50 0.939813 ms,
P99 1.323871 ms, and CV 8.42%. Its campaign process range is 4.06%.
The ac_v1 disable/t1 rounds 2/3 have CVs of 6.63% and 5.59%.

Independent repetitions and median aggregation limit the influence of a
single slow process; retaining the ranges exposes the remaining variability.
Fixed warm-up does not establish convergence. These observations motivate
convergence-based warm-up and targeted diagnostics rather than removing
slow samples.

### Graph structure

The original graph and each level's optimized graph have matching hashes
across all three campaigns.

| Graph | Nodes | Initializers | Serialized bytes | Main observations |
|---|---:|---:|---:|---|
| Original export | 122 | 110 | 10,476,133 | 52 Conv, 19 HardSwish |
| disable | 141 | 110 | 10,180,615 | HardSwish expanded; byte-identical to basic |
| basic | 141 | 110 | 10,180,615 | Same operator distribution and hash as disable |
| extended | 118 | 110 | 10,180,559 | 29 Conv and 23 com.microsoft::FusedConv |
| all | 114 | 110 | 10,180,703 | 22 Conv, 29 FusedConv, one NhwcFusedConv, two Transpose |

The 19 additional nodes under disable/basic match expansion of HardSwish
into HardSigmoid and Mul, as defined by the
[ONNX function body](https://onnx.ai/onnx/operators/onnx__HardSwish.html#hardswish-14).
ORT's [transformation pipeline](https://github.com/microsoft/onnxruntime/blob/f2c39fe/onnxruntime/core/session/inference_session.cc#L1409)
includes function inlining and required preparation alongside optional
optimizations. Disabling optional optimization therefore does not preserve
the original serialized graph unchanged.

Compared with extended, all removes six Add nodes, changes the convolution
operator mix, and adds layout conversion. Fewer nodes do not ensure lower
latency. The optimized models are local diagnostic artifacts; ORT documents
[hardware compatibility requirements](https://onnxruntime.ai/docs/performance/model-optimizations/graph-optimizations.html)
for layout-optimized models.

### PyTorch operator evidence

Each thread setting has three independent captures of ten active calls.
The table reports medians of capture-level totals and shares; each share
is calculated before aggregation. Shape groups of the same operator are
combined by self CPU time, excluding child time.

| Intra-op | Captures | slow_conv2d_forward calls per capture | Median total self CPU time | Median slow_conv self time | Median slow_conv share | Share range |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 3 | 24,090 | 149.584 ms | 75.756 ms | 50.72% | 50.64%–51.87% |
| 6 | 3 | 24,090 | 599.632 ms | 481.919 ms | 80.39% | 80.07%–80.57% |

At t6, the single-channel 5×5 convolution groups with 7×7 and 14×14 inputs
together account for 73.24%–73.88% of total self CPU time. The original
graph contains 11 grouped Conv nodes; narrow, slice, and thnn_conv2d calls
also recur in the profiles. The repeated hotspot is the small-channel
convolution path and its surrounding dispatch work.

Within each campaign, the slow_conv2d_forward self-time increase accounts
for 89.65%–90.29% of the total self-time increase from t1 to t6. This
localizes the instrumented increase. Shape and memory recording introduce
[profiler overhead](https://docs.pytorch.org/docs/2.14/profiler.html#torch.profiler.profile);
a lower-overhead kernel experiment is needed to separate computation,
dispatch, and synchronization costs behind the Day6 latency behavior.

### ORT operator and memory evidence

At disable, basic, extended, and all, every active inference has 141, 141,
118, and 114 Node events respectively, entirely on CPUExecutionProvider.
The graph structure and operator counts are unchanged across repetitions.

All/t6 has 20 Transpose calls per ten-call capture. Their total duration
ranges from 1,030 to 1,250 us, and their Node-time share from 7.59% to 9.35%
(median 8.28%). Layout conversion and changed convolution kernels remain
concrete candidates for targeted diagnosis. Profile time is not directly
subtracted from the formal latency gap.

Runtime unattributed duration is model_run duration minus Node duration.
Across the ten-call captures it spans 1,621–1,627 us for extended/t6 and
1,595–1,671 us for all/t6. It describes time outside Node events without
identifying one runtime subsystem.

ORT memory-in-use peaks are unchanged: 10,822,240 bytes at t1 for
disable/basic/extended and 11,808,864 for all; 12,853,856 bytes at t6 for
disable/basic/extended and 12,857,440 for all. These event-level runtime
observations differ from PyTorch's signed allocator deltas and process RSS.

### Import isolation and telemetry

Earlier attempts aborted during ORT's native telemetry shutdown after
profiling files had been written. Commit 22753d0 defers ORT imports to the
ORT backend and profiling branch. TYPE_CHECKING handles type-only imports,
and version metadata is read without loading the native runtime.

The operator completed all three retained campaigns with ORT_DISABLE_TELEMETRY
unset; ac_v2/ac_v3 notes also record that policy. The JSON has no dedicated
telemetry-environment or exit-code field. Import isolation prevents
PyTorch-only processes from initializing ORT; it does not modify ORT's
native telemetry implementation. No telemetry on/off performance comparison
was performed. Earlier failed outputs were removed before these campaigns.

### Validation and archives

- Three campaigns contain 72 latency records (21,600 samples),
  30 profiling records, and 15 summaries.
- Before/after correctness passes for every record: output `[1, 1000]`,
  Top-1 574, and rtol=atol=1e-4.
- Source, workload, runtime, and hardware fingerprints match across
  campaigns. Formal raw records match their staging copies byte for byte.
- All 15 summaries reproduce byte for byte, and all 30 trace hashes match.
  The 24 ORT summaries reproduce from traces; the six PyTorch captures
  each contain ten active steps and matching normalized operator call counts.
- The original model and 12 optimized models pass ONNX Checker and match
  recorded graph statistics and hashes. Cached model-state and seeded-input
  hashes were also reconstructed.
- Implementation verification recorded 143 passed.

Formal archives:

- [ac_v1](../artifacts/results/day7/formal/ac_v1/)
- [ac_v2](../artifacts/results/day7/formal/ac_v2/)
- [ac_v3](../artifacts/results/day7/formal/ac_v3/)

Each campaign retains its own three-process summaries. The aggregate tables
above are derived from the individual process records across all campaigns.

Traces and optimized models remain under local_archive/ac_v1, ac_v2, and
ac_v3. The ac_v1 files were moved into a batch directory without changing
their hashes. Paths below are relative to artifacts/results/day7:

| ac_v1 path recorded in JSON | Current local path |
|---|---|
| `local_archive/traces/` | `local_archive/ac_v1/traces/` |
| `local_archive/optimized_models/` | `local_archive/ac_v1/optimized_models/` |

Raw commit, hashes, timestamps, samples, and summaries are preserved.

### Follow-up

Day7 provides a repeated-measurement baseline for graph/thread selection
and identifies convolution and layout work for further diagnosis. Shared
script logic extraction is the next bounded maintenance task, followed
by targeted profiling and a second-model extension. Convergence-based
warm-up, ResNet, ViT, Transformer, FP16, INT8, CoreML, CUDA, concurrent
serving, and end-to-end pipelines remain future work.
