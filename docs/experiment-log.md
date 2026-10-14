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

These tests falsified two simple explanations: continuous MPS execution did
not inevitably trigger thermal throttling, and a larger fixed in-process
warm-up count did not remove the process-to-process state difference. The
available evidence localizes the problem to an external MPS/Metal device or
runtime state not controlled by the current harness. It does not identify a
specific internal cache, frequency, or driver mechanism.

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