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