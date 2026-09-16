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