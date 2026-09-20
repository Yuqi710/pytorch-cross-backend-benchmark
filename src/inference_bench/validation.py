from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ClassificationOutputComparison:
    """Immutable summary of a classification-output comparison."""

    reference_shape: tuple[int, ...]
    candidate_shape: tuple[int, ...]
    reference_top1_index: int
    candidate_top1_index: int
    top1_match: bool
    max_absolute_error: float
    mean_absolute_error: float
    rtol: float
    atol: float
    allclose: bool


def compare_classification_outputs(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> ClassificationOutputComparison:
    """Compare two single-sample classification output tensors.

    The reference tensor normally represents the output from the baseline
    backend, while the candidate tensor represents the output from the
    backend being validated.

    Both tensors are detached from their autograd praphs and transferred to
    CPU before the comparison. The function reports output shapes, Top-1
    agreement, absolute-error statistics, and element-wise tolerance status.

    Note:
        This implementation assumes a single-sample classification output,
        typically shaped ``[1, num_classes]``. The use of ``Tensor.item()``
        for Top-1 extraction is not valid for batches containing more than
        one sample.
    """

    # Detach inference outputs and normalize their device placement so that
    # metric computation is independent of the original execution backend.
    reference_cpu = reference.detach().cpu()
    candidate_cpu = candidate.detach().cpu()

    # Require identical output shapes; broadcasting would invalidate
    # element-wise numerical comparison between inference backends.
    if reference_cpu.shape != candidate_cpu.shape:
        raise ValueError(
            "Output shape mismatch: "
            f"reference={tuple(reference_cpu.shape)}, "
            f"candidate={tuple(candidate_cpu.shape)}"
        )

    # Numerical error metrics are meaningful only when every reference
    # value is finite.
    if not bool(torch.isfinite(reference_cpu).all()):
        raise ValueError(
            "reference contains NaN or infinity"
        )

    # Validate the candidate independently so that failures can be
    # attributed to the correct inference backend.
    if not bool(torch.isfinite(candidate_cpu).all()):
        raise ValueError(
            "candidate contains NaN or Infinity"
        )

    # Compute the element-wise absolute error used for aggregate statistics.
    absolute_error = torch.abs(reference_cpu - candidate_cpu)

    # The last tensor dimension is treated as the classificaiton dimension.
    reference_top1 = int(
        torch.argmax(reference_cpu, dim=-1).item()
    )
    candidate_top1 = int(
        torch.argmax(candidate_cpu, dim=-1).item()
    )

    return ClassificationOutputComparison(
        reference_shape=tuple(reference_cpu.shape),
        candidate_shape=tuple(candidate_cpu.shape),
        reference_top1_index=reference_top1,
        candidate_top1_index=candidate_top1,
        top1_match=reference_top1==candidate_top1,
        max_absolute_error=float(absolute_error.max().item()),
        mean_absolute_error=float(absolute_error.mean().item()),
        rtol=rtol,
        atol=atol,
        allclose=bool(
            torch.allclose(
                reference_cpu,
                candidate_cpu,
                rtol=rtol,
                atol=atol,
            )
        ),
    )