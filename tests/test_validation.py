import pytest
import torch

from inference_bench import validation


def test_compare_classification_outputs_reports_error_and_top1() -> None:
    """Verify error metrics and Top-1 agreement.

    The expected values are derived manually so that the test remains
    independent of the implementation under test.
    """

    # Arrange: construct deterministic classificaiton outputs whose
    # expected shapes, Top-1 indices, and error metrics are known.
    reference = torch.tensor(
        [[0.10000, 0.30000, 0.60000]],
        dtype=torch.float32,
    )
    candidate = torch.tensor(
        [[0.10001, 0.30002, 0.59997]],
        dtype=torch.float32,
    )

    # Act: compare the candidate output against the reference output.
    result = validation.compare_classification_outputs(
        reference,
        candidate,
        rtol=1e-4,
        atol=1e-4,
    )

    # Assert: verify the output metadata and Top-1 agreement.
    assert result.reference_shape == (1, 3)
    assert result.candidate_shape == (1, 3)
    assert result.reference_top1_index == 2
    assert result.candidate_top1_index == 2
    assert result.top1_match is True

    # The element-wise absolute erroes are approximately
    # [1e-5, 2e-5, 3e-5], so the expected maximun and mean errrors
    # are 3e-5 and 2e-5, respectively.
    assert result.max_absolute_error == pytest.approx(3e-5, abs=1e-7)
    assert result.mean_absolute_error == pytest.approx(2e-5, abs=1e-7)

    # All differences must remain within the configured tolenrance.
    assert result. allclose is True


def test_compare_classification_outputs_rejects_shape_mismatch() -> None:
    # Arrange: create outputs with incompatible class dimensions.
    reference = torch.zeros((1, 3), dtype=torch.float32)
    candidate = torch.zeros((1, 4), dtype=torch.float32)

    # Act and assert: mismatched output shapes must be rejected before
    # element-wise numerical comparison is attempted.
    with pytest.raises(ValueError, match="shape"):
        validation.compare_classification_outputs(
            reference,
            candidate,
            rtol=1e-4,
            atol=1e-4,
        )


@pytest.mark.parametrize("bad_side", ["reference", "candidate"])
def test_compare_classification_outputs_rejects_non_finite_values(
    bad_side: str,
) -> None:
    # Start with identical, finite outputs so that the injected
    # non-finite value is the only invalid condition.
    reference = torch.tensor(
        [[0.1, 0.2, 0.7]],
        dtype=torch.float32,
    )
    candidate = reference.clone()

    # Inject a non-finite value into each output in a separate test case.
    if bad_side == "reference":
        reference[0, 1] = float("nan")
    else:
        candidate[0, 1] = float("inf")

    # The comparison must reject the values and identify which output
    # contains the invalid element.
    with pytest.raises(ValueError, match=bad_side):
        validation.compare_classification_outputs(
            reference,
            candidate,
            rtol=1e-4,
            atol=1e-4,
        )