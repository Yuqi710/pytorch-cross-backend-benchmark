import pytest
import torch

import inference_bench.models as models


def test_load_mobilenet_v3_small_for_inference():
    """Verify that MobileNetV3 Small is ready for inference."""

    # Load MobileNetV3 Small.
    # pretrained=False avoids downloading pretrained weights, keeping
    # the unit test independent of network access.
    model = models.load_model(
        name="mobilenet_v3_small",
        pretrained=False,
    )

    # model.training=False confirms that model.eval() has been called
    # and the model is in evaluation mode rather than training mode.
    assert model.training is False

    # Standard NCHW input shape for ImageNet classification models.
    inputs = torch.zeros((1, 3, 224, 224))

    # Disable gradient tracking and perform a forward pass.
    # inference_mode() reduces memory usage and Autograd overhead during inference.
    with torch.inference_mode():
        outputs = model(inputs)

    # MobileNetV3 Small use an ImageNet classification head by default.
    # For one input image, the model should produce scores for 1000 ImageNet classes.
    assert outputs.shape == (1, 1000)


def test_load_model_rejects_unknown_model():
    """Verify that unsupported model names are rejected."""

    with pytest.raises(ValueError, match="Unsupported model"):
        models.load_model(name="unknown_model", pretrained=False)