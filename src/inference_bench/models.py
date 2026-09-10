from torch import nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


def load_model(
    name: str,
    # The asterisk (*) makes all following parameters keyword-only.
    *,
    pretrained: bool = True,
) -> nn.Module:
    """Create a supported TorchVision model configured for inference.

    Args:
        name: Name of the model to create.
        pretrained: Whether to load the recommended pretrained weights.

    Returns:
        The requested model in evaluation mode.

    Raises:
        ValueError: If the requested model is unsupported.
    """

    normalized_name = name.strip().lower()

    if normalized_name != "mobilenet_v3_small":
        raise ValueError(f"Unsupported model: {name}")

    # Select weights according to the pretrained argument:
    # True: use the pretrained weights recommended by Torchvision.
    # False: pass None and use randomly initialized parameters.
    weights = (
        MobileNet_V3_Small_Weights.DEFAULT
        if pretrained
        else None
    )

    model = mobilenet_v3_small(weights=weights)

    # Switch the model to evaluation mode.
    # This changes the behavior of modules such as dropout and
    # batch normalization, but does not disable gradient tracking.
    model.eval()

    # Return the constructed model.
    # It remains on the CPU and uses FP32 precision by default.
    return model