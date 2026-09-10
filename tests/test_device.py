"""Unit tests for device capabilities reporting and request resolution.

These tests validate configuraition parsing and availability handling.
Actual tensor execution is covered by separate integration smoke tests.
"""

import platform

import pytest
import torch

import inference_bench.device as device


def test_detect_capabilities_reports_current_runtime():
    capabilities = device.detect_capabilities()

    assert capabilities.architecture == platform.machine()
    assert capabilities.python_version == platform.python_version()
    assert capabilities.torch_version == torch.__version__
    assert capabilities.mps_built == torch.backends.mps.is_built()
    assert capabilities.mps_available == torch.backends.mps.is_available()


def test_resolve_device_returns_cpu():
    select = device.resolve_device("cpu")

    assert select.type == "cpu"


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS is unavailable",
)
def test_resolve_device_returns_mps():
    select = device.resolve_device("mps")

    assert select.type == "mps"


def test_resolve_device_rejects_unknown_device():
    # pytest.raises() expects the indented code to raise the specified exception.
    with pytest.raises(ValueError, match="Unsupported device"):
        device.resolve_device("cuda")