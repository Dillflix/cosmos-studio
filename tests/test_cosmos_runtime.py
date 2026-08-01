import pytest

from app.runtimes.cosmos import CosmosRuntime


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "cuda:1"])
def test_indexed_cuda_devices_use_diffusers_cuda_strategy(device):
    assert CosmosRuntime._loader_device_map(device) == "cuda"


def test_cpu_device_uses_cpu_strategy():
    assert CosmosRuntime._loader_device_map("cpu") == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unsupported Cosmos device"):
        CosmosRuntime._loader_device_map("xpu:0")
