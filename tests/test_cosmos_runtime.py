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


def test_denoise_progress_maps_steps_into_runtime_interval():
    assert CosmosRuntime._denoise_progress(0, 35) > 0.15
    assert CosmosRuntime._denoise_progress(34, 35) == pytest.approx(0.83)


def test_denoise_progress_is_bounded():
    assert CosmosRuntime._denoise_progress(-2, 35) == 0.15
    assert CosmosRuntime._denoise_progress(99, 35) == pytest.approx(0.83)
