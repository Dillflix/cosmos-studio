from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
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


def test_video_loader_disables_only_optional_action_head(monkeypatch):
    torch = MagicMock()
    torch.bfloat16 = "bf16"
    torch.cuda.get_device_name.return_value = "Radeon 8060S Graphics"
    transformer_type = MagicMock()
    transformer = object()
    transformer_type.from_pretrained.return_value = transformer
    pipeline_type = MagicMock()
    pipeline = MagicMock()
    pipeline.scheduler.config = {"name": "scheduler"}
    pipeline_type.from_pretrained.return_value = pipeline

    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    with patch.dict(
        "sys.modules",
        {"diffusers": MagicMock(
            Cosmos3OmniPipeline=pipeline_type,
            Cosmos3OmniTransformer=transformer_type,
        )},
    ):
        settings = replace(
            Settings.from_env(),
            cosmos_video_model="SanDiegoDude/Cosmos3-Super-nf4",
            cosmos_device="cuda:0",
            expected_aux_name="Radeon 8060S",
        )
        runtime = CosmosRuntime(settings)
        runtime._patch_allocator = MagicMock()
        runtime._verify_device = MagicMock()
        runtime._ensure_loaded("cosmos_text_video", MagicMock())

    transformer_type.from_pretrained.assert_called_once_with(
        "SanDiegoDude/Cosmos3-Super-nf4",
        subfolder="transformer",
        torch_dtype="bf16",
        device_map="cuda",
        low_cpu_mem_usage=True,
        action_gen=False,
    )
    assert pipeline_type.from_pretrained.call_args.kwargs["transformer"] is transformer


def test_specialized_image_loader_keeps_checkpoint_managed_transformer(monkeypatch):
    torch = MagicMock()
    torch.bfloat16 = "bf16"
    torch.cuda.get_device_name.return_value = "Radeon 8060S Graphics"
    transformer_type = MagicMock()
    pipeline_type = MagicMock()
    pipeline = MagicMock()
    pipeline.scheduler.config = {"name": "scheduler"}
    pipeline_type.from_pretrained.return_value = pipeline

    monkeypatch.setitem(__import__("sys").modules, "torch", torch)
    with patch.dict(
        "sys.modules",
        {"diffusers": MagicMock(
            Cosmos3OmniPipeline=pipeline_type,
            Cosmos3OmniTransformer=transformer_type,
        )},
    ):
        settings = replace(
            Settings.from_env(),
            cosmos_image_model="/models/Cosmos3-Super-Text2Image-nf4",
            cosmos_device="cuda:0",
            expected_aux_name="Radeon 8060S",
        )
        runtime = CosmosRuntime(settings)
        runtime._patch_allocator = MagicMock()
        runtime._verify_device = MagicMock()
        runtime._ensure_loaded("cosmos_image", MagicMock())

    transformer_type.from_pretrained.assert_not_called()
    assert "transformer" not in pipeline_type.from_pretrained.call_args.kwargs
