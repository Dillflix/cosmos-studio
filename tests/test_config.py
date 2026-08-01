from app.config import Settings, _bool, _int, _string


def test_dotenv_style_quotes_are_removed(monkeypatch):
    monkeypatch.setenv("COSMOS_STUDIO_EXPECTED_AUX_NAME", '"Radeon 8060S"')
    monkeypatch.setenv("COSMOS_STUDIO_API_KEY", "'private-key'")
    monkeypatch.setenv("PROMPT_UPSAMPLER_ENDPOINT_URL", '"https://example.test/v1/"')

    settings = Settings.from_env()

    assert settings.expected_aux_name == "Radeon 8060S"
    assert settings.api_key == "private-key"
    assert settings.prompt_endpoint == "https://example.test/v1/"


def test_typed_values_accept_dotenv_style_quotes(monkeypatch):
    monkeypatch.setenv("QUOTED_STRING", '"value with spaces"')
    monkeypatch.setenv("QUOTED_BOOL", '"true"')
    monkeypatch.setenv("QUOTED_INT", "'42'")

    assert _string("QUOTED_STRING") == "value with spaces"
    assert _bool("QUOTED_BOOL", False) is True
    assert _int("QUOTED_INT", 0) == 42


def test_video_model_defaults_to_full_super_nf4_checkpoint(monkeypatch):
    monkeypatch.delenv("COSMOS_STUDIO_COSMOS_VIDEO_MODEL", raising=False)

    assert (
        Settings.from_env().cosmos_video_model
        == "SanDiegoDude/Cosmos3-Super-nf4"
    )
