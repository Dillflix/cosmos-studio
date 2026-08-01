import pytest

from app.prompting import parse_size


@pytest.mark.parametrize(
    "size",
    [
        "512x512",
        "640x640",
        "768x768",
        "768x432",
        "432x768",
        "1280x720",
        "832x480",
        "2048x256",
    ],
)
def test_parse_size_accepts_presets_and_custom_multiples_of_16(size):
    width, height = parse_size(size)
    assert width % 16 == 0
    assert height % 16 == 0


@pytest.mark.parametrize("size", ["640x360", "513x512", "255x512", "2049x512"])
def test_parse_size_rejects_unsupported_custom_dimensions(size):
    with pytest.raises(ValueError):
        parse_size(size)
