# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).parent.resolve()
PROJECT_ROOT = TESTS_ROOT


@pytest.fixture(scope="session")
def resources() -> Path:
    return Path(TESTS_ROOT) / "resources"


@pytest.fixture(scope="function")
def outdir(tmp_path) -> Path:
    return tmp_path


@pytest.fixture(scope="function")
def outpdf(tmp_path) -> Path:
    return tmp_path / "out.pdf"


def _load_font(size: int):
    from PIL import ImageFont

    for candidate in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default(size=size)


@pytest.fixture(scope="function")
def text_image(tmp_path):
    """Render known text to a high-resolution PNG and return (path, words).

    Used to verify end-to-end that EasyOCR recognized text that can be read back
    out of the OCRmyPDF output.
    """
    from PIL import Image, ImageDraw

    words = ["The", "quick", "brown", "fox"]
    text = " ".join(words)
    dpi = 300

    img = Image.new("RGB", (1800, 400), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((60, 130), text, fill="black", font=_load_font(120))

    path = tmp_path / "text.png"
    img.save(path, dpi=(dpi, dpi))
    return path, words
