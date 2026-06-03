# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

from __future__ import annotations

import shutil
import subprocess

import easyocr
import ocrmypdf
import pikepdf
import pytest


def test_easyocr_engine_is_used(resources, outpdf, monkeypatch):
    """EasyOCR must actually perform the OCR -- not a silent Tesseract fallback.

    We spy on easyocr.Reader.readtext (still calling through) and require it to
    be invoked, and confirm the EasyOCR creator tag lands in the output.
    """
    calls = 0
    original_readtext = easyocr.Reader.readtext

    def spy_readtext(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_readtext(self, *args, **kwargs)

    monkeypatch.setattr(easyocr.Reader, "readtext", spy_readtext)

    ocrmypdf.ocr(
        resources / "jbig2.pdf",
        outpdf,
        progress_bar=False,
    )

    assert calls > 0, "EasyOCR.readtext was never called -- engine did not run"
    assert outpdf.exists()
    with pikepdf.open(outpdf) as pdf:
        assert "EasyOCR" in str(pdf.docinfo["/Creator"])


@pytest.mark.skipif(
    shutil.which("pdftotext") is None, reason="poppler's pdftotext not installed"
)
def test_ocr_text_is_readable_with_pdftotext(text_image, outpdf):
    """End-to-end: OCRmyPDF+EasyOCR output yields selectable text.

    Render known words to an image, OCR it through the plugin, then extract text
    with poppler's pdftotext and confirm the words come back.
    """
    image_path, words = text_image

    ocrmypdf.ocr(
        image_path,
        outpdf,
        image_dpi=300,
        progress_bar=False,
    )
    assert outpdf.exists()

    extracted = subprocess.run(
        ["pdftotext", str(outpdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.lower()

    # Allow minor OCR noise: require most of the distinctive words to survive.
    found = [w for w in words if w.lower() in extracted]
    assert len(found) >= len(words) - 1, (
        f"pdftotext recovered {found} from {words}; full text: {extracted!r}"
    )
