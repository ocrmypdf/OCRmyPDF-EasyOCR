# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

import ocrmypdf
import pikepdf
import pytest

import ocrmypdf_easyocr


def test_easyocr(resources, outpdf):
    ocrmypdf.ocr(resources / "jbig2.pdf", outpdf, pdf_renderer="sandwich")
    assert outpdf.exists()

    with pikepdf.open(outpdf) as pdf:
        assert "EasyOCR" in str(pdf.docinfo["/Creator"])
