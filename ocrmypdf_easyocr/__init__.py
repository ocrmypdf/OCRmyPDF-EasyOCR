# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""EasyOCR plugin for OCRmyPDF.

OCRmyPDF registers this package (named by the ``ocrmypdf`` entry point) as a
plugin and scans it for hook implementations, so the hooks are re-exported here.
The implementation lives in :mod:`ocrmypdf_easyocr._plugin`, with the EasyOCR
library wrapper in :mod:`ocrmypdf_easyocr._easyocr`.
"""

from __future__ import annotations

from ocrmypdf_easyocr._plugin import add_options, get_ocr_engine

__all__ = ["add_options", "get_ocr_engine"]
