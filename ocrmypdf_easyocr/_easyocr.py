# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""Interface to EasyOCR."""

from __future__ import annotations

from typing import NamedTuple


class EasyOCRResult(NamedTuple):
    """Result of OCR with EasyOCR."""

    quad: list
    text: str
    confidence: float


def tidy_easyocr_result(raw_result) -> EasyOCRResult:
    """Converts EasyOCR results to a more convenient format."""
    return EasyOCRResult(
        quad=[el for sublist in raw_result[0] for el in sublist],  # flatten list
        text=raw_result[1],
        confidence=raw_result[2],
    )
