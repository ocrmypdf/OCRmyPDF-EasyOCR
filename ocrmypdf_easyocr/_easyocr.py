# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""Thin wrapper around the EasyOCR library.

This module isolates everything EasyOCR-specific: the language-code mapping, the
lazily-constructed shared :class:`easyocr.Reader`, and the call that performs
recognition and tidies the results. It knows nothing about OCRmyPDF.
"""

from __future__ import annotations

import contextlib
import sys
import threading
from typing import NamedTuple

import easyocr

# OCRmyPDF uses 3-letter ISO 639-3 language codes; EasyOCR uses its own (mostly
# 2-letter) codes. This maps the former to the latter.
ISO_639_3_2: dict[str, str] = {
    "afr": "af",
    "alb": "sq",
    "ara": "ar",
    "aze": "az",
    "bel": "be",
    "ben": "bn",
    "bos": "bs",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "che": "che",
    "chi_sim": "ch_sim",
    "chi_tra": "ch_tra",
    "cym": "cy",
    "cze": "cs",
    "dan": "da",
    "deu": "de",
    "dut": "nl",
    "eng": "en",
    "est": "et",
    "esp": "es",
    "fas": "fa",
    "fra": "fr",
    "gle": "ga",
    "hin": "hi",
    "hrv": "hr",
    "hun": "hu",
    "ice": "is",
    "ind": "id",
    "isl": "is",
    "ita": "it",
    "jpn": "ja",
    "kor": "ko",
    "kur": "ku",
    "lat": "la",
    "lav": "lv",
    "lit": "lt",
    "may": "ms",
    "mlt": "mt",
    "mon": "mn",
    "msa": "ms",
    "nep": "ne",
    "nld": "nl",
    "nor": "no",
    "oci": "oc",
    "per": "fa",
    "pol": "pl",
    "por": "pt",
    "rum": "ro",
    "ron": "ro",
    "rus": "ru",
    "slo": "sk",
    "slk": "sk",
    "slv": "sl",
    "spa": "es",
    "swa": "sw",
    "swe": "sv",
    "tam": "ta",
    "tha": "th",
    "tgl": "tl",
    "tur": "tr",
    "ukr": "uk",
    "urd": "ur",
    "vie": "vi",
}

# Defaults for the EasyOCR options. OCRmyPDF only applies argparse defaults on the
# command-line path; when driven through the Python API these options may be
# absent, so callers read them defensively with these values.
DEFAULT_GPU = True
DEFAULT_BATCH_SIZE = 4

#: Languages supported by this wrapper, as OCRmyPDF-style ISO 639-3 codes.
SUPPORTED_LANGUAGES = frozenset(ISO_639_3_2.keys())


def version() -> str:
    """Return the version of the installed EasyOCR library."""
    return easyocr.__version__


class EasyOCRResult(NamedTuple):
    """Result of OCR with EasyOCR."""

    quad: list
    text: str
    confidence: float


def tidy_easyocr_result(raw_result) -> EasyOCRResult:
    """Convert a raw EasyOCR result to a more convenient format."""
    return EasyOCRResult(
        quad=[el for sublist in raw_result[0] for el in sublist],  # flatten list
        text=raw_result[1],
        confidence=raw_result[2],
    )


# EasyOCR's Reader is expensive to construct (it loads models, possibly onto a
# GPU) and EasyOCR/PyTorch inference is not thread-safe. OCRmyPDF parallelizes
# OCR across pages, so we keep a single lazily-constructed Reader and serialize
# access to it with a lock. This keeps exactly one inference running at a time,
# matching EasyOCR's effectively single-threaded design, without spawning our
# own worker processes (which also keeps us friendly to Celery/paperless-ngx).
_reader: easyocr.Reader | None = None
_reader_lock = threading.Lock()


def _get_reader(easyocr_languages: list[str], gpu: bool) -> easyocr.Reader:
    """Return the shared EasyOCR Reader, constructing it on first use.

    Must be called while holding ``_reader_lock``.
    """
    global _reader
    if _reader is None:
        # Redirect stdout to stderr during Reader initialization to be compliant
        # with ocrmypdf; otherwise the model-loading progress bar interferes with
        # PDFs that are piped to stdout.
        with contextlib.redirect_stdout(sys.stderr):
            _reader = easyocr.Reader(easyocr_languages, gpu=gpu)
    return _reader


def readtext(
    image,
    *,
    languages,
    gpu: bool = DEFAULT_GPU,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> list[EasyOCRResult]:
    """Run EasyOCR on an image and return tidied results.

    Args:
        image: An image array (or path) accepted by ``easyocr.Reader.readtext``.
        languages: OCRmyPDF-style ISO 639-3 language codes; mapped to EasyOCR's
            codes here.
        gpu: Whether to use the GPU.
        batch_size: EasyOCR batch size.
    """
    easyocr_languages = [ISO_639_3_2[lang] for lang in languages]
    with _reader_lock:
        reader = _get_reader(easyocr_languages, gpu)
        raw_results = reader.readtext(image, batch_size=batch_size)
    return [tidy_easyocr_result(r) for r in raw_results]
