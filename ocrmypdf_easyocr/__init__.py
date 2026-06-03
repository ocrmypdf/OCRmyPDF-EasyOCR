# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""EasyOCR plugin for OCRmyPDF."""

from __future__ import annotations

import contextlib
import logging
import os
import sys
import threading
from math import atan2, degrees

import cv2 as cv
import easyocr
from ocrmypdf import BoundingBox, OcrClass, OcrElement, OcrEngine, hookimpl
from ocrmypdf._exec import tesseract
from PIL import Image

from ocrmypdf_easyocr._cv import detect_skew
from ocrmypdf_easyocr._easyocr import tidy_easyocr_result

log = logging.getLogger(__name__)

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
# absent from OcrOptions, so the engine reads them defensively with these values.
DEFAULT_GPU = True
DEFAULT_BATCH_SIZE = 4

# EasyOCR's Reader is expensive to construct (it loads models, possibly onto a
# GPU) and EasyOCR/PyTorch inference is not thread-safe. OCRmyPDF parallelizes
# OCR across pages, so we keep a single lazily-constructed Reader and serialize
# access to it with a lock. This keeps exactly one inference running at a time,
# matching EasyOCR's effectively single-threaded design, without spawning our
# own worker processes (which also keeps us friendly to Celery/paperless-ngx).
_reader: easyocr.Reader | None = None
_reader_lock = threading.Lock()


def _get_reader(options) -> easyocr.Reader:
    """Return the shared EasyOCR Reader, constructing it on first use.

    Must be called while holding ``_reader_lock``.
    """
    global _reader
    if _reader is None:
        languages = [ISO_639_3_2[lang] for lang in options.languages]
        # Redirect stdout to stderr during Reader initialization to be compliant
        # with ocrmypdf; otherwise the model-loading progress bar interferes with
        # PDFs that are piped to stdout.
        with contextlib.redirect_stdout(sys.stderr):
            _reader = easyocr.Reader(languages, gpu=getattr(options, "gpu", DEFAULT_GPU))
    return _reader


@hookimpl
def add_options(parser):
    easyocr_options = parser.add_argument_group("EasyOCR", "EasyOCR options")
    easyocr_options.add_argument(
        "--easyocr-no-gpu", action="store_false", dest="gpu", default=DEFAULT_GPU
    )
    easyocr_options.add_argument(
        "--easyocr-batch-size", type=int, default=DEFAULT_BATCH_SIZE
    )


def _easyocr_to_ocr_tree(input_file, results, page_number) -> OcrElement:
    """Convert tidied EasyOCR results into an OcrElement tree for OCRmyPDF.

    OCRmyPDF 17+ renders the text layer itself from this tree (using its fpdf2
    renderer), so we only describe where text was found. Coordinates are in image
    pixels with a top-left origin, which is what the renderer expects.
    """
    with Image.open(input_file) as im:
        width, height = im.width, im.height
        dpi = im.info.get("dpi", (72.0, 72.0))[0]

    lines: list[OcrElement] = []
    for result in results:
        if not result.text:
            continue
        # quad is flattened [ulx, uly, urx, ury, lrx, lry, llx, lly]
        xs = result.quad[0::2]
        ys = result.quad[1::2]
        bbox = BoundingBox(
            left=min(xs), top=min(ys), right=max(xs), bottom=max(ys)
        )

        # Rotation of the text, in degrees counter-clockwise (per the hOCR
        # convention OCRmyPDF's renderer uses). Image y grows downward, so a
        # counter-clockwise rotation makes the top edge (ul -> ur) rise, i.e. a
        # decreasing y; hence the negated dy.
        ulx, uly, urx, ury = result.quad[0], result.quad[1], result.quad[2], result.quad[3]
        angle = degrees(atan2(-(ury - uly), urx - ulx))
        textangle = angle if abs(angle) >= 0.6 else 0.0  # ignore <0.6 deg noise

        word = OcrElement(ocr_class=OcrClass.WORD, bbox=bbox, text=result.text)
        lines.append(
            OcrElement(
                ocr_class=OcrClass.LINE,
                bbox=bbox,
                textangle=textangle,
                children=[word],
            )
        )

    return OcrElement(
        ocr_class=OcrClass.PAGE,
        bbox=BoundingBox(left=0, top=0, right=width, bottom=height),
        dpi=float(dpi),
        page_number=page_number,
        children=lines,
    )


class EasyOCREngine(OcrEngine):
    """Implements OCR with EasyOCR."""

    @staticmethod
    def version():
        return easyocr.__version__

    @staticmethod
    def creator_tag(options):
        return f"EasyOCR {EasyOCREngine.version()}"

    def __str__(self):
        return f"EasyOCR {EasyOCREngine.version()}"

    @staticmethod
    def languages(options):
        return set(ISO_639_3_2.keys())

    @staticmethod
    def get_orientation(input_file, options):
        return tesseract.get_orientation(
            input_file,
            engine_mode=options.tesseract.oem,
            timeout=options.tesseract.non_ocr_timeout,
        )

    @staticmethod
    def get_deskew(input_file, options) -> float:
        img = cv.imread(os.fspath(input_file))
        angle = detect_skew(img)
        log.debug(f"Detected skew angle: {angle:.2f} degrees")
        return angle

    @staticmethod
    def supports_generate_ocr() -> bool:
        return True

    @staticmethod
    def generate_ocr(input_file, options, page_number=0) -> tuple[OcrElement, str]:
        img = cv.imread(os.fspath(input_file))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        batch_size = getattr(options, "easyocr_batch_size", DEFAULT_BATCH_SIZE)
        with _reader_lock:
            reader = _get_reader(options)
            raw_results = reader.readtext(gray, batch_size=batch_size)

        results = [tidy_easyocr_result(r) for r in raw_results]
        text = " ".join(result.text for result in results)

        page = _easyocr_to_ocr_tree(input_file, results, page_number)
        return page, text

    @staticmethod
    def generate_hocr(input_file, output_hocr, output_text, options):
        raise NotImplementedError(
            "EasyOCR uses the generate_ocr() API; hOCR output is not produced."
        )

    @staticmethod
    def generate_pdf(input_file, output_pdf, output_text, options):
        raise NotImplementedError(
            "EasyOCR uses the generate_ocr() API; OCRmyPDF renders the text layer."
        )


@hookimpl
def get_ocr_engine(options):
    return EasyOCREngine()
