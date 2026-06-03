# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""OCRmyPDF plugin hooks and engine for EasyOCR.

This module wires EasyOCR (see :mod:`ocrmypdf_easyocr._easyocr`) into OCRmyPDF:
it registers options, builds the OcrElement tree OCRmyPDF renders, and selects
EasyOCR as the OCR engine.
"""

from __future__ import annotations

import logging
import os
from math import atan2, degrees

import cv2 as cv
from ocrmypdf import BoundingBox, OcrClass, OcrElement, OcrEngine, hookimpl
from ocrmypdf._exec import tesseract
from PIL import Image

from ocrmypdf_easyocr import _easyocr
from ocrmypdf_easyocr._cv import detect_skew

log = logging.getLogger(__name__)

# Name this engine registers under for OCRmyPDF's --ocr-engine option.
OCR_ENGINE_NAME = "easyocr"


def _register_ocr_engine_choice(parser):
    """Add ``easyocr`` to the core ``--ocr-engine`` choices, if present.

    OCRmyPDF defines ``--ocr-engine`` with a fixed set of choices; argparse has no
    public API to extend them, so we reach into the existing action. This lets
    users explicitly select ``--ocr-engine easyocr``.
    """
    for action in parser._actions:
        if action.dest == "ocr_engine" and action.choices is not None:
            if OCR_ENGINE_NAME not in action.choices:
                action.choices = [*action.choices, OCR_ENGINE_NAME]
            return


@hookimpl
def add_options(parser):
    _register_ocr_engine_choice(parser)

    easyocr_options = parser.add_argument_group("EasyOCR", "EasyOCR options")
    easyocr_options.add_argument(
        "--easyocr-no-gpu", action="store_false", dest="gpu", default=_easyocr.DEFAULT_GPU
    )
    easyocr_options.add_argument(
        "--easyocr-batch-size", type=int, default=_easyocr.DEFAULT_BATCH_SIZE
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
        bbox = BoundingBox(left=min(xs), top=min(ys), right=max(xs), bottom=max(ys))

        # Rotation of the text, in degrees counter-clockwise (per the hOCR
        # convention OCRmyPDF's renderer uses). Image y grows downward, so a
        # counter-clockwise rotation makes the top edge (ul -> ur) rise, i.e. a
        # decreasing y; hence the negated dy.
        ulx, uly, urx, ury = (
            result.quad[0],
            result.quad[1],
            result.quad[2],
            result.quad[3],
        )
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
        return _easyocr.version()

    @staticmethod
    def creator_tag(options):
        return f"EasyOCR {EasyOCREngine.version()}"

    def __str__(self):
        return f"EasyOCR {EasyOCREngine.version()}"

    @staticmethod
    def languages(options):
        return set(_easyocr.SUPPORTED_LANGUAGES)

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
        if page_number == 0:
            log.debug("EasyOCR processing %s", input_file)
        img = cv.imread(os.fspath(input_file))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        results = _easyocr.readtext(
            gray,
            languages=options.languages,
            gpu=getattr(options, "gpu", _easyocr.DEFAULT_GPU),
            batch_size=getattr(options, "easyocr_batch_size", _easyocr.DEFAULT_BATCH_SIZE),
        )
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


@hookimpl(tryfirst=True)
def get_ocr_engine(options):
    """Select EasyOCR as the OCR engine.

    ``get_ocr_engine`` is a firstresult hook shared with OCRmyPDF's built-in
    Tesseract and null engines. We run first (``tryfirst``) so that, when this
    plugin is installed, EasyOCR is used for the default ``--ocr-engine auto`` as
    well as the explicit ``--ocr-engine easyocr``. Returning ``None`` for any
    other selection lets the built-in engines (``tesseract``, ``none``) handle it.
    """
    if options is not None:
        ocr_engine = getattr(options, "ocr_engine", "auto")
        if ocr_engine not in ("auto", OCR_ENGINE_NAME):
            return None
    return EasyOCREngine()
