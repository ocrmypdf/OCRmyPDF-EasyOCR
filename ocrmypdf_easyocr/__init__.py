# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""EasyOCR plugin for OCRmyPDF."""

from __future__ import annotations

import logging
import os
from multiprocessing import Semaphore

import cv2 as cv
import easyocr
import pluggy
from ocrmypdf import OcrEngine, hookimpl
from ocrmypdf._exec import tesseract

from ocrmypdf_easyocr._cv import detect_skew
from ocrmypdf_easyocr._easyocr import tidy_easyocr_result
from ocrmypdf_easyocr._pdf import easyocr_to_pikepdf

log = logging.getLogger(__name__)

GPU_SEMAPHORE = Semaphore(3)

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


@hookimpl
def initialize(plugin_manager: pluggy.PluginManager):
    pass


@hookimpl
def add_options(parser):
    easyocr_options = parser.add_argument_group(
        "EasyOCR", "Advanced control of EasyOCR"
    )
    easyocr_options.add_argument("--easyocr-no-gpu", action="store_false", dest="gpu")


class EasyOCREngine(OcrEngine):
    """Implements OCR with Tesseract."""

    @staticmethod
    def version():
        return easyocr.__version__

    @staticmethod
    def creator_tag(options):
        tag = "-PDF" if options.pdf_renderer == "sandwich" else ""
        return f"EasyOCR{tag} {EasyOCREngine.version()}"

    def __str__(self):
        return f"EasyOCR {EasyOCREngine.version()}"

    @staticmethod
    def languages(options):
        return ISO_639_3_2.keys()

    @staticmethod
    def get_orientation(input_file, options):
        return tesseract.get_orientation(
            input_file,
            engine_mode=options.tesseract_oem,
            timeout=options.tesseract_non_ocr_timeout,
        )

    @staticmethod
    def get_deskew(input_file, options) -> float:
        img = cv.imread(os.fspath(input_file))
        angle = detect_skew(img)
        log.debug(f"Detected skew angle: {angle:.2f} degrees")
        return angle

    @staticmethod
    def generate_hocr(input_file, output_hocr, output_text, options):
        raise NotImplementedError("EasyOCR does not support hOCR output")

    @staticmethod
    def generate_pdf(input_file, output_pdf, output_text, options):
        languages = [ISO_639_3_2[lang] for lang in options.languages]

        img = cv.imread(os.fspath(input_file))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        with GPU_SEMAPHORE:
            reader = easyocr.Reader(languages, gpu=options.gpu)
            raw_results = reader.readtext(gray)
        results = [tidy_easyocr_result(r) for r in raw_results]

        text = " ".join([result.text for result in results])
        output_text.write_text(text)

        # easyocr_to_pdf(input_file, 1.0, results, output_pdf)
        easyocr_to_pikepdf(input_file, 1.0, results, output_pdf)


@hookimpl
def get_ocr_engine():
    return EasyOCREngine()
