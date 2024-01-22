# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""EasyOCR plugin for OCRmyPDF."""

from __future__ import annotations

import logging
import os
import multiprocessing
import multiprocessing.managers
import threading

import cv2 as cv
import easyocr
import pluggy
from ocrmypdf import OcrEngine, hookimpl
from ocrmypdf._exec import tesseract
import time

from ocrmypdf_easyocr._cv import detect_skew
from ocrmypdf_easyocr._easyocr import tidy_easyocr_result
from ocrmypdf_easyocr._pdf import easyocr_to_pikepdf

from typing import Optional, Tuple
import numpy.typing as npt

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

Task = Tuple[npt.NDArray, multiprocessing.Value, threading.Event]

def _ocrThread(q: multiprocessing.Queue[Task], options):
    reader: Optional[easyocr.Reader] = None

    # TODO: signal _ocrThread to quit after OCR completes.
    while True:
        (gray, outputDict, event) = q.get()


        # Init reader on first OCR attempt: Wait until `options` variable is fully initialized.
        # Note: `options` variable is on the same process with the main thread.
        try:
            if reader is None:
                useGPU = options.gpu
                languages = [ISO_639_3_2[lang] for lang in options.languages]
                reader = easyocr.Reader(languages, useGPU)
            outputDict["output"] = reader.readtext(
                gray,
                batch_size=options.easyocr_batch_size,
                workers=options.easyocr_workers
            )
        except Exception as e:
            print(e)
            outputDict["output"] = ""
        finally:
            event.set()


@hookimpl
def initialize(plugin_manager: pluggy.PluginManager):
    pass


@hookimpl
def check_options(options):
    m = multiprocessing.Manager()
    q = multiprocessing.Queue(-1)
    t = threading.Thread(target=_ocrThread, args=(q, options), daemon=True)
    t.start()
    options._easyocr_struct = {
        "manager": m,
        "queue": q
    }

@hookimpl
def add_options(parser):
    easyocr_options = parser.add_argument_group(
        "EasyOCR", "Advanced control of EasyOCR"
    )
    easyocr_options.add_argument("--easyocr-no-gpu", action="store_false", dest="gpu")
    easyocr_options.add_argument("--easyocr-batch-size", type=int, default=4)
    easyocr_options.add_argument("--easyocr-workers", type=int, default=0)


class EasyOCREngine(OcrEngine):
    """Implements OCR with EasyOCR."""

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
        img = cv.imread(os.fspath(input_file))
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        s = options._easyocr_struct
        manager: multiprocessing.managers.SyncManager = s["manager"]
        queue: multiprocessing.Queue[Task] = s["queue"]
        outputDict = manager.dict()
        event = manager.Event()
        queue.put((gray, outputDict, event))
        event.wait()
        raw_results = outputDict["output"]

        results = [tidy_easyocr_result(r) for r in raw_results]
        text = " ".join([result.text for result in results])
        output_text.write_text(text)

        # easyocr_to_pdf(input_file, 1.0, results, output_pdf)
        easyocr_to_pikepdf(input_file, 1.0, results, output_pdf)


@hookimpl
def get_ocr_engine():
    return EasyOCREngine()
