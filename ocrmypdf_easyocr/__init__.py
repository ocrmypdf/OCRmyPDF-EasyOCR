# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""EasyOCR plugin for OCRmyPDF."""

from __future__ import annotations

import logging
import billiard as multiprocessing
import os
import threading
import traceback
from pathlib import Path
from typing import Optional, Sequence, Tuple

import cv2 as cv
import easyocr
import numpy.typing as npt
import pluggy
from ocrmypdf import Executor, OcrEngine, PdfContext, hookimpl
from ocrmypdf._exec import tesseract
from ocrmypdf.builtin_plugins.optimize import optimize_pdf as default_optimize_pdf

from ocrmypdf_easyocr._cv import detect_skew
from ocrmypdf_easyocr._easyocr import tidy_easyocr_result
from ocrmypdf_easyocr._pdf import easyocr_to_pikepdf

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

Task = Tuple[npt.NDArray, multiprocessing.Value, threading.Event] | None


def _ocr_process(q: multiprocessing.Queue[Task], options):
    reader: Optional[easyocr.Reader] = None

    while True:
        message = q.get()
        if message is None:
            return  # exit process
        gray, output_dict, event = message

        # Init reader on first OCR attempt: Wait until `options` variable is fully initialized.
        # Note: `options` variable is on the same process with the main thread.
        try:
            if reader is None:
                use_gpu = options.gpu
                languages = [ISO_639_3_2[lang] for lang in options.languages]
                reader = easyocr.Reader(languages, use_gpu)
            output_dict["output"] = reader.readtext(
                gray, batch_size=options.easyocr_batch_size
            )
        except Exception as e:
            traceback.print_exception(e)
            output_dict["output"] = ""
        finally:
            event.set()


@hookimpl
def initialize(plugin_manager: pluggy.PluginManager):
    pass


class ProcessList:
    def __init__(self, plist):
        self.process_list = plist

    def __getstate__(self):
        return []


@hookimpl
def check_options(options):
    m = multiprocessing.Manager()
    q = multiprocessing.Queue(-1)
    ocr_process_list = []
    for _ in range(options.easyocr_workers):
        t = multiprocessing.Process(target=_ocr_process, args=(q, options), daemon=True)
        t.start()
        ocr_process_list.append(t)

    options._easyocr_struct = {"manager": m, "queue": q}
    options._easyocr_plist = ProcessList(ocr_process_list)


@hookimpl
def optimize_pdf(
    input_pdf: Path,
    output_pdf: Path,
    context: PdfContext,
    executor: Executor,
    linearize: bool,
) -> tuple[Path, Sequence[str]]:
    options = context.options
    for _ in range(options.easyocr_workers):
        q = options._easyocr_struct["queue"]
        q.put(None)  # send stop message
    for p in options._easyocr_plist.process_list:
        p.join(3.0)  # clean up child processes but don't wait forever

    return default_optimize_pdf(
        input_pdf=input_pdf,
        output_pdf=output_pdf,
        context=context,
        executor=executor,
        linearize=linearize,
    )


@hookimpl
def add_options(parser):
    easyocr_options = parser.add_argument_group("EasyOCR", "EasyOCR options")
    easyocr_options.add_argument("--easyocr-no-gpu", action="store_false", dest="gpu")
    easyocr_options.add_argument("--easyocr-batch-size", type=int, default=4)
    easyocr_options.add_argument("--easyocr-workers", type=int, default=1)
    easyocr_options.add_argument(
        "--easyocr-debug-suppress-images",
        action="store_true",
        dest="easyocr_debug_suppress_images",
    )


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

        sync_data = options._easyocr_struct
        manager: multiprocessing.managers.SyncManager = sync_data["manager"]
        queue: multiprocessing.Queue[Task] = sync_data["queue"]
        output_dict = manager.dict()
        event = manager.Event()
        queue.put((gray, output_dict, event))
        event.wait()
        raw_results = output_dict["output"]

        results = [tidy_easyocr_result(r) for r in raw_results]
        text = " ".join([result.text for result in results])
        output_text.write_text(text)

        easyocr_to_pikepdf(
            input_file,
            1.0,
            results,
            output_pdf,
            boxes=options.easyocr_debug_suppress_images,
        )


@hookimpl
def get_ocr_engine():
    return EasyOCREngine()
