# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""EasyOCR plugin for OCRmyPDF."""

from __future__ import annotations

import importlib.resources
import logging
import os
from math import atan2, cos, hypot, sin
from multiprocessing import Semaphore
from pathlib import Path
from typing import Iterable, NamedTuple

import cv2 as cv
import easyocr
import pluggy
from ocrmypdf import OcrEngine, hookimpl
from ocrmypdf._exec import tesseract
from pikepdf import (
    ContentStreamInstruction,
    Dictionary,
    Name,
    Operator,
    Pdf,
    unparse_content_stream,
)
from PIL import Image

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

TEXT_POSITION_DEBUG = False
GLYPHLESS_FONT = importlib.resources.read_binary("ocrmypdf_easyocr", "pdf.ttf")
GPU_SEMAPHORE = Semaphore(3)


@hookimpl
def initialize(plugin_manager: pluggy.PluginManager):
    pass


@hookimpl
def add_options(parser):
    easyocr_options = parser.add_argument_group(
        "EasyOCR", "Advanced control of EasyOCR"
    )
    easyocr_options.add_argument("--easyocr-no-gpu", action="store_false", dest="gpu")


class EasyOCRQuad(NamedTuple):
    ul: tuple[int, int]
    ur: tuple[int, int]
    lr: tuple[int, int]
    ll: tuple[int, int]


class EasyOCRResult(NamedTuple):
    """Result of OCR with EasyOCR."""

    quad: list
    text: str
    confidence: float


def pt_from_pixel(bbox, scale, height):
    point_pairs = [
        (x * scale[0], (height - y) * scale[1]) for x, y in zip(bbox[0::2], bbox[1::2])
    ]
    return [elm for pt in point_pairs for elm in pt]


def bbox_string(bbox):
    return ", ".join(f"{elm:.0f}" for elm in bbox)


CHAR_ASPECT = 2


def register_glyphlessfont(pdf: Pdf):
    """Register the glyphless font.

    Create several data structures in the Pdf to describe the font. While it create
    the data, a reference should be set in at least one page's /Resources dictionary
    to retain the font in the output PDF and ensure it is usable on that page.
    """
    PLACEHOLDER = Name.Placeholder

    basefont = pdf.make_indirect(
        Dictionary(
            BaseFont=Name.GlyphLessFont,
            DescendantFonts=[PLACEHOLDER],
            Encoding=Name("/Identity-H"),
            Subtype=Name.Type0,
            ToUnicode=PLACEHOLDER,
            Type=Name.Font,
        )
    )
    cid_font_type2 = pdf.make_indirect(
        Dictionary(
            BaseFont=Name.GlyphLessFont,
            CIDToGIDMap=PLACEHOLDER,
            CIDSystemInfo=Dictionary(
                Ordering="Identity",
                Registry="Adobe",
                Supplement=0,
            ),
            FontDescriptor=PLACEHOLDER,
            Subtype=Name.CIDFontType2,
            Type=Name.Font,
            DW=1000 // CHAR_ASPECT,
        )
    )
    basefont.DescendantFonts = [cid_font_type2]
    cid_font_type2.CIDToGIDMap = pdf.make_stream(b"\x00\x01" * 65536)
    basefont.ToUnicode = pdf.make_stream(
        b"/CIDInit /ProcSet findresource begin\n"
        b"12 dict begin\n"
        b"begincmap\n"
        b"/CIDSystemInfo\n"
        b"<<\n"
        b"  /Registry (Adobe)\n"
        b"  /Ordering (UCS)\n"
        b"  /Supplement 0\n"
        b">> def\n"
        b"/CMapName /Adobe-Identify-UCS def\n"
        b"/CMapType 2 def\n"
        b"1 begincodespacerange\n"
        b"<0000> <FFFF>\n"
        b"endcodespacerange\n"
        b"1 beginbfrange\n"
        b"<0000> <FFFF> <0000>\n"
        b"endbfrange\n"
        b"endcmap\n"
        b"CMapName currentdict /CMap defineresource pop\n"
        b"end\n"
        b"end\n"
    )
    font_descriptor = pdf.make_indirect(
        Dictionary(
            Ascent=1000,
            CapHeight=1000,
            Descent=-1,
            Flags=5,  # Fixed pitch and symbolic
            FontBBox=[0, 0, 1000 // CHAR_ASPECT, 1000],
            FontFile2=PLACEHOLDER,
            FontName=Name.GlyphLessFont,
            ItalicAngle=0,
            StemV=80,
            Type=Name.FontDescriptor,
        )
    )
    font_descriptor.FontFile2 = pdf.make_stream(GLYPHLESS_FONT)
    cid_font_type2.FontDescriptor = font_descriptor
    return basefont


def cs_q():
    return ContentStreamInstruction([], Operator("q"))


def cs_Q():
    return ContentStreamInstruction([], Operator("Q"))


def cs_BT():
    return ContentStreamInstruction([], Operator("BT"))


def cs_ET():
    return ContentStreamInstruction([], Operator("ET"))


def cs_Tf(font, size):
    return ContentStreamInstruction([font, size], Operator("Tf"))


def cs_Tm(a, b, c, d, e, f):
    return ContentStreamInstruction([a, b, c, d, e, f], Operator("Tm"))


def cs_Tr(mode):
    return ContentStreamInstruction([mode], Operator("Tr"))


def cs_Tz(scale):
    return ContentStreamInstruction([scale], Operator("Tz"))


def cs_TJ(text):
    return ContentStreamInstruction([[text.encode("utf-16be")]], Operator("TJ"))


def generate_text_content_stream(
    results: Iterable[EasyOCRResult], scale: tuple[float, float], height: int
):
    """Generate a content stream for the described by results.

    Args:
        results (Iterable[EasyOCRResult]): Results of OCR.
        scale (tuple[float, float]): Scale of the image.
        height (int): Height of the image.

    Yields:
        ContentStreamInstruction: Content stream instructions.
    """

    yield cs_q()
    for result in results:
        log.debug(f"Textline '{result.text}' in-image bbox: {bbox_string(result.quad)}")
        bbox = pt_from_pixel(result.quad, scale, height)
        angle = -atan2(bbox[5] - bbox[7], bbox[4] - bbox[6])
        if abs(angle) < 0.01:  # 0.01 radians is 0.57 degrees
            angle = 0.0
        cos_a, sin_a = cos(angle), sin(angle)

        font_size = hypot(bbox[0] - bbox[6], bbox[1] - bbox[7])

        log.debug(f"Textline '{result.text}' PDF bbox: {bbox_string(bbox)}")
        space_width = 0
        box_width = hypot(bbox[4] - bbox[6], bbox[5] - bbox[7]) + space_width
        if len(result.text) == 0 or box_width == 0 or font_size == 0:
            continue
        h_stretch = 100.0 * box_width / len(result.text) / font_size * CHAR_ASPECT

        yield cs_BT()
        yield cs_Tr(3)  # Invisible ink
        yield cs_Tm(cos_a, -sin_a, sin_a, cos_a, bbox[6], bbox[7])
        yield cs_Tf(Name("/f-0-0"), font_size)
        yield cs_Tz(h_stretch)
        yield cs_TJ(result.text)
        yield cs_ET()
    yield cs_Q()


def easyocr_to_pikepdf(
    image_filename: Path,
    image_scale: float,
    results: Iterable[EasyOCRResult],
    output_pdf: Path,
):
    """Convert EasyOCR results to a PDF with text annotations (no images).

    Args:
        image_filename: Path to the image file that was OCR'd.
        image_scale: Scale factor applied to the OCR image. 1.0 means the
            image is at the scale implied by its DPI. 2.0 means the image
            is twice as large as implied by its DPI.
        results: List of EasyOCRResult objects.
        output_pdf: Path to the output PDF file that this will function will
            create.

    Returns:
        output_pdf
    """

    with Image.open(image_filename) as im:
        dpi = im.info["dpi"]
        scale = 72.0 / dpi[0] / image_scale, 72.0 / dpi[1] / image_scale
        width = im.width
        height = im.height

    with Pdf.new() as pdf:
        pdf.add_blank_page(page_size=(width * scale[0], height * scale[1]))
        pdf.pages[0].Resources = Dictionary(
            Font=Dictionary({"/f-0-0": register_glyphlessfont(pdf)})
        )

        cs = list(generate_text_content_stream(results, scale, height))
        pdf.pages[0].Contents = pdf.make_stream(unparse_content_stream(cs))

        pdf.save(output_pdf)
    return output_pdf


def tidy_easyocr_result(raw_result) -> EasyOCRResult:
    """Converts EasyOCR results to a more convenient format."""
    return EasyOCRResult(
        quad=[el for sublist in raw_result[0] for el in sublist],  # flatten list
        text=raw_result[1],
        confidence=raw_result[2],
    )


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
        return tesseract.get_deskew(
            input_file,
            languages=options.languages,
            engine_mode=options.tesseract_oem,
            timeout=options.tesseract_non_ocr_timeout,
        )

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
