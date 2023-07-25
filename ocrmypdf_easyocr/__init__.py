# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT


import logging
import os
from math import atan2, cos, hypot, sin
from pathlib import Path
from typing import NamedTuple

from multiprocessing import Semaphore
import importlib.resources

import cv2 as cv
import easyocr
import pluggy
from ocrmypdf import OcrEngine, hookimpl
from ocrmypdf._exec import tesseract
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen.canvas import Canvas
from pikepdf import (
    Pdf,
    ContentStreamInstruction,
    Operator,
    Dictionary,
    Name,
    unparse_content_stream,
)

log = logging.getLogger(__name__)


ISO_639_3_2 = {
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
GPU_SEMAPHORE = Semaphore(1)


@hookimpl
def initialize(plugin_manager: pluggy.PluginManager):
    # plugin_manager.set_blocked("ocrmypdf.builtin_plugins.tesseract_ocr")
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


def register_glyphlessfont(pdf):
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


def easyocr_to_pikepdf(image_filename, image_scale, results, output_pdf):
    with Image.open(image_filename) as im:
        dpi = im.info["dpi"]
        scale = 72.0 / dpi[0] / image_scale, 72.0 / dpi[1] / image_scale
        width = im.width
        height = im.height
    pdf = Pdf.new()
    pdf.add_blank_page(page_size=(width * scale[0], height * scale[1]))

    pdf.pages[0].Resources = Dictionary(
        Font=Dictionary({"/f-0-0": register_glyphlessfont(pdf)})
    )

    cs = []
    cs.append(ContentStreamInstruction([], Operator("q")))

    for result in results:
        log.info(f"Textline '{result.text}' in-image bbox: {bbox_string(result.quad)}")
        bbox = pt_from_pixel(result.quad, scale, height)

        angle = -atan2(bbox[5] - bbox[7], bbox[4] - bbox[6])
        if abs(angle) < 0.01:  # 0.01 radians is 0.57 degrees
            angle = 0.0
        cos_a, sin_a = cos(angle), sin(angle)

        cs.append(ContentStreamInstruction([], Operator("BT")))
        cs.append(ContentStreamInstruction([3], Operator("Tr")))  # Invisible ink
        cs.append(
            ContentStreamInstruction(
                [cos_a, -sin_a, sin_a, cos_a, bbox[6], bbox[7]], Operator("Tm")
            )
        )

        font_size = hypot(bbox[0] - bbox[6], bbox[1] - bbox[7])
        cs.append(ContentStreamInstruction([Name("/f-0-0"), font_size], Operator("Tf")))
        log.info(f"Textline '{result.text}' PDF bbox: {bbox_string(bbox)}")
        space_width = 0
        box_width = hypot(bbox[4] - bbox[6], bbox[5] - bbox[7]) + space_width
        if len(result.text) == 0 or box_width == 0 or font_size == 0:
            continue
        h_stretch = 100.0 * box_width / len(result.text) / font_size * CHAR_ASPECT
        cs.append(ContentStreamInstruction([h_stretch], Operator("Tz")))

        cs.append(
            ContentStreamInstruction(
                [[result.text.encode("utf-16be")]],
                Operator("TJ"),
            )
        )
        cs.append(ContentStreamInstruction([], Operator("ET")))

    cs.append(ContentStreamInstruction([], Operator("Q")))
    pdf.pages[0].Contents = pdf.make_stream(unparse_content_stream(cs))

    pdf.save(output_pdf)
    return output_pdf


def easyocr_to_pdf(image_filename, image_scale, results, output_pdf):
    with Image.open(image_filename) as im:
        dpi = im.info["dpi"]
        scale = 72.0 / dpi[0] / image_scale, 72.0 / dpi[1] / image_scale
        width = im.width
        height = im.height
        pdf = Canvas(
            os.fspath(output_pdf),
            pagesize=(width * scale[0], height * scale[1]),
            bottomup=1,
            pageCompression=1,
        )

    font_name = "Helvetica"
    helv = pdfmetrics.getFont(font_name)
    # helv.substitutionFonts

    # helv = pdfmetrics.Font(
    #         name=font_name,
    #         faceName='Helvetica',
    #         encName='WinAnsiEncoding',
    #         substitutionFonts=['Nimbus Sans', 'Roboto', 'Arial'],
    #     )
    # pdfmetrics.registerFont(helv)

    for result in results:
        if not result.text.isascii():
            continue  # for now

        log.info(f"Word '{result.text}' in-image bbox: {bbox_string(result.quad)}")
        bbox = pt_from_pixel(result.quad, scale, height)

        if TEXT_POSITION_DEBUG:
            pdf.setDash()
            pdf.setStrokeColorRGB(0.95, 0.65, 0.95)
            pdf.setLineWidth(0.5)
            pdf.lines(
                [
                    bbox[0:4],
                    bbox[2:6],
                    bbox[4:8],
                    [*bbox[6:8], *bbox[0:2]],
                ]
            )
            pdf.cross(bbox[6], bbox[7])  # bottom left
            pdf.circle(bbox[4], bbox[5], 3)  # bottom right
            pdf.circle(bbox[0], bbox[1], 1.5)  # top left

        angle = -atan2(bbox[5] - bbox[7], bbox[4] - bbox[6])
        if abs(angle) < 0.01:  # 0.01 radians is 0.57 degrees
            angle = 0.0
        cos_a, sin_a = cos(angle), sin(angle)

        text = pdf.beginText()
        text.setTextTransform(cos_a, -sin_a, sin_a, cos_a, bbox[6], bbox[7])

        fontsize = hypot(bbox[0] - bbox[6], bbox[1] - bbox[7])

        log.info(f"Word '{result.text}' PDF bbox: {bbox_string(bbox)}")
        text.setFont(font_name, fontsize)
        cursor = text.getStartOfLine()

        word_width = pdf.stringWidth(result.text, font_name, fontsize)
        space_width = 0
        # box_width = bbox[4] - bbox[6] + space_width
        box_width = hypot(bbox[4] - bbox[6], bbox[5] - bbox[7]) + space_width
        last_word = False
        if word_width > 0:
            text.setHorizScale(100.0 * box_width / (word_width + space_width))
            if not last_word:
                text.textOut(result.text + " ")
            else:
                text.textOut(result.text)
        pdf.drawText(text)

    pdf.showPage()
    pdf.save()
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

        with GPU_SEMAPHORE:
            reader = easyocr.Reader(languages, gpu=options.gpu)
            img = cv.imread(os.fspath(input_file))
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            raw_results = reader.readtext(gray)
        results = [tidy_easyocr_result(r) for r in raw_results]

        text = " ".join([result.text for result in results])
        output_text.write_text(text)

        # easyocr_to_pdf(input_file, 1.0, results, output_pdf)
        easyocr_to_pikepdf(input_file, 1.0, results, output_pdf)


@hookimpl
def get_ocr_engine():
    return EasyOCREngine()
