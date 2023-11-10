# SPDX-FileCopyrightText: 2023 James R. Barlow
# SPDX-License-Identifier: MIT

"""Convert EasyOCR results to a PDF with text annotations (no images)."""

from __future__ import annotations

import importlib.resources
import logging
from math import atan2, cos, hypot, sin
from pathlib import Path
from typing import Iterable

from pikepdf import (
    ContentStreamInstruction,
    Dictionary,
    Name,
    Operator,
    Pdf,
    unparse_content_stream,
)
from PIL import Image

from ocrmypdf_easyocr._easyocr import EasyOCRResult

log = logging.getLogger(__name__)
TEXT_POSITION_DEBUG = False
GLYPHLESS_FONT = importlib.resources.read_binary("ocrmypdf_easyocr", "pdf.ttf")
CHAR_ASPECT = 2


def pt_from_pixel(bbox, scale, height):
    point_pairs = [
        (x * scale[0], (height - y) * scale[1]) for x, y in zip(bbox[0::2], bbox[1::2])
    ]
    return [elm for pt in point_pairs for elm in pt]


def bbox_string(bbox):
    return ", ".join(f"{elm:.0f}" for elm in bbox)


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


class ContentStreamBuilder:
    def __init__(self, instructions=None):
        self._instructions = instructions or []

    def q(self):
        """Save the graphics state."""
        inst = [ContentStreamInstruction([], Operator("q"))]
        return ContentStreamBuilder(self._instructions + inst)

    def Q(self):
        """Restore the graphics state."""
        inst = [ContentStreamInstruction([], Operator("Q"))]
        return ContentStreamBuilder(self._instructions + inst)

    def cm(self, a: float, b: float, c: float, d: float, e: float, f: float):
        """Concatenate matrix."""
        inst = [ContentStreamInstruction([a, b, c, d, e, f], Operator("cm"))]
        return ContentStreamBuilder(self._instructions + inst)

    def BT(self):
        """Begin text object."""
        inst = [ContentStreamInstruction([], Operator("BT"))]
        return ContentStreamBuilder(self._instructions + inst)

    def ET(self):
        """End text object."""
        inst = [ContentStreamInstruction([], Operator("ET"))]
        return ContentStreamBuilder(self._instructions + inst)

    def BDC(self, mctype: Name, mcid: int):
        """Begin marked content sequence."""
        inst = [
            ContentStreamInstruction([mctype, Dictionary(MCID=mcid)], Operator("BDC"))
        ]
        return ContentStreamBuilder(self._instructions + inst)

    def EMC(self):
        """End marked content sequence."""
        inst = [ContentStreamInstruction([], Operator("EMC"))]
        return ContentStreamBuilder(self._instructions + inst)

    def Tf(self, font: Name, size: int):
        """Set text font and size."""
        inst = [ContentStreamInstruction([font, size], Operator("Tf"))]
        return ContentStreamBuilder(self._instructions + inst)

    def Tm(self, a: float, b: float, c: float, d: float, e: float, f: float):
        """Set text matrix."""
        inst = [ContentStreamInstruction([a, b, c, d, e, f], Operator("Tm"))]
        return ContentStreamBuilder(self._instructions + inst)

    def Tr(self, mode: int):
        """Set text rendering mode."""
        inst = [ContentStreamInstruction([mode], Operator("Tr"))]
        return ContentStreamBuilder(self._instructions + inst)

    def Tz(self, scale: float):
        """Set text horizontal scaling."""
        inst = [ContentStreamInstruction([scale], Operator("Tz"))]
        return ContentStreamBuilder(self._instructions + inst)

    def TJ(self, text):
        """Show text."""
        inst = [ContentStreamInstruction([[text.encode("utf-16be")]], Operator("TJ"))]
        return ContentStreamBuilder(self._instructions + inst)

    def s(self):
        """Stroke and close path."""
        inst = [ContentStreamInstruction([], Operator("s"))]
        return ContentStreamBuilder(self._instructions + inst)

    def re(self, x: float, y: float, w: float, h: float):
        """Append rectangle to path."""
        inst = [ContentStreamInstruction([x, y, w, h], Operator("re"))]
        return ContentStreamBuilder(self._instructions + inst)

    def RG(self, r: float, g: float, b: float):
        """Set RGB stroke color."""
        inst = [ContentStreamInstruction([r, g, b], Operator("RG"))]
        return ContentStreamBuilder(self._instructions + inst)

    def build(self):
        return self._instructions

    def add(self, other: ContentStreamBuilder):
        return ContentStreamBuilder(self._instructions + other._instructions)


def generate_text_content_stream(
    results: Iterable[EasyOCRResult],
    scale: tuple[float, float],
    height: int,
    boxes=False,
):
    """Generate a content stream for the described by results.

    Args:
        results (Iterable[EasyOCRResult]): Results of OCR.
        scale (tuple[float, float]): Scale of the image.
        height (int): Height of the image.

    Yields:
        ContentStreamInstruction: Content stream instructions.
    """

    cs = ContentStreamBuilder()
    cs = cs.add(cs.q())
    for n, result in enumerate(results):
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

        cs = cs.add(
            ContentStreamBuilder()
            .BT()
            .BDC(Name.Span, n)
            .Tr(3)  # Invisible ink
            .Tm(cos_a, -sin_a, sin_a, cos_a, bbox[6], bbox[7])
            .Tf(Name("/f-0-0"), font_size)
            .Tz(h_stretch)
            .TJ(result.text)
            .EMC()
            .ET()
        )
        if boxes:
            cs = cs.add(
                ContentStreamBuilder()
                .q()
                .cm(cos_a, -sin_a, sin_a, cos_a, bbox[6], bbox[7])
                .re(0, 0, box_width, font_size)
                .RG(1, 0, 0)
                .s()
                .Q()
            )

    cs = cs.Q()
    return cs.build()


def easyocr_to_pikepdf(
    image_filename: Path,
    image_scale: float,
    results: Iterable[EasyOCRResult],
    output_pdf: Path,
    boxes: bool,
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

        cs = generate_text_content_stream(results, scale, height, boxes=boxes)
        pdf.pages[0].Contents = pdf.make_stream(unparse_content_stream(cs))

        pdf.save(output_pdf)
    return output_pdf
