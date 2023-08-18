# OCRmyPDF EasyOCR

This is plugin to run OCRmyPDF with the EasyOCR engine instead of Tesseract OCR,
the default OCR engine for OCRmyPDF. Since EasyOCR is based on PyTorch, it makes
use of Nvidia GPUs. Hopefully it will be more performant and accurate than Tesseract OCR.

It is currently experimental and does not implement all of the features of
OCRmyPDF with Tesseract, and still relies on Tesseract for certain operations.

## Installation

To test this plugin, create a new virtual environment and install, as follows:

```bash
python -m venv .venv
source .venv/bin/activate
pip install git+https://github.com/ocrmypdf/OCRmyPDF-EasyOCR.git
```

When installed in the same virtual environment as OCRmyPDF, it will override Tesseract.

## To do

Contributions, especially pull requests are quite welcome!

At the moment this plugin is alpha status and missing some essential features:
- Tesseract is still required for determine page skew and for orientation correction
- EasyOCR is effectively single threaded, to eliminate race conditions


