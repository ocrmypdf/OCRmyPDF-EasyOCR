# ocrmypdf-easyocr

This is plugin to run OCRmyPDF with the EasyOCR engine instead of Tesseract OCR,
the default OCR engine for OCRmyPDF. Since EasyOCR is based on PyTorch, it makes
use of Nvidia GPUs. Hopefully it will be more performant and accurate than Tesseract OCR.

It is currently experimental and does not implement all of the features of
OCRmyPDF with Tesseract, and still relies on Tesseract for certain operations.

A major limitation is that reportlab is used for rendering the PDF OCR text layer,
so at the moment, only Latin alphabet languages are properly supported.

Contributions, especially pull requests are quite welcome!

To test this plugin, create a new virtual environment and install, as follows:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

When installed in the same virtual environment as OCRmyPDF, it will override Tesseract.
