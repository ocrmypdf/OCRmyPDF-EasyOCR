# OCRmyPDF EasyOCR

This is plugin to run OCRmyPDF with the EasyOCR engine instead of Tesseract OCR,
the default OCR engine for OCRmyPDF. Since EasyOCR is based on PyTorch, it makes
use of Nvidia GPUs. Hopefully it will be more performant and accurate than Tesseract OCR.

It is currently experimental and does not implement all of the features of
OCRmyPDF with Tesseract, and still relies on Tesseract for certain operations.

## Installation

To use this plugin, first
[install PyTorch according to the official instructions](https://pytorch.org/),
which may differ for your platform.

Then install OCRmyPDF-EasyOCR to the same virtual environment or conda environment
as you installed PyTorch:

```bash
pip install git+https://github.com/ocrmypdf/OCRmyPDF-EasyOCR.git
```

The OCRmyPDF-EasyOCR will override Tesseract for OCR; however, OCR still depends
on Tesseract for some tasks.

If [Celery's multiprocessing](https://docs.celeryq.dev/en/stable/getting-started/introduction.html)
is installed in the virtual environment, it will be used instead of the standard
Python multiprocessing. This allows paperless-ngx, which uses Celery, to function correctly.

## Troubleshooting

If you see a log message
``Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU``
then PyTorch is not installed.

## To do

Contributions, especially pull requests are quite welcome!

At the moment this plugin is alpha status and missing some essential features:
- Tesseract is still required for determine page skew and for orientation correction
- EasyOCR is effectively single threaded, to eliminate race conditions


