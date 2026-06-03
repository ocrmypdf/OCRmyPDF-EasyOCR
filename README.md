# OCRmyPDF EasyOCR

This is a plugin to run OCRmyPDF with the EasyOCR engine instead of Tesseract OCR,
the default OCR engine for OCRmyPDF. Since EasyOCR is based on PyTorch, it makes
use of Nvidia GPUs. Hopefully it will be more performant and accurate than Tesseract OCR.

It is currently experimental and does not implement all of the features of
OCRmyPDF with Tesseract, and still relies on Tesseract for certain operations.

## Requirements

- **OCRmyPDF 17 or newer.** As of 0.3.0, this plugin returns structured OCR
  results to OCRmyPDF, which renders the text layer itself using its built-in
  renderer. (For OCRmyPDF 14–16, use a `< 0.3.0` release of this plugin.)
- Python 3.11 or newer.

## Installation

To use this plugin, first
[install PyTorch according to the official instructions](https://pytorch.org/),
which may differ for your platform.

Then install OCRmyPDF-EasyOCR into the same virtual environment or conda environment
as you installed PyTorch:

```bash
pip install git+https://github.com/ocrmypdf/OCRmyPDF-EasyOCR.git
```

Once installed, the plugin is discovered automatically through its entry point,
and OCRmyPDF uses EasyOCR for OCR by default (i.e. for `--ocr-engine auto`). You
can also select it explicitly with `--ocr-engine easyocr`, or fall back to the
built-in engines with `--ocr-engine tesseract` / `--ocr-engine none`. OCR still
depends on Tesseract for some non-OCR tasks (see below).

## Development

This project uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv sync --group test   # create the virtual environment and install dependencies
uv run pytest          # run the test suite
```

The tests need poppler's `pdftotext` and a Tesseract installation available on
`PATH`, and will download EasyOCR's models on first run.

## Troubleshooting

If you see a log message
``Neither CUDA nor MPS are available - defaulting to CPU. Note: This module is much faster with a GPU``
then PyTorch is not installed with GPU support.

Pass `--easyocr-no-gpu` to force CPU mode, or `--easyocr-batch-size` to tune the
EasyOCR batch size.

## To do

Contributions, especially pull requests, are quite welcome!

This plugin is experimental and is missing some features:
- Tesseract is still required to determine page orientation.
- EasyOCR is effectively single-threaded, to eliminate race conditions; OCRmyPDF
  still parallelizes other stages of the pipeline across pages.
