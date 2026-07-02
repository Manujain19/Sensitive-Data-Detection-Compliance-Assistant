from __future__ import annotations

import csv
import io
import os
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LoadedDocument:
    file_name: str
    file_type: str
    text: str
    metadata: dict[str, str | int]


def load_document(file_name: str, data: bytes) -> LoadedDocument:
    suffix = Path(file_name).suffix.lower().lstrip(".")
    if suffix == "pdf":
        text, metadata = _load_pdf(data)
    elif suffix == "csv":
        text, metadata = _load_csv(data)
    elif suffix == "txt":
        text = _decode_text(data)
        metadata = {"characters": len(text)}
    else:
        raise ValueError("Unsupported file type. Please upload PDF, TXT, or CSV.")

    return LoadedDocument(
        file_name=file_name,
        file_type=suffix.upper(),
        text=text.strip(),
        metadata=metadata,
    )


def _load_pdf(data: bytes) -> tuple[str, dict[str, str | int]]:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="ARC4 has been moved",
                category=Warning,
            )
            from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF support requires the pypdf package.") from exc

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for index, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        pages.append(f"\n--- Page {index} ---\n{page_text}")

    text = "\n".join(pages)
    metadata: dict[str, str | int] = {
        "pages": len(reader.pages),
        "characters": len(text),
        "extraction_method": "embedded_text",
    }

    if len(text.strip()) < 50 and reader.pages:
        ocr_text, ocr_status = _ocr_pdf(data)
        metadata["ocr_status"] = ocr_status
        if ocr_text.strip():
            text = ocr_text
            metadata["characters"] = len(text)
            metadata["extraction_method"] = "ocr"
    else:
        metadata["ocr_status"] = "not_needed"

    return text, metadata


def _ocr_pdf(data: bytes) -> tuple[str, str]:
    try:
        import pytesseract
    except ImportError:
        return "", "ocr_dependencies_missing"

    tesseract_cmd = _tesseract_command()
    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    try:
        return _ocr_pdf_with_pdfium(data, pytesseract)
    except Exception as pdfium_exc:
        try:
            return _ocr_pdf_with_pymupdf(data, pytesseract)
        except Exception as pymupdf_exc:
            return "", f"ocr_failed: pdfium={pdfium_exc}; pymupdf={pymupdf_exc}"


def _ocr_pdf_with_pdfium(data: bytes, pytesseract) -> tuple[str, str]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    pages = []
    for index, page in enumerate(pdf, start=1):
        bitmap = page.render(scale=2).to_pil()
        page_text = pytesseract.image_to_string(bitmap)
        pages.append(f"\n--- OCR Page {index} ---\n{page_text}")
    return "\n".join(pages), "completed"


def _ocr_pdf_with_pymupdf(data: bytes, pytesseract) -> tuple[str, str]:
    import fitz
    from PIL import Image

    try:
        pdf = fitz.open(stream=data, filetype="pdf")
        pages = []
        for index, page in enumerate(pdf, start=1):
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            page_text = pytesseract.image_to_string(image)
            pages.append(f"\n--- OCR Page {index} ---\n{page_text}")
        return "\n".join(pages), "completed"
    except Exception:
        raise


def _tesseract_command() -> str:
    configured = os.getenv("TESSERACT_CMD", "").strip()
    candidates = [
        configured,
        shutil.which("tesseract") or "",
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    return ""


def _load_csv(data: bytes) -> tuple[str, dict[str, str | int]]:
    decoded = _decode_text(data)
    sample = decoded[:2048]
    try:
        dialect = csv.Sniffer().sniff(sample) if sample else csv.excel
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(io.StringIO(decoded), dialect))
    if not rows:
        return "", {"rows": 0, "columns": 0, "characters": 0}

    header = rows[0]
    text_lines = []
    for row_number, row in enumerate(rows[1:], start=1):
        pairs = []
        for index, value in enumerate(row):
            column = header[index] if index < len(header) and header[index] else f"column_{index + 1}"
            pairs.append(f"{column}: {value}")
        text_lines.append(f"Row {row_number}: " + "; ".join(pairs))

    if len(rows) == 1:
        text_lines.append("; ".join(header))

    text = "\n".join(text_lines)
    return text, {"rows": max(0, len(rows) - 1), "columns": len(header), "characters": len(text)}


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")
