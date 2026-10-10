#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import cv2
import numpy as np

try:
    import pytesseract
    from pytesseract.pytesseract import TesseractNotFoundError
except ImportError as exc:
    raise SystemExit(
        "Missing Python dependency 'pytesseract'. Install it with: pip install pytesseract"
    ) from exc


SEG_PATTERNS = [
    re.compile(r"\bSeg\s*([-+]?\d*\.?\d+)\s*ms?\b", re.IGNORECASE),
    re.compile(r"\bSeg\s*[:=]\s*([-+]?\d*\.?\d+)\b", re.IGNORECASE),
    re.compile(r"\bSeg\s*([-+]?\d*\.?\d+)\b", re.IGNORECASE),
    re.compile(r"\b([-+]?\d*\.?\d+)\s*ms\b", re.IGNORECASE),
]


def preprocess_variants(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"failed to read image: {image_path}")

    h, w = img.shape[:2]
    # Telemetry text is consistently rendered in the lower-left corner.
    roi = img[int(h * 0.78): h, 0: int(w * 0.45)]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_gray = cv2.resize(roi_gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
    roi_gray = cv2.GaussianBlur(roi_gray, (3, 3), 0)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    roi_clahe = clahe.apply(roi_gray)

    _, thresh = cv2.threshold(roi_clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_inv = cv2.threshold(roi_clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        roi_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11
    )
    adaptive_inv = cv2.adaptiveThreshold(
        roi_clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )

    return [
        ("roi_gray", roi_gray),
        ("roi_clahe", roi_clahe),
        ("thresh", thresh),
        ("thresh_inv", thresh_inv),
        ("adaptive", adaptive),
        ("adaptive_inv", adaptive_inv),
    ]


def extract_seg_from_text(text: str):
    for pattern in SEG_PATTERNS:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def extract_seg(image_path: Path):
    variants = preprocess_variants(image_path)
    configs = [
        "--psm 7 -c tessedit_char_whitelist=Seg0123456789ms:.= ",
        "--psm 6 -c tessedit_char_whitelist=Seg0123456789ms:.= ",
        "--psm 13 -c tessedit_char_whitelist=Seg0123456789ms:.= ",
    ]
    texts = []
    for variant_name, proc in variants:
        for config in configs:
            try:
                text = pytesseract.image_to_string(proc, config=config)
            except TesseractNotFoundError as exc:
                raise SystemExit(
                    "Missing system dependency 'tesseract'. Install it with: sudo apt install tesseract-ocr"
                ) from exc
            texts.append(f"[{variant_name} | {config}] {text.strip()}")
            value = extract_seg_from_text(text)
            if value is not None:
                return value, "\n".join(texts)
    return None, "\n".join(texts)


def iter_images(path: Path):
    if path.is_file():
        yield path
        return
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for p in sorted(path.rglob("*")):
        if p.suffix.lower() in exts:
            yield p


def main():
    parser = argparse.ArgumentParser(description="Extract Seg value from images using OCR")
    parser.add_argument("path", help="Image file or directory")
    parser.add_argument("--show-text", action="store_true", help="Print OCR text too")
    args = parser.parse_args()

    target = Path(args.path)
    if not target.exists():
        raise SystemExit(f"path not found: {target}")

    found_any = False
    values = []
    for image_path in iter_images(target):
        value, text = extract_seg(image_path)
        found_any = True
        print(f"{image_path}: Seg={value if value is not None else 'NOT_FOUND'}")
        if value is not None:
            values.append(float(value))
        if args.show_text:
            print("----- OCR -----")
            print(text.strip())
            print("---------------")

    if not found_any:
        raise SystemExit("no images found")

    if values:
        avg = sum(values) / len(values)
        print(f"\nDetected values: {len(values)}")
        print(f"Average Seg: {avg:.3f}")
    else:
        print("\nDetected values: 0")
        print("Average Seg: N/A")


if __name__ == "__main__":
    main()
