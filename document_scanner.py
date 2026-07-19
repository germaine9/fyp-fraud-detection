# document_scanner.py
# OCR-assisted healthcare claim scanner for Streamlit.
# Supports JPG/JPEG/PNG, handwriting-aware vision extraction, and a strengthened
# Tesseract fallback for the supplied MediLife claim form.

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
from datetime import date, datetime
from typing import Any

import pandas as pd
import pytesseract
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Load values from a local .env file when python-dotenv is installed.
# This fixes the common situation where ANTHROPIC_API_KEY exists in .env but was
# never loaded into os.environ.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# -----------------------------------------------------------------------------
# Tesseract configuration
# -----------------------------------------------------------------------------

_tesseract_env = os.environ.get("TESSERACT_CMD", "").strip()
_windows_default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if _tesseract_env and os.path.exists(_tesseract_env):
    pytesseract.pytesseract.tesseract_cmd = _tesseract_env
elif os.path.exists(_windows_default):
    pytesseract.pytesseract.tesseract_cmd = _windows_default


FIELD_NAMES = [
    "Claim_ID",
    "Provider_ID",
    "Claim_Amount",
    "Approved_Amount",
    "Diagnosis_Code",
    "Procedure_Code",
    "Patient_Age",
    "Insurance_Type",
    "Patient_Gender",
    "Claim_Status",
    "Days_Between_Service_and_Claim",
    "Number_of_Claims_Per_Provider_Monthly",
    "Length_of_Stay",
    "Prior_Visits_12m",
    "Chronic_Condition_Flag",
    "Provider_Specialty",
    "Patient_State",
    "Visit_Type",
]

DEFAULTS = {
    "Claim_ID": "CLM-OCR",
    "Provider_ID": "PRV-UNKNOWN",
    "Claim_Amount": 5000.0,
    "Approved_Amount": 4500.0,
    "Diagnosis_Code": "D001",
    "Procedure_Code": "P001",
    "Patient_Age": 45,
    "Insurance_Type": "Private",
    "Patient_Gender": "Male",
    "Claim_Status": "Pending",
    "Days_Between_Service_and_Claim": 5,
    "Number_of_Claims_Per_Provider_Monthly": 10,
    "Length_of_Stay": 3,
    "Prior_Visits_12m": 2,
    "Chronic_Condition_Flag": 0,
    "Provider_Specialty": "General Practice",
    "Patient_State": "CA",
    "Visit_Type": "Outpatient",
}


# -----------------------------------------------------------------------------
# Vision extraction
# -----------------------------------------------------------------------------

CLAUDE_VISION_PROMPT = r"""
You are extracting structured fields from a healthcare claim form image.
The form may contain printed labels, handwritten values, ticks, crossed boxes,
and closely spaced numbers. Read the image carefully twice before answering.

Return one valid JSON object with EXACTLY these keys. Use null when a value is
not present or cannot be read reliably.

{
  "Claim_ID": null,
  "Provider_ID": null,
  "Claim_Amount": null,
  "Approved_Amount": null,
  "Diagnosis_Code": null,
  "Procedure_Code": null,
  "Patient_Age": null,
  "Insurance_Type": null,
  "Patient_Gender": null,
  "Claim_Status": null,
  "Days_Between_Service_and_Claim": null,
  "Number_of_Claims_Per_Provider_Monthly": null,
  "Length_of_Stay": null,
  "Prior_Visits_12m": null,
  "Chronic_Condition_Flag": null,
  "Provider_Specialty": null,
  "Patient_State": null,
  "Visit_Type": null
}

Extraction rules:
1. Claim_Amount means Total Claim Amount, not Total Approved Amount.
2. Approved_Amount means Total Approved Amount.
3. Return monetary values as numbers without currency symbols or commas.
4. Read handwritten digits exactly. Distinguish 1/7, 0/6, 2/7, 5/S and 8/B.
5. Prior_Visits_12m and Number_of_Claims_Per_Provider_Monthly are different fields.
6. If Patient_Age is absent but Date of Birth and a service/admission date are
   present, calculate age on the service/admission date.
7. If Days_Between_Service_and_Claim is absent, calculate it from the claim
   submitted date and service/admission date. Do not use a negative result.
8. If Length_of_Stay is absent but admission and discharge dates are present,
   calculate the inclusive stay: discharge - admission + 1 day.
9. If both Inpatient and Emergency are ticked, return Emergency.
10. Insurance_Type must be one of Private, Government, Medicaid, Self-Pay.
11. Patient_Gender must be Male or Female.
12. Claim_Status must be Approved, Pending or Rejected. If the claim is merely
    submitted and no decision is shown, return Pending.
13. Chronic_Condition_Flag must be 1 for Yes and 0 for No.
14. Visit_Type must be Inpatient, Outpatient or Emergency.
15. Do not invent a Claim_ID or Provider_ID. A policy number may be used as
    Claim_ID only when the form has no separate populated claim number.
16. Output JSON only. Do not add markdown or explanation.
"""


def _open_image(image_file: Any) -> Image.Image:
    """Open an uploaded file or copy an existing PIL image safely."""
    if isinstance(image_file, Image.Image):
        return ImageOps.exif_transpose(image_file.copy()).convert("RGB")

    image_file.seek(0)
    image = Image.open(image_file)
    image = ImageOps.exif_transpose(image).convert("RGB")
    image_file.seek(0)
    return image


def image_to_base64(image_file: Any) -> tuple[str, str]:
    """Convert an uploaded image or PIL image to base64 plus media type."""
    if isinstance(image_file, Image.Image):
        buffer = io.BytesIO()
        image_file.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8"), "image/png"

    image_file.seek(0)
    raw = image_file.read()
    image_file.seek(0)
    extension = os.path.splitext(getattr(image_file, "name", "image.png"))[1].lower()
    media_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }.get(extension, "image/png")
    return base64.b64encode(raw).decode("utf-8"), media_type


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    """Extract the first JSON object even if the model adds code fences."""
    cleaned = re.sub(r"```(?:json)?", "", raw_text, flags=re.IGNORECASE).replace("```", "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError("No JSON object was returned by the vision model.")
    parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Vision response was not a JSON object.")
    return parsed


def extract_fields_via_claude_vision(image_file: Any) -> tuple[dict[str, Any] | None, str | None]:
    """Use Claude Vision as the primary handwriting extractor."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return None, "ANTHROPIC_API_KEY is not configured; Tesseract fallback was used."

    model_name = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514").strip()

    try:
        encoded_image, media_type = image_to_base64(image_file)
        payload = {
            "model": model_name,
            "max_tokens": 1200,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": encoded_image,
                            },
                        },
                        {"type": "text", "text": CLAUDE_VISION_PROMPT},
                    ],
                }
            ],
        }

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=60,
        )

        if response.status_code != 200:
            detail = response.text[:500].replace("\n", " ")
            return None, f"Vision API returned HTTP {response.status_code}: {detail}"

        response_data = response.json()
        raw_text = "".join(
            block.get("text", "")
            for block in response_data.get("content", [])
            if block.get("type") == "text"
        )
        return _extract_json_object(raw_text), None

    except requests.exceptions.Timeout:
        return None, "Vision API request timed out; Tesseract fallback was used."
    except requests.exceptions.RequestException as exc:
        return None, f"Vision API request failed: {exc}"
    except (ValueError, json.JSONDecodeError, KeyError, TypeError) as exc:
        return None, f"Vision response could not be parsed: {exc}"


# -----------------------------------------------------------------------------
# Image preprocessing and Tesseract OCR
# -----------------------------------------------------------------------------


def _resize_for_ocr(image: Image.Image, target_long_side: int = 3000) -> Image.Image:
    width, height = image.size
    longest = max(width, height)
    if longest <= 0:
        return image

    scale = min(3.0, max(1.0, target_long_side / float(longest)))
    if scale <= 1.01:
        return image
    return image.resize(
        (int(round(width * scale)), int(round(height * scale))),
        Image.Resampling.LANCZOS,
    )


def build_ocr_variants(image: Image.Image) -> list[Image.Image]:
    """Create complementary OCR variants for printed and handwritten content."""
    rgb = _resize_for_ocr(image.convert("RGB"))
    gray = ImageOps.autocontrast(rgb.convert("L"), cutoff=1)

    contrast = ImageEnhance.Contrast(gray).enhance(1.55)
    contrast = contrast.filter(ImageFilter.UnsharpMask(radius=1.4, percent=190, threshold=2))

    softer = ImageEnhance.Contrast(gray).enhance(1.2)
    softer = softer.filter(ImageFilter.SHARPEN)

    variants = [contrast, softer]

    # OpenCV thresholding is optional. The rest of the scanner still works
    # without opencv-python.
    try:
        import cv2
        import numpy as np

        array = np.array(gray)
        otsu = cv2.threshold(array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        adaptive = cv2.adaptiveThreshold(
            array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            41,
            11,
        )
        variants.extend([Image.fromarray(otsu), Image.fromarray(adaptive)])
    except ImportError:
        pass

    return variants


def _check_tesseract() -> str | None:
    try:
        pytesseract.get_tesseract_version()
        return None
    except Exception as exc:
        return str(exc)


def extract_text_from_image_tesseract(image_file: Any) -> str:
    """
    Run several OCR layouts and preprocessing variants, then combine unique
    outputs. Combining passes is more reliable than selecting only the longest
    pass because different passes recognise different handwritten fields.
    """
    error = _check_tesseract()
    if error:
        return f"ERROR: Tesseract is unavailable: {error}"

    try:
        image = _open_image(image_file)
        outputs: list[str] = []
        seen: set[str] = set()

        # Two complementary passes are normally sufficient and keep the app
        # responsive. PSM 11 recognises sparse form fields; PSM 4 preserves
        # two-column layout better.
        primary_variant = build_ocr_variants(image)[0]
        for psm in (11, 4):
            config = f"--psm {psm} --oem 3 -l eng"
            text = pytesseract.image_to_string(primary_variant, config=config).strip()
            normalised_key = re.sub(r"\s+", " ", text).strip().lower()
            if text and normalised_key not in seen:
                seen.add(normalised_key)
                outputs.append(text)

        if not outputs:
            return ""

        return "\n\n===== OCR PASS =====\n\n".join(outputs)
    except Exception as exc:
        return f"ERROR: {exc}"


# -----------------------------------------------------------------------------
# OCR parsing helpers
# -----------------------------------------------------------------------------


def _empty_fields() -> dict[str, Any]:
    return {field: None for field in FIELD_NAMES}


def _normalise_spacing(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = text.replace("–", "-").replace("—", "-")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def _label_window(text: str, label_patterns: list[str], size: int = 180) -> str:
    for pattern in label_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return text[match.end() : match.end() + size]
    return ""


def _first_number(window: str, *, money: bool = False, max_value: int | None = None) -> float | int | None:
    """Read the first plausible OCR number from a short field window."""
    if not window:
        return None

    # Prevent the parser from drifting into the next labelled field.
    window = re.split(
        r"\b(?:date|number|length|currency|chronic|name|section|total|hospital|department|reason|icd|gender|occupation|contact|email|residential)\b",
        window,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    candidates = re.findall(r"(?<![A-Za-z])[/|IlO0-9][0-9IlO/|,\. ]{0,18}", window)
    for candidate in candidates:
        token = candidate.strip()
        token = re.sub(r"\s+", "", token)

        # Common handwriting/OCR substitutions, applied only inside a numeric token.
        token = token.replace("O", "0").replace("o", "0")
        token = token.replace("I", "1").replace("l", "1").replace("|", "1")
        if token.startswith("/") and len(token) > 1 and token[1].isdigit():
            token = "1" + token[1:]

        if money:
            token = token.replace(",", "")
            # Printed forms often show a separate trailing .00. Keep the main amount.
            match = re.match(r"(\d+(?:\.\d{1,2})?)", token)
            if not match:
                continue
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if value >= 0:
                return value
        else:
            digits = re.sub(r"\D", "", token)
            if not digits:
                continue
            value = int(digits)
            if max_value is not None and value > max_value:
                continue
            return value

    return None


def _extract_date_after_label(text: str, labels: list[str]) -> date | None:
    window = _label_window(text, labels, size=120)
    if not window:
        return None

    # Correct a few OCR confusions only in the small date window.
    fixed = window.replace("O", "0").replace("o", "0")
    fixed = fixed.replace("I", "1").replace("l", "1").replace("|", "/")
    fixed = fixed.replace("[", "/").replace("]", "/")

    match = re.search(r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{2,4})", fixed)
    if not match:
        # Handles output such as 06/052024 where one slash is omitted.
        match = re.search(r"(\d{1,2})\s*[/.-]\s*(\d{2})(\d{4})", fixed)
    if not match:
        return None

    try:
        day = int(match.group(1))
        month = int(match.group(2))
        year = int(match.group(3))
        if year < 100:
            year += 2000 if year < 50 else 1900
        return date(year, month, day)
    except (ValueError, TypeError):
        return None


def _calculate_age(dob: date | None, reference: date | None) -> int | None:
    if dob is None or reference is None or reference < dob:
        return None
    age = reference.year - dob.year - ((reference.month, reference.day) < (dob.month, dob.day))
    return age if 0 <= age <= 120 else None


def _extract_code_after_label(text: str, labels: list[str], pattern: str) -> str | None:
    window = _label_window(text, labels, size=100)
    match = re.search(pattern, window, flags=re.IGNORECASE)
    return match.group(1).upper().strip() if match else None


def _parse_money(text: str, labels: list[str]) -> float | None:
    return _first_number(_label_window(text, labels, size=100), money=True)


def _parse_integer(text: str, labels: list[str], max_value: int | None = None) -> int | None:
    value = _first_number(_label_window(text, labels, size=130), max_value=max_value)
    return int(value) if value is not None else None


def _blue_ink_score(image: Image.Image, box: tuple[float, float, float, float]) -> int:
    """Count blue-ink pixels inside a normalised checkbox region."""
    try:
        import numpy as np

        width, height = image.size
        left = max(0, int(box[0] * width))
        top = max(0, int(box[1] * height))
        right = min(width, int(box[2] * width))
        bottom = min(height, int(box[3] * height))
        crop = np.asarray(image.crop((left, top, right, bottom)).convert("RGB"), dtype=np.int16)
        red = crop[:, :, 0]
        green = crop[:, :, 1]
        blue = crop[:, :, 2]
        mask = (blue > 55) & (blue > red + 18) & (blue > green + 8)
        return int(mask.sum())
    except Exception:
        return 0


def _extract_medilife_checkbox_fields(image: Image.Image, text: str) -> dict[str, Any]:
    """Read checkbox fields on the supplied portrait MediLife template."""
    if not re.search(r"healthcare\s+claim\s+form", text, flags=re.IGNORECASE):
        return {}

    result: dict[str, Any] = {}

    male_score = _blue_ink_score(image, (0.348, 0.257, 0.389, 0.292))
    female_score = _blue_ink_score(image, (0.411, 0.257, 0.452, 0.292))
    if max(male_score, female_score) >= 3:
        result["Patient_Gender"] = "Male" if male_score > female_score else "Female"

    outpatient_score = _blue_ink_score(image, (0.626, 0.220, 0.676, 0.258))
    inpatient_score = _blue_ink_score(image, (0.738, 0.220, 0.789, 0.258))
    emergency_score = _blue_ink_score(image, (0.839, 0.220, 0.892, 0.258))
    if max(outpatient_score, inpatient_score, emergency_score) >= 3:
        # When Emergency and Inpatient are both checked, Emergency takes priority.
        if emergency_score >= 3:
            result["Visit_Type"] = "Emergency"
        elif inpatient_score >= 3:
            result["Visit_Type"] = "Inpatient"
        else:
            result["Visit_Type"] = "Outpatient"

    chronic_yes_score = _blue_ink_score(image, (0.657, 0.599, 0.707, 0.637))
    chronic_no_score = _blue_ink_score(image, (0.756, 0.599, 0.806, 0.637))
    if max(chronic_yes_score, chronic_no_score) >= 3:
        result["Chronic_Condition_Flag"] = 1 if chronic_yes_score > chronic_no_score else 0

    return result


def _ocr_region(
    image: Image.Image,
    box: tuple[float, float, float, float],
    whitelist: str | None = None,
    psm: int = 7,
) -> str:
    width, height = image.size
    crop_box = (
        int(box[0] * width),
        int(box[1] * height),
        int(box[2] * width),
        int(box[3] * height),
    )
    crop = image.crop(crop_box).convert("L")
    crop = ImageOps.autocontrast(crop, cutoff=1)
    crop = crop.resize((crop.width * 4, crop.height * 4), Image.Resampling.LANCZOS)
    crop = ImageEnhance.Contrast(crop).enhance(1.45)
    crop = crop.filter(ImageFilter.UnsharpMask(radius=1, percent=170, threshold=2))

    config = f"--psm {psm} --oem 3 -l eng"
    if whitelist:
        config += f" -c tessedit_char_whitelist={whitelist}"
    return pytesseract.image_to_string(crop, config=config).strip()


def _extract_medilife_targeted_fields(
    image: Image.Image,
    text: str,
    existing_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Targeted OCR for the supplied MediLife portrait template. Values are only
    used to fill fields missed by full-page OCR, except for highly reliable
    numeric regions where they may correct a malformed OCR token.
    """
    if not re.search(r"healthcare\s+claim\s+form", text, flags=re.IGNORECASE):
        return {}

    result: dict[str, Any] = {}
    existing_fields = existing_fields or {}

    regions = {
        "Claim_Amount": ((0.250, 0.515, 0.420, 0.545), "0123456789,./-"),
        "Approved_Amount": ((0.250, 0.548, 0.420, 0.578), "0123456789,./-"),
        "Prior_Visits_12m": ((0.720, 0.540, 0.920, 0.585), "0123456789"),
        "Length_of_Stay": ((0.680, 0.578, 0.800, 0.608), "0123456789"),
        "Number_of_Claims_Per_Provider_Monthly": ((0.790, 0.675, 0.900, 0.705), "0123456789"),
        "Diagnosis_Code": ((0.650, 0.438, 0.800, 0.470), "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789."),
    }

    for field, (box, whitelist) in regions.items():
        if existing_fields.get(field) is not None:
            continue
        region_text = _ocr_region(image, box, whitelist=whitelist)

        if field in {"Claim_Amount", "Approved_Amount"}:
            value = _first_number(region_text, money=True)
            if value is not None:
                result[field] = float(value)
        elif field == "Diagnosis_Code":
            code_match = re.search(r"([A-Z][0-9]{2}(?:\.[0-9]{1,4})?)", region_text.upper())
            if code_match:
                result[field] = code_match.group(1)
        else:
            value = _first_number(region_text, max_value=100000)
            if value is not None:
                result[field] = int(value)

    return result


def parse_claim_fields_from_text(text: str, image: Image.Image | None = None) -> dict[str, Any]:
    """Parse healthcare claim fields from multi-pass OCR output."""
    fields = _empty_fields()
    text = _normalise_spacing(text)

    # IDs. Require at least one digit so that a blank "Claim No." field does
    # not accidentally capture a nearby heading such as INSURANCE.
    def _best_identifier(label_pattern: str) -> str | None:
        candidates: list[tuple[int, str]] = []
        for match in re.finditer(label_pattern, text, flags=re.IGNORECASE):
            window = text[match.end() : match.end() + 70]
            first_line = next((line.strip() for line in window.splitlines() if line.strip()), "")
            token_match = re.search(
                r"\b("
                r"[A-Z]{1,8}\s*-\s*[A-Z0-9#]{2,}(?:\s*-\s*[A-Z0-9#]{1,})*"
                r"|[A-Z]{1,8}[A-Z0-9#]{3,}"
                r"|\d{4,}(?:-\d+)*"
                r")\b",
                first_line,
                flags=re.IGNORECASE,
            )
            if not token_match:
                continue
            token = token_match.group(1).strip().upper()
            if not re.search(r"\d", token):
                continue
            cleaned = re.sub(r"\s+", "", token)
            cleaned = re.sub(r"[^A-Z0-9#-]", "", cleaned)
            score = len(re.findall(r"\d", cleaned)) * 4 - cleaned.count("#") * 3
            candidates.append((score, cleaned))
        return max(candidates, default=(0, None), key=lambda item: item[0])[1]

    fields["Claim_ID"] = _best_identifier(r"claim\s*(?:id|no\.?|number|#)\s*[:#-]?")
    if fields["Claim_ID"] is None:
        fields["Claim_ID"] = _best_identifier(r"policy\s*(?:id|no\.?|number|#)\s*[:#-]?")

    fields["Provider_ID"] = _best_identifier(r"provider\s*(?:id|no\.?|number|#)\s*[:#-]?")

    fields["Claim_Amount"] = _parse_money(
        text,
        [
            r"total\s+claim\s+amount(?:\s*\([^)]*\))?\s*[:]?",
            r"claim\s+amount(?:\s*\([^)]*\))?\s*[:]?",
        ],
    )
    fields["Approved_Amount"] = _parse_money(
        text,
        [
            r"total\s+approved\s+amount(?:\s*\([^)]*\))?\s*[:]?",
            r"approved\s+amount(?:\s*\([^)]*\))?\s*[:]?",
            r"paid\s+amount\s*[:]?...",
        ],
    )

    fields["Diagnosis_Code"] = _extract_code_after_label(
        text,
        [r"icd\s+code(?:\s*\([^)]*\))?\s*[:]?", r"diagnosis\s+code\s*[:]?"],
        r"([A-Z][0-9]{2}(?:\.[0-9]{1,4})?)",
    )
    fields["Procedure_Code"] = _extract_code_after_label(
        text,
        [r"procedure\s+(?:code|cpt)\s*[:]?", r"cpt\s+code\s*[:]?"],
        r"([A-Z0-9]{4,8})",
    )

    admission_date = _extract_date_after_label(text, [r"date\s+of\s+admission\s*[:]?"])
    discharge_date = _extract_date_after_label(text, [r"date\s+of\s+discharge\s*[:]?"])
    claim_submitted_date = _extract_date_after_label(
        text,
        [r"date\s+claim\s+submitted\s*[:]?", r"claim\s+submitted\s+date\s*[:]?"],
    )
    dob = _extract_date_after_label(text, [r"date\s+of\s+birth\s*[:]?", r"dob\s*[:]?" ])

    explicit_age = _parse_integer(text, [r"(?:patient\s+)?age\s*[:]?"], max_value=120)
    fields["Patient_Age"] = explicit_age or _calculate_age(
        dob,
        admission_date or claim_submitted_date or discharge_date,
    )

    insurance_match = re.search(
        r"insurance\s*(?:type)?\s*[:]?\s*(private|government|medicare|medicaid|self[- ]?pay)",
        text,
        flags=re.IGNORECASE,
    )
    if insurance_match:
        insurance = insurance_match.group(1).lower().replace(" ", "-")
        if insurance == "medicare":
            insurance = "government"
        fields["Insurance_Type"] = {
            "private": "Private",
            "government": "Government",
            "medicaid": "Medicaid",
            "self-pay": "Self-Pay",
        }.get(insurance)
    elif re.search(r"medilife\s+insurance", text, flags=re.IGNORECASE):
        # This is an explicit private insurer form, not an inference from a random document.
        fields["Insurance_Type"] = "Private"

    status_match = re.search(
        r"(?:claim\s+)?status\s*[:]?\s*(approved|pending|rejected|denied)",
        text,
        flags=re.IGNORECASE,
    )
    if status_match:
        fields["Claim_Status"] = "Rejected" if status_match.group(1).lower() == "denied" else status_match.group(1).capitalize()
    elif claim_submitted_date is not None:
        fields["Claim_Status"] = "Pending"

    fields["Prior_Visits_12m"] = _parse_integer(
        text,
        [
            r"number\s+of\s+prior\s+visits(?:\s*\([^)]*12\s*months?[^)]*\))?\s*[:]?",
            r"prior\s+visits(?:\s+in\s+12\s+months?)?\s*[:]?",
        ],
        max_value=1000,
    )

    fields["Length_of_Stay"] = _parse_integer(
        text,
        [r"length\s+of\s+stay(?:\s*\([^)]*\))?\s*[:]?"],
        max_value=1000,
    )
    if fields["Length_of_Stay"] is None and admission_date and discharge_date and discharge_date >= admission_date:
        fields["Length_of_Stay"] = (discharge_date - admission_date).days + 1

    fields["Number_of_Claims_Per_Provider_Monthly"] = _parse_integer(
        text,
        [
            r"number\s+of\s+claims\s+from\s+this\s+provider\s*\(monthly\)\s*[:]?",
            r"claims?\s+per\s+provider\s+monthly\s*[:]?",
            r"claims?\s+from\s+this\s+provider\s*\(monthly\)\s*[:]?",
        ],
        max_value=100000,
    )

    direct_days = _parse_integer(
        text,
        [r"days?\s+between\s+service\s+and\s+claim\s*[:]?", r"claim\s+delay\s*[:]?"],
        max_value=10000,
    )
    fields["Days_Between_Service_and_Claim"] = direct_days
    if direct_days is None and claim_submitted_date is not None:
        # Prefer discharge if the claim was submitted on/after discharge. Otherwise
        # use admission/service date so that an inconsistent form does not create a
        # negative delay.
        reference_service_date = None
        if discharge_date and claim_submitted_date >= discharge_date:
            reference_service_date = discharge_date
        elif admission_date and claim_submitted_date >= admission_date:
            reference_service_date = admission_date
        if reference_service_date:
            fields["Days_Between_Service_and_Claim"] = (claim_submitted_date - reference_service_date).days

    # Text-only fallbacks for categories. Checkbox analysis below is stronger.
    if re.search(r"\bemergency\s+depar\s*tment\b", text, flags=re.IGNORECASE):
        fields["Provider_Specialty"] = "Emergency Medicine"
    else:
        specialty_terms = {
            "cardiology": "Cardiology",
            "orthopedics": "Orthopedics",
            "neurology": "Neurology",
            "oncology": "Oncology",
            "general practice": "General Practice",
            "radiology": "Radiology",
            "surgery": "Surgery",
            "psychiatry": "Psychiatry",
            "dermatology": "Dermatology",
            "pediatrics": "Pediatrics",
        }
        for term, normalised in specialty_terms.items():
            if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE):
                fields["Provider_Specialty"] = normalised
                break

    if re.search(r"\bsingapore\b", text, flags=re.IGNORECASE):
        fields["Patient_State"] = "SG"
    else:
        state_match = re.search(r"(?:patient\s+)?state\s*[:]?\s*([A-Z]{2})\b", text)
        if state_match:
            fields["Patient_State"] = state_match.group(1)

    # Basic text check for chronic condition. Template checkbox detection may override it.
    chronic_line = _label_window(text, [r"chronic\s+condition\s*[:]?"], size=80)
    yes_pos = re.search(r"\byes\b", chronic_line, flags=re.IGNORECASE)
    no_pos = re.search(r"\bno\b", chronic_line, flags=re.IGNORECASE)
    if yes_pos and not no_pos:
        fields["Chronic_Condition_Flag"] = 1
    elif no_pos and not yes_pos:
        fields["Chronic_Condition_Flag"] = 0

    if image is not None:
        targeted = _extract_medilife_targeted_fields(image, text, existing_fields=fields)
        for key, value in targeted.items():
            # Targeted template OCR is especially useful for these handwritten numerics.
            if value is not None:
                fields[key] = value

        checkbox_fields = _extract_medilife_checkbox_fields(image, text)
        fields.update(checkbox_fields)

    return fields


def extract_fields_via_tesseract(image_file: Any) -> tuple[dict[str, Any] | None, str, str | None]:
    """Run enhanced Tesseract OCR and parse fields."""
    image = _open_image(image_file)
    text = extract_text_from_image_tesseract(image)
    if text.startswith("ERROR:"):
        return None, "", text
    if not text.strip():
        return None, "", "Tesseract returned no readable text."
    fields = parse_claim_fields_from_text(text, image=image)
    return fields, text, None


# -----------------------------------------------------------------------------
# Field sanitation and merging
# -----------------------------------------------------------------------------


def _string_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return None if not text or text.lower() in {"null", "none", "n/a"} else text


def _float_or_none(value: Any) -> float | None:
    try:
        if value is None:
            return None
        cleaned = re.sub(r"[^0-9.\-]", "", str(value))
        return float(cleaned) if cleaned else None
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(round(float(str(value).replace(",", "").strip())))
    except (TypeError, ValueError):
        return None


def sanitise_extracted_fields(raw: dict[str, Any] | None) -> dict[str, Any]:
    raw = raw or {}
    out = _empty_fields()

    out["Claim_ID"] = _string_or_none(raw.get("Claim_ID"))
    out["Provider_ID"] = _string_or_none(raw.get("Provider_ID"))
    out["Claim_Amount"] = _float_or_none(raw.get("Claim_Amount"))
    out["Approved_Amount"] = _float_or_none(raw.get("Approved_Amount"))
    out["Diagnosis_Code"] = _string_or_none(raw.get("Diagnosis_Code"))
    out["Procedure_Code"] = _string_or_none(raw.get("Procedure_Code"))
    out["Patient_Age"] = _int_or_none(raw.get("Patient_Age"))
    out["Days_Between_Service_and_Claim"] = _int_or_none(raw.get("Days_Between_Service_and_Claim"))
    out["Number_of_Claims_Per_Provider_Monthly"] = _int_or_none(raw.get("Number_of_Claims_Per_Provider_Monthly"))
    out["Length_of_Stay"] = _int_or_none(raw.get("Length_of_Stay"))
    out["Prior_Visits_12m"] = _int_or_none(raw.get("Prior_Visits_12m"))
    out["Chronic_Condition_Flag"] = _int_or_none(raw.get("Chronic_Condition_Flag"))
    out["Provider_Specialty"] = _string_or_none(raw.get("Provider_Specialty"))
    out["Patient_State"] = _string_or_none(raw.get("Patient_State"))

    insurance = _string_or_none(raw.get("Insurance_Type"))
    if insurance:
        normalised = insurance.lower().replace("_", " ").replace("-", " ").strip()
        out["Insurance_Type"] = {
            "private": "Private",
            "government": "Government",
            "medicare": "Government",
            "medicaid": "Medicaid",
            "self pay": "Self-Pay",
        }.get(normalised)

    gender = _string_or_none(raw.get("Patient_Gender"))
    if gender:
        out["Patient_Gender"] = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}.get(gender.lower())

    status = _string_or_none(raw.get("Claim_Status"))
    if status:
        out["Claim_Status"] = {
            "approved": "Approved",
            "pending": "Pending",
            "rejected": "Rejected",
            "denied": "Rejected",
        }.get(status.lower())

    visit = _string_or_none(raw.get("Visit_Type"))
    if visit:
        out["Visit_Type"] = {
            "inpatient": "Inpatient",
            "outpatient": "Outpatient",
            "emergency": "Emergency",
        }.get(visit.lower())

    # Range validation prevents a bad OCR token from silently entering the model.
    if out["Patient_Age"] is not None and not 0 <= out["Patient_Age"] <= 120:
        out["Patient_Age"] = None
    for field in (
        "Days_Between_Service_and_Claim",
        "Number_of_Claims_Per_Provider_Monthly",
        "Length_of_Stay",
        "Prior_Visits_12m",
    ):
        if out[field] is not None and out[field] < 0:
            out[field] = None
    if out["Chronic_Condition_Flag"] not in (0, 1):
        out["Chronic_Condition_Flag"] = None

    return out


# Backward-compatible function name used by the previous file.
def sanitise_vision_fields(raw: dict[str, Any]) -> dict[str, Any]:
    return sanitise_extracted_fields(raw)


def merge_extraction_results(
    vision_fields: dict[str, Any] | None,
    tesseract_fields: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Merge both routes. Vision takes priority, but the Tesseract result fills any
    missing value. The source map is used by the review table.
    """
    vision = sanitise_extracted_fields(vision_fields)
    tess = sanitise_extracted_fields(tesseract_fields)

    merged = _empty_fields()
    source: dict[str, str] = {}
    for field in FIELD_NAMES:
        if vision.get(field) is not None:
            merged[field] = vision[field]
            source[field] = "Vision"
        elif tess.get(field) is not None:
            merged[field] = tess[field]
            source[field] = "Tesseract"
        else:
            merged[field] = None
            source[field] = "Default"
    return merged, source


def fill_defaults(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        field: fields.get(field) if fields.get(field) is not None else DEFAULTS[field]
        for field in FIELD_NAMES
    }


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------

SESSION_KEY = "ocr_filled_fields"


def _init_session_fields(filled: dict[str, Any]) -> None:
    st.session_state[SESSION_KEY] = filled.copy()


def _get_session_fields() -> dict[str, Any] | None:
    return st.session_state.get(SESSION_KEY)


# -----------------------------------------------------------------------------
# Fraud model preprocessing and prediction
# -----------------------------------------------------------------------------


def _replace_unseen_categories(df: pd.DataFrame, preprocess_info: dict[str, Any]) -> pd.DataFrame:
    """
    Replace OCR categories not present during training with the training mode.
    This avoids an all-zero one-hot category for values such as a new state or
    specialty while preserving the visible OCR value in the review form.
    """
    feature_columns = set(preprocess_info.get("feature_columns", []))
    categorical_modes = preprocess_info.get("categorical_modes", {})

    for column in preprocess_info.get("categorical_cols", []):
        if column not in df.columns:
            continue
        value = str(df.at[0, column])
        expected_dummy = f"{column}_{value}"
        if expected_dummy not in feature_columns and column in categorical_modes:
            df.at[0, column] = categorical_modes[column]
    return df


def preprocess_for_model(fields: dict[str, Any], scaler: Any, preprocess_info: dict[str, Any]):
    input_dict = {
        key: fields[key]
        for key in [
            "Patient_Age",
            "Patient_Gender",
            "Diagnosis_Code",
            "Procedure_Code",
            "Claim_Amount",
            "Approved_Amount",
            "Insurance_Type",
            "Days_Between_Service_and_Claim",
            "Number_of_Claims_Per_Provider_Monthly",
            "Provider_Specialty",
            "Patient_State",
            "Claim_Status",
            "Length_of_Stay",
            "Visit_Type",
            "Chronic_Condition_Flag",
            "Prior_Visits_12m",
        ]
    }
    df = pd.DataFrame([input_dict])

    for column in preprocess_info["numeric_cols"]:
        fallback = preprocess_info["numeric_means"][column]
        if column not in df.columns:
            df[column] = fallback
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(fallback)

    for column in preprocess_info["categorical_cols"]:
        fallback = preprocess_info["categorical_modes"][column]
        if column not in df.columns:
            df[column] = fallback
        df[column] = df[column].fillna(fallback).astype(str)

    df = _replace_unseen_categories(df, preprocess_info)

    df["claim_to_cost_ratio"] = df["Claim_Amount"] / (df["Approved_Amount"] + 1)
    df["cost_outlier_flag"] = (
        df["Claim_Amount"]
        > preprocess_info["claim_q3"] + 1.5 * preprocess_info["claim_iqr"]
    ).astype(int)
    df["high_claim_frequency"] = (
        df["Number_of_Claims_Per_Provider_Monthly"]
        > preprocess_info["high_claim_threshold"]
    ).astype(int)

    df = pd.get_dummies(df)
    df = df.reindex(columns=preprocess_info["feature_columns"], fill_value=0)
    return scaler.transform(df)


def predict_fraud_score(model: Any, X_scaled: Any) -> float:
    try:
        prediction = model.predict(X_scaled, verbose=0)
        return float(prediction.flatten()[0])
    except TypeError:
        pass

    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X_scaled)[0][1])
    return float(model.predict(X_scaled)[0])


# -----------------------------------------------------------------------------
# Blockchain helpers
# -----------------------------------------------------------------------------


def add_blockchain_record(bc: Any, claim_data: dict[str, Any], fraud_score: float):
    try:
        return bc.add_record(claim_data, fraud_score, source="OCR Scanner")
    except TypeError:
        return bc.add_record(claim_data, fraud_score)


def get_block_hash(block: Any) -> str:
    return getattr(block, "hash", getattr(block, "block_hash", "N/A"))


# -----------------------------------------------------------------------------
# Streamlit page
# -----------------------------------------------------------------------------


def render_document_scanner(model: Any, scaler: Any, bc: Any, preprocess_info: dict[str, Any] | None = None):
    st.title("OCR Claim Document Scanner")
    st.write(
        "Upload a healthcare claim document in JPG, JPEG, or PNG format. "
        "The scanner uses handwriting-aware vision extraction when configured, "
        "then validates and supplements it with multi-pass Tesseract OCR."
    )
    st.markdown(
        """
        <div style="
            font-size: 0.80rem;
            line-height: 1.35;
            color: #6f6258;
            background: #fff7ed;
            border-left: 3px solid #c56a12;
            border-radius: 5px;
            padding: 0.45rem 0.65rem;
            margin: 0.15rem 0 0.35rem 0;
        ">
            <strong>OCR tip:</strong> Upload a clear, well-lit and properly aligned JPG or PNG image.
            Blur, glare, shadows, cropping or unclear handwriting can reduce accuracy.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Review every extracted value before running fraud detection.")
    st.divider()

    if preprocess_info is None:
        try:
            import joblib

            preprocess_info = joblib.load("preprocess_info.pkl")
        except Exception:
            st.error("preprocess_info.pkl is missing. Please run train_ann_fixed.py first.")
            return

    uploaded_file = st.file_uploader(
        "Upload claim document",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear printed or handwritten claim form in JPG, JPEG, or PNG format.",
    )

    if uploaded_file is None:
        for key in (
            SESSION_KEY,
            "ocr_file_id",
            "ocr_raw_fields",
            "ocr_field_sources",
            "ocr_extracted_text",
            "ocr_extraction_method",
            "ocr_vision_warning",
        ):
            st.session_state.pop(key, None)
        st.info("Upload a claim document to begin.")
        return

    allowed_extensions = {".jpg", ".jpeg", ".png"}
    file_extension = os.path.splitext(uploaded_file.name)[1].lower()
    if file_extension not in allowed_extensions:
        st.error("Unsupported file type. Please upload a JPG, JPEG, or PNG image.")
        return

    st.success(f"Uploaded: **{uploaded_file.name}**")

    try:
        uploaded_file.seek(0)
        preview_image = _open_image(uploaded_file)
        uploaded_file.seek(0)
    except Exception as exc:
        st.error(f"The uploaded file could not be opened as an image: {exc}")
        return

    st.subheader("Document Preview")
    st.image(preview_image, caption=uploaded_file.name, use_container_width=True)

    st.subheader("Step 1: Text & Field Extraction")
    uploaded_file.seek(0)
    file_bytes = uploaded_file.read()
    uploaded_file.seek(0)
    file_id = hashlib.sha256(file_bytes).hexdigest()

    if st.session_state.get("ocr_file_id") != file_id:
        with st.spinner("Reading printed text and handwriting..."):
            uploaded_file.seek(0)
            vision_result, vision_warning = extract_fields_via_claude_vision(uploaded_file)

            uploaded_file.seek(0)
            tesseract_result, extracted_text, tesseract_error = extract_fields_via_tesseract(uploaded_file)

        if vision_result is None and tesseract_result is None:
            st.error(
                "The document could not be read. "
                + (tesseract_error or vision_warning or "No extraction route succeeded.")
            )
            return

        raw_fields, field_sources = merge_extraction_results(vision_result, tesseract_result)
        filled_fields = fill_defaults(raw_fields)

        methods = []
        if vision_result is not None:
            methods.append("Vision")
        if tesseract_result is not None:
            methods.append("multi-pass Tesseract")

        _init_session_fields(filled_fields)
        st.session_state["ocr_raw_fields"] = raw_fields
        st.session_state["ocr_field_sources"] = field_sources
        st.session_state["ocr_file_id"] = file_id
        st.session_state["ocr_extraction_method"] = " + ".join(methods)
        st.session_state["ocr_extracted_text"] = extracted_text
        st.session_state["ocr_vision_warning"] = vision_warning

    raw_fields = st.session_state.get("ocr_raw_fields", _empty_fields())
    field_sources = st.session_state.get("ocr_field_sources", {})
    session_fields = _get_session_fields() or fill_defaults({})
    extraction_method = st.session_state.get("ocr_extraction_method", "")
    extracted_text = st.session_state.get("ocr_extracted_text", "")
    vision_warning = st.session_state.get("ocr_vision_warning")

    st.success(f"Extraction complete using **{extraction_method}**.")
    if vision_warning and "not configured" not in vision_warning.lower():
        st.warning(vision_warning)
    elif vision_warning and "not configured" in vision_warning.lower():
        st.info(
            "Handwriting vision is not configured, so the local Tesseract fallback was used. "
            "Add ANTHROPIC_API_KEY to your .env file for stronger handwriting recognition."
        )

    if extracted_text:
        with st.expander("View raw OCR text"):
            st.text_area("Raw text", extracted_text, height=260)

    st.subheader("Step 2: Auto-Detected Fields")
    left, right = st.columns(2)

    with left:
        st.write("**Extracted from document**")
        detected_rows = [
            {
                "Field": field,
                "Detected Value": raw_fields[field],
                "Extraction Source": field_sources.get(field, "Unknown"),
            }
            for field in FIELD_NAMES
            if raw_fields.get(field) is not None
        ]
        if detected_rows:
            st.dataframe(pd.DataFrame(detected_rows), use_container_width=True, hide_index=True)
        else:
            st.warning(
                "No fields were auto-detected. Enter and verify the values manually below."
            )

    with right:
        st.write("**Values loaded into form**")
        st.dataframe(
            pd.DataFrame(
                {
                    "Field": FIELD_NAMES,
                    "Value": [session_fields[field] for field in FIELD_NAMES],
                    "Source": [
                        field_sources.get(field, "Default")
                        if raw_fields.get(field) is not None
                        else "Default"
                        for field in FIELD_NAMES
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Step 3: Review and Correct Values")
    st.caption(
        "OCR is assistive. Confirm the handwritten numbers, especially claim amount, approved amount, "
        "length of stay, prior visits and monthly provider claims."
    )

    f = session_fields
    insurance_options = ["Private", "Government", "Medicaid", "Self-Pay"]
    status_options = ["Approved", "Pending", "Rejected"]
    visit_options = ["Inpatient", "Outpatient", "Emergency"]

    with st.form("ocr_review_form"):
        col1, col2 = st.columns(2)

        with col1:
            claim_id = st.text_input("Claim ID", value=str(f["Claim_ID"]))
            provider_id = st.text_input("Provider ID", value=str(f["Provider_ID"]))
            claim_amount = st.number_input(
                "Claim Amount",
                min_value=0.0,
                value=float(f["Claim_Amount"]),
                step=100.0,
                format="%.2f",
            )
            approved_amount = st.number_input(
                "Approved Amount",
                min_value=0.0,
                value=float(f["Approved_Amount"]),
                step=100.0,
                format="%.2f",
            )
            patient_age = st.number_input(
                "Patient Age",
                min_value=0,
                max_value=120,
                value=int(f["Patient_Age"]),
                step=1,
            )
            patient_gender = st.selectbox(
                "Patient Gender",
                ["Male", "Female"],
                index=0 if f["Patient_Gender"] == "Male" else 1,
            )
            insurance_type = st.selectbox(
                "Insurance Type",
                insurance_options,
                index=insurance_options.index(f["Insurance_Type"])
                if f["Insurance_Type"] in insurance_options
                else 0,
            )
            claim_status = st.selectbox(
                "Claim Status",
                status_options,
                index=status_options.index(f["Claim_Status"])
                if f["Claim_Status"] in status_options
                else 1,
            )

        with col2:
            diagnosis_code = st.text_input("Diagnosis Code", value=str(f["Diagnosis_Code"]))
            procedure_code = st.text_input("Procedure Code", value=str(f["Procedure_Code"]))
            provider_specialty = st.text_input(
                "Provider Specialty", value=str(f["Provider_Specialty"])
            )
            patient_state = st.text_input("Patient State", value=str(f["Patient_State"]))
            days_between = st.number_input(
                "Days Between Service and Claim",
                min_value=0,
                value=int(f["Days_Between_Service_and_Claim"]),
                step=1,
            )
            claims_monthly = st.number_input(
                "Claims Per Provider Monthly",
                min_value=0,
                value=int(f["Number_of_Claims_Per_Provider_Monthly"]),
                step=1,
            )
            length_of_stay = st.number_input(
                "Length of Stay",
                min_value=0,
                value=int(f["Length_of_Stay"]),
                step=1,
            )
            prior_visits = st.number_input(
                "Prior Visits in 12 Months",
                min_value=0,
                value=int(f["Prior_Visits_12m"]),
                step=1,
            )
            chronic = st.selectbox(
                "Chronic Condition",
                [0, 1],
                index=int(f["Chronic_Condition_Flag"]),
                format_func=lambda value: "Yes" if value == 1 else "No",
            )
            visit_type = st.selectbox(
                "Visit Type",
                visit_options,
                index=visit_options.index(f["Visit_Type"])
                if f["Visit_Type"] in visit_options
                else 1,
            )

        run_button = st.form_submit_button("Run Fraud Detection", type="primary")

    if run_button:
        final_fields = {
            "Claim_ID": claim_id.strip(),
            "Provider_ID": provider_id.strip(),
            "Claim_Amount": float(claim_amount),
            "Approved_Amount": float(approved_amount),
            "Diagnosis_Code": diagnosis_code.strip(),
            "Procedure_Code": procedure_code.strip(),
            "Patient_Age": int(patient_age),
            "Insurance_Type": insurance_type,
            "Patient_Gender": patient_gender,
            "Claim_Status": claim_status,
            "Days_Between_Service_and_Claim": int(days_between),
            "Number_of_Claims_Per_Provider_Monthly": int(claims_monthly),
            "Length_of_Stay": int(length_of_stay),
            "Prior_Visits_12m": int(prior_visits),
            "Chronic_Condition_Flag": int(chronic),
            "Provider_Specialty": provider_specialty.strip(),
            "Patient_State": patient_state.strip().upper(),
            "Visit_Type": visit_type,
        }

        if final_fields["Approved_Amount"] > final_fields["Claim_Amount"]:
            st.warning("Approved Amount is greater than Claim Amount. Please verify both values.")

        try:
            X_scaled = preprocess_for_model(final_fields, scaler, preprocess_info)
            fraud_score = max(0.0, min(1.0, predict_fraud_score(model, X_scaled)))
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        st.divider()
        st.subheader("Prediction Result")
        decision = "Fraudulent" if fraud_score >= 0.5 else "Legitimate"
        risk = "High" if fraud_score >= 0.70 else ("Medium" if fraud_score >= 0.50 else "Low")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Fraud Probability", f"{fraud_score * 100:.2f}%")
        metric2.metric("Decision", decision)
        metric3.metric("Risk Level", risk)
        st.progress(fraud_score)

        claim_data = {
            "source": "OCR Scanner",
            "filename": uploaded_file.name,
            "Claim_ID": final_fields["Claim_ID"],
            "Provider_ID": final_fields["Provider_ID"],
            "Fraud_Score": round(fraud_score, 4),
            "Decision": decision,
        }
        block = add_blockchain_record(bc, claim_data, fraud_score)
        block_hash = get_block_hash(block)

        st.subheader("Blockchain Record")
        st.success(f"Prediction result recorded in block #{block.index}")
        st.code(
            f"Block Index    : {block.index}\n"
            f"Timestamp      : {block.timestamp}\n"
            f"Source         : OCR Scanner\n"
            f"File Name      : {uploaded_file.name}\n"
            f"Decision       : {block.decision}\n"
            f"Fraud Score    : {block.fraud_score}\n"
            f"Claim Hash     : {block.claim_hash}\n"
            f"Block Hash     : {block_hash}\n"
            f"Previous Hash  : {block.previous_hash}"
        )

        st.subheader("Final Values Used for Prediction")
        st.dataframe(
            pd.DataFrame(
                {
                    "Field": list(final_fields.keys()),
                    "Value": list(final_fields.values()),
                    "Source": [
                        field_sources.get(field, "Default/Edited")
                        if raw_fields.get(field) is not None
                        else "Default/Edited"
                        for field in final_fields
                    ],
                }
            ),
            use_container_width=True,
            hide_index=True,
        )