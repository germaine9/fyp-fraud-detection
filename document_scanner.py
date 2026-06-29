# document_scanner.py
# OCR-assisted claim document scanner for the Streamlit app.
# Uses Claude Vision API for handwriting detection (much more accurate than Tesseract alone).

import streamlit as st
import pandas as pd
import re
import base64
import json
import requests
import os
from datetime import datetime
from PIL import Image, ImageFilter, ImageEnhance
import io

import pytesseract
import pdfplumber
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ============================================================
# CLAUDE VISION — PRIMARY EXTRACTOR (handles handwriting well)
# ============================================================

CLAUDE_VISION_PROMPT = """You are a medical claims data extractor. Carefully read every word in this claim document image — including any handwritten text — and extract the following fields.

Return your answer as a SINGLE valid JSON object with EXACTLY these keys (use null for any field you cannot find):

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

Rules:
- Claim_Amount and Approved_Amount: return as a number (e.g. 1500.00), no currency symbols
- Patient_Age: return as an integer
- Insurance_Type: one of "Private", "Government", "Medicaid", "Self-Pay"
- Patient_Gender: "Male" or "Female"
- Claim_Status: one of "Approved", "Pending", "Rejected"
- Chronic_Condition_Flag: 0 or 1
- Visit_Type: one of "Inpatient", "Outpatient", "Emergency"
- Days_Between_Service_and_Claim, Number_of_Claims_Per_Provider_Monthly, Length_of_Stay, Prior_Visits_12m: integers
- Do NOT include any explanation, markdown, or text outside the JSON object.
"""


def image_to_base64(image_file) -> tuple[str, str]:
    """Convert an uploaded file or PIL Image to base64 + media_type."""
    if isinstance(image_file, Image.Image):
        buf = io.BytesIO()
        image_file.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode(), "image/png"

    image_file.seek(0)
    raw = image_file.read()
    image_file.seek(0)
    ext = image_file.name.rsplit(".", 1)[-1].lower()
    media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png"}.get(ext, "image/png")
    return base64.b64encode(raw).decode(), media_type


def extract_fields_via_claude_vision(image_file) -> dict | None:
    """
    Send the image to Claude Vision and get back a structured JSON of claim fields.
    Returns the parsed dict, or None if the API call fails.
    """
    try:
        b64, media_type = image_to_base64(image_file)

        payload = {
            "model": "claude-sonnet-4-6",
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": b64,
                            },
                        },
                        {
                            "type": "text",
                            "text": CLAUDE_VISION_PROMPT,
                        },
                    ],
                }
            ],
        }

        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            return None  # silently fall back to Tesseract

        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            json=payload,
            timeout=30,
        )

        if response.status_code != 200:
            return None  # silently fall back to Tesseract

        data = response.json()
        raw_text = "".join(
            block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
        )

        # Strip markdown fences if present
        raw_text = re.sub(r"```(?:json)?", "", raw_text).strip().rstrip("`").strip()

        parsed = json.loads(raw_text)
        return parsed

    except requests.exceptions.Timeout:
        return None  # silently fall back to Tesseract
    except Exception:
        return None  # silently fall back to Tesseract


# ============================================================
# IMAGE PRE-PROCESSING (Tesseract fallback)
# ============================================================

def preprocess_image_for_ocr(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    w, h = image.size
    scale = max(1, 2400 // max(w, h))
    if scale > 1:
        image = image.resize((w * scale, h * scale), Image.LANCZOS)
    image = image.convert("L")
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2.0)
    image = image.filter(ImageFilter.SHARPEN)
    return image


# ============================================================
# TEXT EXTRACTION (Tesseract fallback for PDFs / when Vision fails)
# ============================================================

def extract_text_from_image_tesseract(image_file) -> str:
    try:
        image = Image.open(image_file)
        processed = preprocess_image_for_ocr(image)
        results = []
        for psm in [6, 4, 11, 12]:
            config = f"--psm {psm} --oem 3"
            text = pytesseract.image_to_string(processed, config=config).strip()
            if text:
                results.append(text)
        return max(results, key=len) if results else ""
    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_text_from_pdf(pdf_file) -> str:
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


# ============================================================
# FIELD EXTRACTION (regex fallback for plain text)
# ============================================================

def parse_claim_fields_from_text(text: str) -> dict:
    fields = {k: None for k in [
        "Claim_ID", "Provider_ID", "Claim_Amount", "Approved_Amount",
        "Diagnosis_Code", "Procedure_Code", "Patient_Age", "Insurance_Type",
        "Patient_Gender", "Claim_Status", "Days_Between_Service_and_Claim",
        "Number_of_Claims_Per_Provider_Monthly", "Length_of_Stay",
        "Prior_Visits_12m", "Chronic_Condition_Flag", "Provider_Specialty",
        "Patient_State", "Visit_Type",
    ]}

    m = re.search(r"claim\s*(?:id|no|number|#)[\s:]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if m: fields["Claim_ID"] = m.group(1).strip()

    m = re.search(r"provider\s*(?:id|no|number|#)[\s:]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if m: fields["Provider_ID"] = m.group(1).strip()

    for pat in [r"claim\s*amount[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
                r"total\s*amount[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
                r"\b(?:RM|\$)\s*([0-9,]+(?:\.[0-9]{1,2})?)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: fields["Claim_Amount"] = float(m.group(1).replace(",", "")); break

    for pat in [r"approved\s*amount[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
                r"paid\s*amount[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m: fields["Approved_Amount"] = float(m.group(1).replace(",", "")); break

    m = re.search(r"diagnosis\s*(?:code|icd)?[\s:]*([A-Z][0-9]{2,3}(?:\.[0-9]{1,4})?)", text, re.IGNORECASE)
    if m: fields["Diagnosis_Code"] = m.group(1).strip()

    m = re.search(r"procedure\s*(?:code|cpt)?[\s:]*([0-9]{4,5}[A-Z0-9]*)", text, re.IGNORECASE)
    if m: fields["Procedure_Code"] = m.group(1).strip()

    for pat in [r"(?:patient\s*)?age[\s:]*(\d{1,3})", r"(\d{1,3})\s*(?:year[s]?\s*old|y/?o)"]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            v = int(m.group(1))
            if v > 1900: v = datetime.now().year - v
            if 0 < v < 130: fields["Patient_Age"] = v
            break

    m = re.search(r"insurance\s*(?:type)?[\s:]*(private|government|medicare|medicaid|self-pay|self pay)", text, re.IGNORECASE)
    if m:
        raw = m.group(1).lower()
        fields["Insurance_Type"] = "Government" if raw in ["medicare","medicaid","government"] else ("Self-Pay" if "self" in raw else "Private")

    m = re.search(r"(?:gender|sex)[\s:]*(male|female|m|f)\b", text, re.IGNORECASE)
    if m: fields["Patient_Gender"] = "Male" if m.group(1).lower() in ["male","m"] else "Female"

    m = re.search(r"status[\s:]*(approved|pending|rejected|denied)", text, re.IGNORECASE)
    if m:
        s = m.group(1).capitalize()
        fields["Claim_Status"] = "Rejected" if s == "Denied" else s

    m = re.search(r"days?\s*(?:between|delay)[\s:]*(\d+)", text, re.IGNORECASE)
    if m: fields["Days_Between_Service_and_Claim"] = int(m.group(1))

    m = re.search(r"claims?\s*per\s*(?:provider|month)[\s:]*(\d+)", text, re.IGNORECASE)
    if m: fields["Number_of_Claims_Per_Provider_Monthly"] = int(m.group(1))

    m = re.search(r"length\s*of\s*stay[\s:]*(\d+)", text, re.IGNORECASE)
    if m: fields["Length_of_Stay"] = int(m.group(1))

    m = re.search(r"prior\s*visits?[\s:]*(\d+)", text, re.IGNORECASE)
    if m: fields["Prior_Visits_12m"] = int(m.group(1))

    m = re.search(r"chronic\s*condition[\s:]*(yes|no|1|0)", text, re.IGNORECASE)
    if m: fields["Chronic_Condition_Flag"] = 1 if m.group(1).lower() in ["yes","1"] else 0

    for spec in ["cardiology","orthopedics","neurology","oncology","general practice","radiology","surgery","psychiatry","dermatology","pediatrics"]:
        if re.search(spec, text, re.IGNORECASE):
            fields["Provider_Specialty"] = spec.title(); break

    m = re.search(r"(?:state|location)[\s:]*([A-Z]{2})\b", text)
    if m: fields["Patient_State"] = m.group(1)

    m = re.search(r"visit\s*type[\s:]*(inpatient|outpatient|emergency)", text, re.IGNORECASE)
    if m: fields["Visit_Type"] = m.group(1).capitalize()

    return fields


def sanitise_vision_fields(raw: dict) -> dict:
    """
    Normalise Claude Vision JSON output to match the expected types/values.
    """
    insurance_map = {
        "private": "Private", "government": "Government",
        "medicare": "Government", "medicaid": "Medicaid", "self-pay": "Self-Pay", "self pay": "Self-Pay",
    }
    status_map  = {"approved": "Approved", "pending": "Pending", "rejected": "Rejected", "denied": "Rejected"}
    visit_map   = {"inpatient": "Inpatient", "outpatient": "Outpatient", "emergency": "Emergency"}
    gender_map  = {"male": "Male", "m": "Male", "female": "Female", "f": "Female"}

    def _str(v):  return str(v).strip() if v not in (None, "", "null") else None
    def _float(v):
        try: return float(str(v).replace(",", "").replace("$", "").replace("RM", "").strip())
        except: return None
    def _int(v):
        try: return int(str(v).strip())
        except: return None

    out = {}
    out["Claim_ID"]           = _str(raw.get("Claim_ID"))
    out["Provider_ID"]        = _str(raw.get("Provider_ID"))
    out["Claim_Amount"]       = _float(raw.get("Claim_Amount"))
    out["Approved_Amount"]    = _float(raw.get("Approved_Amount"))
    out["Diagnosis_Code"]     = _str(raw.get("Diagnosis_Code"))
    out["Procedure_Code"]     = _str(raw.get("Procedure_Code"))
    out["Patient_Age"]        = _int(raw.get("Patient_Age"))
    out["Days_Between_Service_and_Claim"]        = _int(raw.get("Days_Between_Service_and_Claim"))
    out["Number_of_Claims_Per_Provider_Monthly"] = _int(raw.get("Number_of_Claims_Per_Provider_Monthly"))
    out["Length_of_Stay"]     = _int(raw.get("Length_of_Stay"))
    out["Prior_Visits_12m"]   = _int(raw.get("Prior_Visits_12m"))
    out["Chronic_Condition_Flag"] = _int(raw.get("Chronic_Condition_Flag"))
    out["Provider_Specialty"] = _str(raw.get("Provider_Specialty"))
    out["Patient_State"]      = _str(raw.get("Patient_State"))

    ins = _str(raw.get("Insurance_Type"))
    out["Insurance_Type"] = insurance_map.get(ins.lower(), ins) if ins else None

    gen = _str(raw.get("Patient_Gender"))
    out["Patient_Gender"] = gender_map.get(gen.lower(), gen) if gen else None

    sta = _str(raw.get("Claim_Status"))
    out["Claim_Status"] = status_map.get(sta.lower(), sta) if sta else None

    vis = _str(raw.get("Visit_Type"))
    out["Visit_Type"] = visit_map.get(vis.lower(), vis) if vis else None

    return out


def fill_defaults(fields: dict) -> dict:
    defaults = {
        "Claim_ID": "CLM-OCR", "Provider_ID": "PRV-UNKNOWN",
        "Claim_Amount": 5000.0, "Approved_Amount": 4500.0,
        "Diagnosis_Code": "D001", "Procedure_Code": "P001",
        "Patient_Age": 45, "Insurance_Type": "Private",
        "Patient_Gender": "Male", "Claim_Status": "Pending",
        "Days_Between_Service_and_Claim": 5,
        "Number_of_Claims_Per_Provider_Monthly": 10,
        "Length_of_Stay": 3, "Prior_Visits_12m": 2,
        "Chronic_Condition_Flag": 0, "Provider_Specialty": "General Practice",
        "Patient_State": "CA", "Visit_Type": "Outpatient",
    }
    return {k: fields.get(k) if fields.get(k) is not None else v for k, v in defaults.items()}


# ============================================================
# SESSION STATE
# ============================================================

SESSION_KEY = "ocr_filled_fields"

def _init_session_fields(filled: dict):
    st.session_state[SESSION_KEY] = filled.copy()

def _get_session_fields() -> dict | None:
    return st.session_state.get(SESSION_KEY)


# ============================================================
# PREPROCESSING + PREDICTION
# ============================================================

def preprocess_for_ann(fields, scaler, preprocess_info):
    input_dict = {k: fields[k] for k in [
        "Patient_Age","Patient_Gender","Diagnosis_Code","Procedure_Code",
        "Claim_Amount","Approved_Amount","Insurance_Type",
        "Days_Between_Service_and_Claim","Number_of_Claims_Per_Provider_Monthly",
        "Provider_Specialty","Patient_State","Claim_Status",
        "Length_of_Stay","Visit_Type","Chronic_Condition_Flag","Prior_Visits_12m",
    ]}
    df = pd.DataFrame([input_dict])

    for col in preprocess_info["numeric_cols"]:
        df[col] = df.get(col, pd.Series([preprocess_info["numeric_means"][col]])).fillna(preprocess_info["numeric_means"][col])
    for col in preprocess_info["categorical_cols"]:
        df[col] = df.get(col, pd.Series([preprocess_info["categorical_modes"][col]])).fillna(preprocess_info["categorical_modes"][col])

    df["claim_to_cost_ratio"]  = df["Claim_Amount"] / (df["Approved_Amount"] + 1)
    df["cost_outlier_flag"]    = (df["Claim_Amount"] > preprocess_info["claim_q3"] + 1.5 * preprocess_info["claim_iqr"]).astype(int)
    df["high_claim_frequency"] = (df["Number_of_Claims_Per_Provider_Monthly"] > preprocess_info["high_claim_threshold"]).astype(int)

    df = pd.get_dummies(df)
    df = df.reindex(columns=preprocess_info["feature_columns"], fill_value=0)
    return scaler.transform(df)


def predict_fraud_score(model, X_scaled) -> float:
    try:
        return float(model.predict(X_scaled, verbose=0).flatten()[0])
    except TypeError:
        pass
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X_scaled)[0][1])
    return float(model.predict(X_scaled)[0])


# ============================================================
# BLOCKCHAIN HELPERS
# ============================================================

def add_blockchain_record(bc, claim_data, fraud_score):
    try:    return bc.add_record(claim_data, fraud_score, source="OCR Scanner")
    except TypeError: return bc.add_record(claim_data, fraud_score)

def get_block_hash(block):
    return getattr(block, "hash", getattr(block, "block_hash", "N/A"))


# ============================================================
# MAIN RENDER
# ============================================================

def render_document_scanner(model, scaler, bc, preprocess_info=None):
    st.title("OCR Claim Document Scanner")
    st.write(
        "Upload a healthcare claim document (PDF, JPG, JPEG, or PNG). "
        "Supports **printed** and **handwritten** documents — Claude Vision AI reads the document directly."
    )
    st.caption("Auto-detected fields are pre-filled below. Review and correct before running fraud detection.")
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
        type=["pdf", "jpg", "jpeg", "png"],
        help="Printed or handwritten claim forms — PDF, JPG, or PNG.",
    )

    if uploaded_file is None:
        if SESSION_KEY in st.session_state:
            del st.session_state[SESSION_KEY]
        st.info("Upload a claim document to begin.")
        return

    file_type = uploaded_file.type
    st.success(f"Uploaded: **{uploaded_file.name}**")

    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        st.subheader("Document Preview")
        preview_image = Image.open(uploaded_file)
        st.image(preview_image, caption=uploaded_file.name, use_container_width=True)
        uploaded_file.seek(0)

    # ── Extraction (cached per file) ───────────────────────────────────────────
    st.subheader("Step 1: Text & Field Extraction")
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("ocr_file_id") != file_id:
        extraction_method = "unknown"

        if file_type == "application/pdf":
            # PDFs: use pdfplumber text extraction → regex
            with st.spinner("Extracting text from PDF..."):
                text = extract_text_from_pdf(uploaded_file)
            if text.startswith("ERROR") or not text:
                st.error(f"PDF extraction failed: {text}")
                return
            raw_fields = parse_claim_fields_from_text(text)
            st.session_state["ocr_extracted_text"] = text
            extraction_method = "PDF text + regex"

        else:
            # Images: try Claude Vision first (best for handwriting)
            with st.spinner("Reading document with Claude Vision AI (handles handwriting)..."):
                vision_result = extract_fields_via_claude_vision(uploaded_file)

            if vision_result is not None:
                raw_fields = sanitise_vision_fields(vision_result)
                extraction_method = "Claude Vision AI"
                # Also run Tesseract to show raw text in expander
                uploaded_file.seek(0)
                tess_text = extract_text_from_image_tesseract(uploaded_file)
                st.session_state["ocr_extracted_text"] = tess_text
            else:
                # Silently fall back to Tesseract + regex
                uploaded_file.seek(0)
                with st.spinner("Running Tesseract OCR..."):
                    text = extract_text_from_image_tesseract(uploaded_file)
                if text.startswith("ERROR") or not text:
                    st.error(f"OCR failed: {text}")
                    return
                raw_fields = parse_claim_fields_from_text(text)
                st.session_state["ocr_extracted_text"] = text
                extraction_method = "Tesseract OCR + regex"

        filled = fill_defaults(raw_fields)
        _init_session_fields(filled)
        st.session_state["ocr_raw_fields"]      = raw_fields
        st.session_state["ocr_file_id"]         = file_id
        st.session_state["ocr_extraction_method"] = extraction_method

    # Read back from session state
    raw_fields        = st.session_state.get("ocr_raw_fields", {})
    session_fields    = _get_session_fields() or fill_defaults({})
    extraction_method = st.session_state.get("ocr_extraction_method", "")
    extracted_text    = st.session_state.get("ocr_extracted_text", "")

    st.success(f"Extraction complete using **{extraction_method}**.")
    if extracted_text:
        with st.expander("View raw OCR text"):
            st.text_area("Raw text", extracted_text, height=200)

    # ── Detected fields summary ────────────────────────────────────────────────
    st.subheader("Step 2: Auto-Detected Fields")
    left, right = st.columns(2)
    with left:
        st.write("**Extracted from document**")
        detected = {k: v for k, v in raw_fields.items() if v is not None}
        if detected:
            st.dataframe(
                pd.DataFrame({"Field": list(detected.keys()), "Detected Value": list(detected.values())}),
                use_container_width=True,
            )
        else:
            st.warning("No fields auto-detected. Fill in manually below.")

    with right:
        st.write("**Values loaded into form**")
        st.dataframe(
            pd.DataFrame({
                "Field":  list(session_fields.keys()),
                "Value":  list(session_fields.values()),
                "Source": ["Extracted" if raw_fields.get(k) is not None else "Default" for k in session_fields],
            }),
            use_container_width=True,
        )

    # ── Review form ────────────────────────────────────────────────────────────
    st.subheader("Step 3: Review and Correct Values")
    st.caption("Fields auto-filled from the document are marked **Extracted**. Anything not found uses a default.")

    f = session_fields
    insurance_options = ["Private", "Government", "Medicaid", "Self-Pay"]
    status_options    = ["Approved", "Pending", "Rejected"]
    visit_options     = ["Inpatient", "Outpatient", "Emergency"]

    with st.form("ocr_review_form"):
        col1, col2 = st.columns(2)

        with col1:
            claim_id        = st.text_input("Claim ID",        value=str(f["Claim_ID"]))
            provider_id     = st.text_input("Provider ID",     value=str(f["Provider_ID"]))
            claim_amount    = st.number_input("Claim Amount",    min_value=0.0, value=float(f["Claim_Amount"]))
            approved_amount = st.number_input("Approved Amount", min_value=0.0, value=float(f["Approved_Amount"]))
            patient_age     = st.number_input("Patient Age", min_value=0, max_value=120, value=int(f["Patient_Age"]))
            patient_gender  = st.selectbox("Patient Gender", ["Male", "Female"],
                                           index=0 if f["Patient_Gender"] == "Male" else 1)
            insurance_type  = st.selectbox("Insurance Type", insurance_options,
                                           index=insurance_options.index(f["Insurance_Type"]) if f["Insurance_Type"] in insurance_options else 0)
            claim_status    = st.selectbox("Claim Status", status_options,
                                           index=status_options.index(f["Claim_Status"]) if f["Claim_Status"] in status_options else 1)

        with col2:
            diagnosis_code    = st.text_input("Diagnosis Code",     value=str(f["Diagnosis_Code"]))
            procedure_code    = st.text_input("Procedure Code",     value=str(f["Procedure_Code"]))
            provider_specialty= st.text_input("Provider Specialty", value=str(f["Provider_Specialty"]))
            patient_state     = st.text_input("Patient State",      value=str(f["Patient_State"]))
            days_between      = st.number_input("Days Between Service and Claim", min_value=0, value=int(f["Days_Between_Service_and_Claim"]))
            claims_monthly    = st.number_input("Claims Per Provider Monthly",    min_value=0, value=int(f["Number_of_Claims_Per_Provider_Monthly"]))
            length_of_stay    = st.number_input("Length of Stay",                min_value=0, value=int(f["Length_of_Stay"]))
            prior_visits      = st.number_input("Prior Visits in 12 Months",     min_value=0, value=int(f["Prior_Visits_12m"]))
            chronic           = st.selectbox("Chronic Condition", [0, 1],
                                             index=int(f["Chronic_Condition_Flag"]),
                                             format_func=lambda v: "Yes" if v == 1 else "No")
            visit_type        = st.selectbox("Visit Type", visit_options,
                                             index=visit_options.index(f["Visit_Type"]) if f["Visit_Type"] in visit_options else 1)

        run_button = st.form_submit_button("Run Fraud Detection")

    # ── Prediction ─────────────────────────────────────────────────────────────
    if run_button:
        final_fields = {
            "Claim_ID": claim_id, "Provider_ID": provider_id,
            "Claim_Amount": claim_amount, "Approved_Amount": approved_amount,
            "Diagnosis_Code": diagnosis_code, "Procedure_Code": procedure_code,
            "Patient_Age": patient_age, "Insurance_Type": insurance_type,
            "Patient_Gender": patient_gender, "Claim_Status": claim_status,
            "Days_Between_Service_and_Claim": days_between,
            "Number_of_Claims_Per_Provider_Monthly": claims_monthly,
            "Length_of_Stay": length_of_stay, "Prior_Visits_12m": prior_visits,
            "Chronic_Condition_Flag": chronic, "Provider_Specialty": provider_specialty,
            "Patient_State": patient_state, "Visit_Type": visit_type,
        }

        try:
            X_scaled    = preprocess_for_ann(final_fields, scaler, preprocess_info)
            fraud_score = predict_fraud_score(model, X_scaled)
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            return

        st.divider()
        st.subheader("Prediction Result")
        decision = "Fraudulent" if fraud_score >= 0.5 else "Legitimate"
        risk     = "High" if fraud_score >= 0.70 else ("Medium" if fraud_score >= 0.50 else "Low")

        c1, c2, c3 = st.columns(3)
        c1.metric("Fraud Probability", f"{fraud_score * 100:.2f}%")
        c2.metric("Decision", decision)
        c3.metric("Risk Level", risk)
        st.progress(min(fraud_score, 1.0))

        claim_data = {
            "source": "OCR Scanner", "filename": uploaded_file.name,
            "Claim_ID": final_fields["Claim_ID"], "Provider_ID": final_fields["Provider_ID"],
            "Fraud_Score": round(fraud_score, 4), "Decision": decision,
        }
        block      = add_blockchain_record(bc, claim_data, fraud_score)
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
            pd.DataFrame({
                "Field":  list(final_fields.keys()),
                "Value":  list(final_fields.values()),
                "Source": ["Extracted" if raw_fields.get(k) is not None else "Default/Edited" for k in final_fields],
            }),
            use_container_width=True,
        )