# document_scanner_fixed.py
# OCR-assisted claim document scanner for the Streamlit app.
# Rename this file to document_scanner.py before running your main Streamlit app.

import streamlit as st
import pandas as pd
import hashlib
import json
import re
from datetime import datetime
from PIL import Image

import pytesseract
import pdfplumber


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_from_image(image_file):
    """Extract text from JPG/PNG images using pytesseract."""
    try:
        image = Image.open(image_file)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as error:
        return f"ERROR: {str(error)}"


def extract_text_from_pdf(pdf_file):
    """Extract text from text-based PDF using pdfplumber."""
    try:
        text = ""
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as error:
        return f"ERROR: {str(error)}"


# ============================================================
# FIELD EXTRACTION
# ============================================================

def parse_claim_fields(text):
    """
    Extract claim-related fields from OCR/PDF text using regex patterns.
    The user should review and correct these values before prediction.
    """
    fields = {
        "Claim_ID": None,
        "Provider_ID": None,
        "Claim_Amount": None,
        "Approved_Amount": None,
        "Diagnosis_Code": None,
        "Procedure_Code": None,
        "Patient_Age": None,
        "Insurance_Type": None,
        "Patient_Gender": None,
        "Claim_Status": None,
        "Days_Between_Service_and_Claim": None,
        "Number_of_Claims_Per_Provider_Monthly": None,
        "Length_of_Stay": None,
        "Prior_Visits_12m": None,
        "Chronic_Condition_Flag": None,
        "Provider_Specialty": None,
        "Patient_State": None,
        "Visit_Type": None,
    }

    match = re.search(r"claim\s*(?:id|no|number|#)[\s:]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if match:
        fields["Claim_ID"] = match.group(1).strip()

    match = re.search(r"provider\s*(?:id|no|number|#)[\s:]*([A-Z0-9\-]+)", text, re.IGNORECASE)
    if match:
        fields["Provider_ID"] = match.group(1).strip()

    amount_patterns = [
        r"claim\s*amount[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
        r"total\s*amount[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
        r"billed\s*amount[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
        r"amount\s*billed[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
        r"\b(?:RM|\$)\s*([0-9,]+(?:\.[0-9]{1,2})?)",
    ]
    for pattern in amount_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fields["Claim_Amount"] = float(match.group(1).replace(",", ""))
            break

    approved_patterns = [
        r"approved\s*amount[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
        r"amount\s*approved[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
        r"paid\s*amount[\s:$RM]*([0-9,]+(?:\.[0-9]{1,2})?)",
    ]
    for pattern in approved_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            fields["Approved_Amount"] = float(match.group(1).replace(",", ""))
            break

    match = re.search(
        r"diagnosis\s*(?:code|icd)?[\s:]*([A-Z][0-9]{2,3}(?:\.[0-9]{1,4})?)",
        text,
        re.IGNORECASE,
    )
    if match:
        fields["Diagnosis_Code"] = match.group(1).strip()
    else:
        match = re.search(r"\b([A-Z][0-9]{2}(?:\.[0-9]{1,4})?)\b", text)
        if match:
            fields["Diagnosis_Code"] = match.group(1).strip()

    match = re.search(r"procedure\s*(?:code|cpt)?[\s:]*([0-9]{4,5}[A-Z0-9]*)", text, re.IGNORECASE)
    if match:
        fields["Procedure_Code"] = match.group(1).strip()

    age_patterns = [
        r"(?:patient\s*)?age[\s:]*(\d{1,3})",
        r"(\d{1,3})\s*(?:year[s]?\s*old|y/?o)",
        r"dob[\s:]*\d{1,2}[\/\-]\d{1,2}[\/\-](\d{4})",
    ]
    for pattern in age_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = int(match.group(1))
            if value > 1900:
                value = datetime.now().year - value
            if 0 < value < 130:
                fields["Patient_Age"] = value
            break

    match = re.search(
        r"insurance\s*(?:type|plan|provider)?[\s:]*(private|government|medicare|medicaid|commercial|self-pay|self pay)",
        text,
        re.IGNORECASE,
    )
    if match:
        raw = match.group(1).lower()
        if raw in ["medicare", "medicaid", "government"]:
            fields["Insurance_Type"] = "Government"
        elif raw in ["self-pay", "self pay"]:
            fields["Insurance_Type"] = "Self-Pay"
        else:
            fields["Insurance_Type"] = "Private"

    match = re.search(r"(?:gender|sex)[\s:]*(male|female|m|f)\b", text, re.IGNORECASE)
    if match:
        raw = match.group(1).lower()
        fields["Patient_Gender"] = "Male" if raw in ["male", "m"] else "Female"

    match = re.search(r"(?:claim\s*)?status[\s:]*(approved|pending|rejected|denied)", text, re.IGNORECASE)
    if match:
        status = match.group(1).capitalize()
        fields["Claim_Status"] = "Rejected" if status == "Denied" else status

    match = re.search(r"days?\s*(?:between|to\s*claim|delay)[\s:]*(\d+)", text, re.IGNORECASE)
    if match:
        fields["Days_Between_Service_and_Claim"] = int(match.group(1))

    match = re.search(r"(?:monthly\s*)?claims?\s*(?:per\s*)?(?:provider|month)[\s:]*(\d+)", text, re.IGNORECASE)
    if match:
        fields["Number_of_Claims_Per_Provider_Monthly"] = int(match.group(1))

    match = re.search(r"(?:length\s*of\s*stay|los|days?\s*(?:admitted|hospitalized))[\s:]*(\d+)", text, re.IGNORECASE)
    if match:
        fields["Length_of_Stay"] = int(match.group(1))

    match = re.search(r"prior\s*visits?[\s:]*(\d+)", text, re.IGNORECASE)
    if match:
        fields["Prior_Visits_12m"] = int(match.group(1))

    match = re.search(r"chronic\s*condition[\s:]*(yes|no|1|0|true|false)", text, re.IGNORECASE)
    if match:
        raw = match.group(1).lower()
        fields["Chronic_Condition_Flag"] = 1 if raw in ["yes", "1", "true"] else 0

    specialties = [
        "cardiology", "orthopedics", "neurology", "oncology", "general practice",
        "radiology", "surgery", "psychiatry", "dermatology", "pediatrics"
    ]
    for specialty in specialties:
        if re.search(specialty, text, re.IGNORECASE):
            fields["Provider_Specialty"] = specialty.title()
            break

    match = re.search(r"(?:state|location)[\s:]*([A-Z]{2})\b", text)
    if match:
        fields["Patient_State"] = match.group(1)

    match = re.search(r"(?:visit\s*type|admission\s*type)[\s:]*(inpatient|outpatient|emergency)", text, re.IGNORECASE)
    if match:
        fields["Visit_Type"] = match.group(1).capitalize()

    return fields


def fill_defaults(fields):
    """Fill missing fields with defaults so the model can run after user review."""
    defaults = {
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
    return {key: fields.get(key) if fields.get(key) is not None else default for key, default in defaults.items()}


# ============================================================
# PREPROCESSING FOR ANN
# ============================================================

def preprocess_for_ann(fields, scaler, preprocess_info):
    """
    Convert OCR fields into ANN input using the SAME preprocessing rules
    saved during train_ann_fixed.py.
    """
    input_dict = {
        "Patient_Age": fields["Patient_Age"],
        "Patient_Gender": fields["Patient_Gender"],
        "Diagnosis_Code": fields["Diagnosis_Code"],
        "Procedure_Code": fields["Procedure_Code"],
        "Claim_Amount": fields["Claim_Amount"],
        "Approved_Amount": fields["Approved_Amount"],
        "Insurance_Type": fields["Insurance_Type"],
        "Days_Between_Service_and_Claim": fields["Days_Between_Service_and_Claim"],
        "Number_of_Claims_Per_Provider_Monthly": fields["Number_of_Claims_Per_Provider_Monthly"],
        "Provider_Specialty": fields["Provider_Specialty"],
        "Patient_State": fields["Patient_State"],
        "Claim_Status": fields["Claim_Status"],
        "Length_of_Stay": fields["Length_of_Stay"],
        "Visit_Type": fields["Visit_Type"],
        "Chronic_Condition_Flag": fields["Chronic_Condition_Flag"],
        "Prior_Visits_12m": fields["Prior_Visits_12m"],
    }

    df = pd.DataFrame([input_dict])

    numeric_cols = preprocess_info["numeric_cols"]
    categorical_cols = preprocess_info["categorical_cols"]
    numeric_means = preprocess_info["numeric_means"]
    categorical_modes = preprocess_info["categorical_modes"]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = numeric_means[col]
        else:
            df[col] = df[col].fillna(numeric_means[col])

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = categorical_modes[col]
        else:
            df[col] = df[col].fillna(categorical_modes[col])

    claim_q3 = preprocess_info["claim_q3"]
    claim_iqr = preprocess_info["claim_iqr"]
    high_claim_threshold = preprocess_info["high_claim_threshold"]

    df["claim_to_cost_ratio"] = df["Claim_Amount"] / (df["Approved_Amount"] + 1)
    df["cost_outlier_flag"] = (df["Claim_Amount"] > claim_q3 + 1.5 * claim_iqr).astype(int)
    df["high_claim_frequency"] = (df["Number_of_Claims_Per_Provider_Monthly"] > high_claim_threshold).astype(int)

    df = pd.get_dummies(df)
    df = df.reindex(columns=preprocess_info["feature_columns"], fill_value=0)

    return scaler.transform(df)


# ============================================================
# BLOCKCHAIN HELPERS
# ============================================================

def add_blockchain_record(bc, claim_data, fraud_score):
    """Support both app Blockchain class and standalone blockchain.py class."""
    try:
        return bc.add_record(claim_data, fraud_score, source="OCR Scanner")
    except TypeError:
        return bc.add_record(claim_data, fraud_score)


def get_block_hash(block):
    """Return block hash for different block class versions."""
    return getattr(block, "hash", getattr(block, "block_hash", "N/A"))


# ============================================================
# MAIN STREAMLIT RENDER FUNCTION
# ============================================================

def render_document_scanner(model, scaler, bc, preprocess_info=None):
    """
    Render OCR document scanner page.

    If preprocess_info is not passed from the main app, this function tries to
    load preprocess_info.pkl automatically.
    """
    st.title("OCR Claim Document Scanner")
    st.write(
        "Upload a healthcare claim document in PDF, JPG, JPEG, or PNG format. "
        "The scanner extracts possible claim fields, allows review, runs the ANN model, "
        "and records the prediction result in the blockchain-style ledger."
    )
    st.caption("OCR extraction is assisted and may not be perfect. Please review extracted values before prediction.")
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
        help="Supported formats: PDF, JPG, JPEG, PNG",
    )

    if uploaded_file is None:
        st.info("Upload a claim document to begin.")
        return

    file_type = uploaded_file.type
    st.success(f"Uploaded file: {uploaded_file.name}")

    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        st.subheader("Document Preview")
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_container_width=True)
        uploaded_file.seek(0)

    st.subheader("Step 1: Text Extraction")
    with st.spinner("Extracting text..."):
        extracted_text = extract_text_from_pdf(uploaded_file) if file_type == "application/pdf" else extract_text_from_image(uploaded_file)

    if extracted_text.startswith("ERROR"):
        st.error(f"Text extraction failed: {extracted_text}")
        st.warning("For image OCR, make sure Tesseract OCR is installed and configured on your computer.")
        return

    if not extracted_text:
        st.warning("No text could be extracted from this document.")
        return

    st.success(f"Text extraction completed. Characters extracted: {len(extracted_text)}")
    with st.expander("View extracted text"):
        st.text_area("Raw OCR output", extracted_text, height=220)

    st.subheader("Step 2: Extract Claim Fields")
    raw_fields = parse_claim_fields(extracted_text)
    filled_fields = fill_defaults(raw_fields)

    left, right = st.columns(2)
    with left:
        st.write("**Fields detected from document**")
        extracted_display = {key: value for key, value in raw_fields.items() if value is not None}
        if extracted_display:
            st.dataframe(pd.DataFrame({"Field": list(extracted_display.keys()), "Detected Value": list(extracted_display.values())}), use_container_width=True)
        else:
            st.warning("No fields were detected automatically.")

    with right:
        st.write("**Values prepared for prediction**")
        prepared_display = pd.DataFrame({
            "Field": list(filled_fields.keys()),
            "Value": list(filled_fields.values()),
            "Source": ["Extracted" if raw_fields.get(key) is not None else "Default" for key in filled_fields.keys()],
        })
        st.dataframe(prepared_display, use_container_width=True)

    st.subheader("Step 3: Review and Correct Values")
    with st.form("ocr_review_form"):
        col1, col2 = st.columns(2)

        with col1:
            filled_fields["Claim_ID"] = st.text_input("Claim ID", value=str(filled_fields["Claim_ID"]))
            filled_fields["Provider_ID"] = st.text_input("Provider ID", value=str(filled_fields["Provider_ID"]))
            filled_fields["Claim_Amount"] = st.number_input("Claim Amount", min_value=0.0, value=float(filled_fields["Claim_Amount"]))
            filled_fields["Approved_Amount"] = st.number_input("Approved Amount", min_value=0.0, value=float(filled_fields["Approved_Amount"]))
            filled_fields["Patient_Age"] = st.number_input("Patient Age", min_value=0, max_value=120, value=int(filled_fields["Patient_Age"]))
            filled_fields["Patient_Gender"] = st.selectbox("Patient Gender", ["Male", "Female"], index=0 if filled_fields["Patient_Gender"] == "Male" else 1)

            insurance_options = ["Private", "Government", "Medicaid", "Self-Pay"]
            filled_fields["Insurance_Type"] = st.selectbox(
                "Insurance Type",
                insurance_options,
                index=insurance_options.index(filled_fields["Insurance_Type"]) if filled_fields["Insurance_Type"] in insurance_options else 0,
            )

            status_options = ["Approved", "Pending", "Rejected"]
            filled_fields["Claim_Status"] = st.selectbox(
                "Claim Status",
                status_options,
                index=status_options.index(filled_fields["Claim_Status"]) if filled_fields["Claim_Status"] in status_options else 1,
            )

        with col2:
            filled_fields["Diagnosis_Code"] = st.text_input("Diagnosis Code", value=str(filled_fields["Diagnosis_Code"]))
            filled_fields["Procedure_Code"] = st.text_input("Procedure Code", value=str(filled_fields["Procedure_Code"]))
            filled_fields["Provider_Specialty"] = st.text_input("Provider Specialty", value=str(filled_fields["Provider_Specialty"]))
            filled_fields["Patient_State"] = st.text_input("Patient State", value=str(filled_fields["Patient_State"]))
            filled_fields["Days_Between_Service_and_Claim"] = st.number_input("Days Between Service and Claim", min_value=0, value=int(filled_fields["Days_Between_Service_and_Claim"]))
            filled_fields["Number_of_Claims_Per_Provider_Monthly"] = st.number_input("Claims Per Provider Monthly", min_value=0, value=int(filled_fields["Number_of_Claims_Per_Provider_Monthly"]))
            filled_fields["Length_of_Stay"] = st.number_input("Length of Stay", min_value=0, value=int(filled_fields["Length_of_Stay"]))
            filled_fields["Prior_Visits_12m"] = st.number_input("Prior Visits in 12 Months", min_value=0, value=int(filled_fields["Prior_Visits_12m"]))
            filled_fields["Chronic_Condition_Flag"] = st.selectbox("Chronic Condition", [0, 1], index=int(filled_fields["Chronic_Condition_Flag"]), format_func=lambda value: "Yes" if value == 1 else "No")

            visit_options = ["Inpatient", "Outpatient", "Emergency"]
            filled_fields["Visit_Type"] = st.selectbox(
                "Visit Type",
                visit_options,
                index=visit_options.index(filled_fields["Visit_Type"]) if filled_fields["Visit_Type"] in visit_options else 1,
            )

        run_button = st.form_submit_button("Run Fraud Detection")

    if run_button:
        try:
            X_scaled = preprocess_for_ann(filled_fields, scaler, preprocess_info)
            fraud_score = float(model.predict(X_scaled, verbose=0).flatten()[0])
            decision = "Fraudulent" if fraud_score >= 0.5 else "Legitimate"
        except Exception as error:
            st.error(f"Prediction failed: {str(error)}")
            return

        st.divider()
        st.subheader("Prediction Result")
        risk = "High" if fraud_score >= 0.70 else ("Medium" if fraud_score >= 0.50 else "Low")

        result_col1, result_col2, result_col3 = st.columns(3)
        result_col1.metric("Fraud Probability", f"{fraud_score * 100:.2f}%")
        result_col2.metric("Decision", decision)
        result_col3.metric("Risk Level", risk)
        st.progress(min(fraud_score, 1.0))

        claim_data = {
            "source": "OCR Scanner",
            "filename": uploaded_file.name,
            "Claim_ID": filled_fields["Claim_ID"],
            "Provider_ID": filled_fields["Provider_ID"],
            "Fraud_Score": round(fraud_score, 4),
            "Decision": decision,
        }
        block = add_blockchain_record(bc, claim_data, fraud_score)
        block_hash = get_block_hash(block)

        st.subheader("Blockchain Record")
        st.success(f"Prediction result recorded in block #{block.index}")
        st.code(
            f"""Block Index    : {block.index}
Timestamp      : {block.timestamp}
Source         : OCR Scanner
File Name      : {uploaded_file.name}
Decision       : {block.decision}
Fraud Score    : {block.fraud_score}
Claim Hash     : {block.claim_hash}
Block Hash     : {block_hash}
Previous Hash  : {block.previous_hash}"""
        )

        st.subheader("Final Values Used for Prediction")
        summary_table = pd.DataFrame({
            "Field": list(filled_fields.keys()),
            "Value": list(filled_fields.values()),
            "Source": ["Extracted" if raw_fields.get(key) is not None else "Default / Reviewed" for key in filled_fields.keys()],
        })
        st.dataframe(summary_table, use_container_width=True)
