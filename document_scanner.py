import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import json
import re
import io
from datetime import datetime, timezone
from PIL import Image

# OCR and PDF
import pytesseract
import pdfplumber

def extract_text_from_image(image_file):
    """Extract text from JPG or PNG using pytesseract."""
    try:
        image = Image.open(image_file)
        text  = pytesseract.image_to_string(image)
        return text.strip()
    except Exception as e:
        return f"ERROR: {str(e)}"


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF using pdfplumber."""
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


def parse_claim_fields(text):
    """
    Parse important claim fields from extracted text.
    Uses regex patterns to find values.
    Returns a dict with extracted fields.
    """
    fields = {
        "Claim_Amount":                          None,
        "Approved_Amount":                       None,
        "Provider_ID":                           None,
        "Diagnosis_Code":                        None,
        "Procedure_Code":                        None,
        "Patient_Age":                           None,
        "Insurance_Type":                        None,
        "Patient_Gender":                        None,
        "Claim_Status":                          None,
        "Days_Between_Service_and_Claim":        None,
        "Number_of_Claims_Per_Provider_Monthly": None,
        "Length_of_Stay":                        None,
        "Prior_Visits_12m":                      None,
        "Chronic_Condition_Flag":                None,
        "Provider_Specialty":                    None,
        "Patient_State":                         None,
        "Visit_Type":                            None,
    }

    # ── Claim Amount ──────────────────────────────────────────────
    patterns_amount = [
        r'claim\s*amount[\s:$]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'total\s*amount[\s:$]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'billed\s*amount[\s:$]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'amount\s*billed[\s:$]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'\$\s*([0-9,]+(?:\.[0-9]{1,2})?)',
    ]
    for p in patterns_amount:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            fields["Claim_Amount"] = float(m.group(1).replace(",", ""))
            break

    # ── Approved Amount ───────────────────────────────────────────
    patterns_approved = [
        r'approved\s*amount[\s:$]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'amount\s*approved[\s:$]*([0-9,]+(?:\.[0-9]{1,2})?)',
        r'paid\s*amount[\s:$]*([0-9,]+(?:\.[0-9]{1,2})?)',
    ]
    for p in patterns_approved:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            fields["Approved_Amount"] = float(m.group(1).replace(",", ""))
            break

    # ── Provider ID ───────────────────────────────────────────────
    m = re.search(r'provider\s*(?:id|no|number|#)[\s:]*([A-Z0-9\-]+)',
                  text, re.IGNORECASE)
    if m:
        fields["Provider_ID"] = m.group(1).strip()

    # ── Diagnosis Code ────────────────────────────────────────────
    m = re.search(
        r'diagnosis\s*(?:code|icd)[\s:]*([A-Z][0-9]{2,3}(?:\.[0-9]{1,4})?)',
        text, re.IGNORECASE)
    if m:
        fields["Diagnosis_Code"] = m.group(1).strip()
    else:
        # Try bare ICD-10 pattern
        m = re.search(r'\b([A-Z][0-9]{2}(?:\.[0-9]{1,4})?)\b', text)
        if m:
            fields["Diagnosis_Code"] = m.group(1).strip()

    # ── Procedure Code ────────────────────────────────────────────
    m = re.search(
        r'procedure\s*(?:code|cpt)[\s:]*([0-9]{4,5}[A-Z0-9]*)',
        text, re.IGNORECASE)
    if m:
        fields["Procedure_Code"] = m.group(1).strip()

    # ── Patient Age ───────────────────────────────────────────────
    patterns_age = [
        r'(?:patient\s*)?age[\s:]*(\d{1,3})',
        r'(\d{1,3})\s*(?:year[s]?\s*old|y/?o)',
        r'dob[\s:]*\d{1,2}[\/\-]\d{1,2}[\/\-](\d{4})',
    ]
    for p in patterns_age:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            val = int(m.group(1))
            # If it looks like a birth year, estimate age
            if val > 1900:
                val = datetime.now().year - val
            if 0 < val < 130:
                fields["Patient_Age"] = val
            break

    # ── Insurance Type ────────────────────────────────────────────
    m = re.search(
        r'insurance\s*(?:type|plan|provider)?[\s:]*(private|government|medicare|medicaid|commercial)',
        text, re.IGNORECASE)
    if m:
        raw = m.group(1).lower()
        if raw in ["medicare", "medicaid", "government"]:
            fields["Insurance_Type"] = "Government"
        else:
            fields["Insurance_Type"] = "Private"

    # ── Patient Gender ────────────────────────────────────────────
    m = re.search(r'(?:gender|sex)[\s:]*(male|female|m|f)\b',
                  text, re.IGNORECASE)
    if m:
        g = m.group(1).lower()
        fields["Patient_Gender"] = "Male" if g in ["male", "m"] else "Female"

    # ── Claim Status ──────────────────────────────────────────────
    m = re.search(r'(?:claim\s*)?status[\s:]*(approved|pending|rejected|denied)',
                  text, re.IGNORECASE)
    if m:
        fields["Claim_Status"] = m.group(1).capitalize()

    # ── Days Between Service and Claim ────────────────────────────
    m = re.search(r'days?\s*(?:between|to\s*claim|delay)[\s:]*(\d+)',
                  text, re.IGNORECASE)
    if m:
        fields["Days_Between_Service_and_Claim"] = int(m.group(1))

    # ── Number of Claims Per Provider Monthly ─────────────────────
    m = re.search(r'(?:monthly\s*)?claims?\s*(?:per\s*)?(?:provider|month)[\s:]*(\d+)',
                  text, re.IGNORECASE)
    if m:
        fields["Number_of_Claims_Per_Provider_Monthly"] = int(m.group(1))

    # ── Length of Stay ────────────────────────────────────────────
    m = re.search(r'(?:length\s*of\s*stay|los|days?\s*(?:admitted|hospitalized))[\s:]*(\d+)',
                  text, re.IGNORECASE)
    if m:
        fields["Length_of_Stay"] = int(m.group(1))

    # ── Prior Visits ──────────────────────────────────────────────
    m = re.search(r'prior\s*visits?[\s:]*(\d+)', text, re.IGNORECASE)
    if m:
        fields["Prior_Visits_12m"] = int(m.group(1))

    # ── Chronic Condition ─────────────────────────────────────────
    m = re.search(r'chronic\s*condition[\s:]*(yes|no|1|0|true|false)',
                  text, re.IGNORECASE)
    if m:
        val = m.group(1).lower()
        fields["Chronic_Condition_Flag"] = 1 if val in ["yes", "1", "true"] else 0

    # ── Provider Specialty ────────────────────────────────────────
    specialties = ["cardiology", "orthopedics", "neurology", "oncology",
                   "general practice", "radiology", "surgery", "psychiatry",
                   "dermatology", "pediatrics"]
    for spec in specialties:
        if re.search(spec, text, re.IGNORECASE):
            fields["Provider_Specialty"] = spec.title()
            break

    # ── Patient State ─────────────────────────────────────────────
    m = re.search(r'(?:state|location)[\s:]*([A-Z]{2})\b', text)
    if m:
        fields["Patient_State"] = m.group(1)

    # ── Visit Type ────────────────────────────────────────────────
    m = re.search(r'(?:visit\s*type|admission\s*type)[\s:]*(inpatient|outpatient|emergency)',
                  text, re.IGNORECASE)
    if m:
        fields["Visit_Type"] = m.group(1).capitalize()

    return fields


def fill_defaults(fields):
    """Fill missing fields with safe defaults so ANN can still run."""
    defaults = {
        "Claim_Amount":                          5000.0,
        "Approved_Amount":                       4500.0,
        "Provider_ID":                           "PRV-UNKNOWN",
        "Diagnosis_Code":                        "D001",
        "Procedure_Code":                        "P001",
        "Patient_Age":                           45,
        "Insurance_Type":                        "Private",
        "Patient_Gender":                        "Male",
        "Claim_Status":                          "Pending",
        "Days_Between_Service_and_Claim":        5,
        "Number_of_Claims_Per_Provider_Monthly": 10,
        "Length_of_Stay":                        3,
        "Prior_Visits_12m":                      2,
        "Chronic_Condition_Flag":                0,
        "Provider_Specialty":                    "General Practice",
        "Patient_State":                         "CA",
        "Visit_Type":                            "Outpatient",
    }
    filled = {}
    for key, default in defaults.items():
        filled[key] = fields.get(key) if fields.get(key) is not None else default
    return filled


def preprocess_for_ann(fields, scaler):
    """Convert extracted fields into ANN input format."""
    input_dict = {k: v for k, v in fields.items()
                  if k not in ["Provider_ID", "Claim_ID"]}

    df = pd.DataFrame([input_dict])

    # Feature engineering
    df['claim_to_cost_ratio'] = (
        df['Claim_Amount'] / (df['Approved_Amount'] + 1)
    )
    df['cost_outlier_flag']    = 0
    df['high_claim_frequency'] = (
        1 if fields.get("Number_of_Claims_Per_Provider_Monthly", 0) > 10 else 0
    )

    # One-hot encode
    df = pd.get_dummies(df)

    # Align with training columns
    ref_df = pd.read_csv("healthcare_fraud_detection.csv")
    ref_df = ref_df.drop(
        ['Provider_ID', 'Claim_ID', 'Claim_Submission_Date', 'Is_Fraud'],
        axis=1
    )
    ref_df['claim_to_cost_ratio']  = 0
    ref_df['cost_outlier_flag']    = 0
    ref_df['high_claim_frequency'] = 0
    ref_encoded = pd.get_dummies(ref_df)

    df = df.reindex(columns=ref_encoded.columns, fill_value=0)
    df_scaled = scaler.transform(df)
    return df_scaled


def render_document_scanner(model, scaler, bc):
    """
    Main function — call this from app.py to render the page.
    Pass in your loaded model, scaler, and blockchain instance.
    """

    st.title("📄 Smart Document Fraud Scanner")
    st.markdown(
        "Upload a healthcare claim document (PDF, JPG, or PNG). "
        "The system will extract claim information using OCR, "
        "run ANN fraud detection, and log the result to the blockchain."
    )
    st.markdown("---")

    # ── File Upload ───────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload claim document",
        type=["pdf", "jpg", "jpeg", "png"],
        help="Supported formats: PDF, JPG, PNG"
    )

    if not uploaded_file:
        st.info("Please upload a claim document to begin.")
        return

    file_type = uploaded_file.type
    st.success(f"File uploaded: **{uploaded_file.name}** ({file_type})")

    # ── Show preview for images ───────────────────────────────────
    if file_type in ["image/jpeg", "image/png", "image/jpg"]:
        st.subheader("Document Preview")
        image = Image.open(uploaded_file)
        st.image(image, caption=uploaded_file.name, use_column_width=True)
        uploaded_file.seek(0)

    # ── Extract text ──────────────────────────────────────────────
    st.subheader("Step 1 — Text Extraction")

    with st.spinner("Extracting text from document..."):
        if file_type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
        else:
            extracted_text = extract_text_from_image(uploaded_file)

    if extracted_text.startswith("ERROR"):
        st.error(f"Text extraction failed: {extracted_text}")
        st.warning(
            "For images: make sure Tesseract OCR is installed.\n\n"
            "Download from: https://github.com/UB-Mannheim/tesseract/wiki"
        )
        return

    if not extracted_text:
        st.warning("No text could be extracted from this document.")
        return

    with st.expander("View extracted raw text", expanded=False):
        st.text_area("Raw OCR Output", extracted_text, height=200)

    st.success(f"Text extracted successfully — {len(extracted_text)} characters")

    # ── Parse fields ──────────────────────────────────────────────
    st.subheader("Step 2 — Field Extraction")

    raw_fields    = parse_claim_fields(extracted_text)
    filled_fields = fill_defaults(raw_fields)

    # Show extracted vs default
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Extracted from document**")
        extracted_display = {
            k: v for k, v in raw_fields.items() if v is not None
        }
        if extracted_display:
            for k, v in extracted_display.items():
                st.markdown(f"- **{k}:** `{v}`")
        else:
            st.warning("No fields could be automatically extracted.")

    with col2:
        st.markdown("**Values used for prediction**")
        for k, v in filled_fields.items():
            source = "📄 extracted" if raw_fields.get(k) is not None else "⚙️ default"
            st.markdown(f"- **{k}:** `{v}` *({source})*")

    # ── Allow manual correction ───────────────────────────────────
    st.markdown("---")
    st.subheader("Step 3 — Review and Correct Fields")
    st.markdown("Adjust any incorrectly extracted values before running prediction.")

    with st.expander("Edit extracted fields", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            filled_fields["Claim_Amount"] = st.number_input(
                "Claim Amount ($)",
                value=float(filled_fields["Claim_Amount"]), min_value=0.0)
            filled_fields["Approved_Amount"] = st.number_input(
                "Approved Amount ($)",
                value=float(filled_fields["Approved_Amount"]), min_value=0.0)
            filled_fields["Patient_Age"] = st.number_input(
                "Patient Age",
                value=int(filled_fields["Patient_Age"]),
                min_value=0, max_value=120)
            filled_fields["Patient_Gender"] = st.selectbox(
                "Patient Gender",
                ["Male", "Female"],
                index=0 if filled_fields["Patient_Gender"] == "Male" else 1)
            filled_fields["Insurance_Type"] = st.selectbox(
                "Insurance Type",
                ["Private", "Government"],
                index=0 if filled_fields["Insurance_Type"] == "Private" else 1)
            filled_fields["Claim_Status"] = st.selectbox(
                "Claim Status",
                ["Approved", "Pending", "Rejected"],
                index=["Approved","Pending","Rejected"].index(
                    filled_fields["Claim_Status"])
                if filled_fields["Claim_Status"] in
                   ["Approved","Pending","Rejected"] else 1)

        with c2:
            filled_fields["Days_Between_Service_and_Claim"] = st.number_input(
                "Days Between Service and Claim",
                value=int(filled_fields["Days_Between_Service_and_Claim"]),
                min_value=0)
            filled_fields["Number_of_Claims_Per_Provider_Monthly"] = st.number_input(
                "Claims Per Provider Monthly",
                value=int(filled_fields["Number_of_Claims_Per_Provider_Monthly"]),
                min_value=0)
            filled_fields["Length_of_Stay"] = st.number_input(
                "Length of Stay (days)",
                value=int(filled_fields["Length_of_Stay"]), min_value=0)
            filled_fields["Prior_Visits_12m"] = st.number_input(
                "Prior Visits (12 months)",
                value=int(filled_fields["Prior_Visits_12m"]), min_value=0)
            filled_fields["Chronic_Condition_Flag"] = st.selectbox(
                "Chronic Condition Flag",
                [0, 1],
                index=int(filled_fields["Chronic_Condition_Flag"]))
            filled_fields["Visit_Type"] = st.selectbox(
                "Visit Type",
                ["Inpatient", "Outpatient", "Emergency"],
                index=["Inpatient","Outpatient","Emergency"].index(
                    filled_fields["Visit_Type"])
                if filled_fields["Visit_Type"] in
                   ["Inpatient","Outpatient","Emergency"] else 1)

    # ── Run prediction ────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Step 4 — Fraud Prediction")

    if st.button("🔍 Run Fraud Detection", use_container_width=True):
        with st.spinner("Running ANN prediction..."):
            try:
                X_scaled    = preprocess_for_ann(filled_fields, scaler)
                fraud_score = float(
                    model.predict(X_scaled, verbose=0).flatten()[0]
                )
                decision = "Fraudulent" if fraud_score > 0.5 else "Legitimate"

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
                return

        # ── Results ───────────────────────────────────────────────
        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fraud Probability", f"{fraud_score * 100:.1f}%")
        with col2:
            if decision == "Fraudulent":
                st.error(f"🚨 {decision}")
            else:
                st.success(f"✅ {decision}")
        with col3:
            if fraud_score > 0.7:
                risk = "🔴 High Risk"
            elif fraud_score > 0.3:
                risk = "🟡 Medium Risk"
            else:
                risk = "🟢 Low Risk"
            st.metric("Risk Level", risk)

        # Fraud probability bar
        st.markdown("**Fraud Probability Score**")
        st.progress(float(fraud_score))

        # ── Blockchain logging ────────────────────────────────────
        st.markdown("---")
        st.subheader("Blockchain Record")

        claim_data = {
            "source":       "document_scanner",
            "filename":     uploaded_file.name,
            "claim_amount": filled_fields["Claim_Amount"],
            "provider_id":  filled_fields.get("Provider_ID", "UNKNOWN"),
            "fraud_score":  round(fraud_score, 4)
        }

        with st.spinner("Logging to blockchain..."):
            block = bc.add_record(claim_data, fraud_score)

        st.success(f"Result logged to blockchain — Block #{block.index}")

        st.code(f"""
Block Index    : {block.index}
Timestamp      : {block.timestamp}
Source         : Document Scanner
File Name      : {uploaded_file.name}
Decision       : {block.decision}
Fraud Score    : {block.fraud_score}
Claim Hash     : {block.claim_hash}
Block Hash     : {block.hash}
Previous Hash  : {block.previous_hash}
        """)

        # ── Extracted fields summary ──────────────────────────────
        st.markdown("---")
        st.subheader("Extracted Fields Summary")

        summary_data = {
            "Field":  list(filled_fields.keys()),
            "Value":  list(filled_fields.values()),
            "Source": [
                "Extracted" if raw_fields.get(k) is not None else "Default"
                for k in filled_fields.keys()
            ]
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True)