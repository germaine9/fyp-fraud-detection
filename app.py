# app_clean.py
# Streamlit FYP prototype
# Run with: streamlit run app_clean.py

import streamlit as st
import pandas as pd
import joblib
import hashlib
import json
from datetime import datetime, timezone
from tensorflow.keras.models import load_model


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Healthcare Fraud Detection System",
    page_icon="🏥",
    layout="wide"
)


# ------------------------------------------------------------
# Minimal professional styling
# ------------------------------------------------------------

st.markdown("""
<style>
/* ============================================================
   Warm clean theme v2
   White + cream + amber accent. More polished but still FYP-like.
   ============================================================ */

:root {
    --bg: #fffaf3;
    --panel: #ffffff;
    --panel-soft: #fff7ed;
    --line: #eadfd1;
    --text: #2b2118;
    --muted: #6f6258;
    --title: #1f160f;
    --amber: #c56a12;
    --amber-dark: #9a4f0e;
    --amber-soft: #fff1d6;
    --red: #b91c1c;
}

/* Hide Streamlit's black top toolbar/header for cleaner demo screenshots */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0rem !important;
}
#MainMenu, footer {
    visibility: hidden !important;
}

/* Overall app */
.stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
}

.block-container {
    max-width: 1120px;
    padding-top: 1.4rem;
    padding-bottom: 2.4rem;
}

/* Text */
h1, h2, h3 {
    color: var(--title) !important;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
    letter-spacing: -0.01em !important;
}

h1 {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 1.28rem !important;
    font-weight: 650 !important;
}

h3 {
    font-size: 1.05rem !important;
    font-weight: 650 !important;
}

p, li, label {
    color: var(--text) !important;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background: #fff7ed !important;
    border-right: 1px solid var(--line) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
    background: transparent !important;
    opacity: 1 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: var(--title) !important;
}

/* Sidebar radio options: make them look like a simple menu */
[data-testid="stSidebar"] .stRadio label {
    padding: 6px 10px !important;
    border-radius: 8px !important;
    margin-bottom: 3px !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #ffedd5 !important;
}

[data-testid="stSidebar"] [aria-checked="true"] {
    background: #fed7aa !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: var(--amber) !important;
    color: #ffffff !important;
    border: 1px solid var(--amber) !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    min-height: 38px !important;
    box-shadow: none !important;
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--amber-dark) !important;
    border-color: var(--amber-dark) !important;
    color: #ffffff !important;
}

.stButton > button *,
.stDownloadButton > button *,
[data-testid="stFormSubmitButton"] > button * {
    color: #ffffff !important;
}

/* Inputs */
.stTextInput input,
.stNumberInput input,
.stTextArea textarea {
    background: #ffffff !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 7px !important;
}

.stTextInput input:focus,
.stNumberInput input:focus,
.stTextArea textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 2px rgba(197, 106, 18, 0.14) !important;
}

/* Number input buttons */
.stNumberInput button,
.stNumberInput button * {
    background: #fff7ed !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

/* Selectboxes and dropdowns */
div[data-baseweb="select"],
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

div[data-baseweb="select"] *,
.stSelectbox *,
.stMultiSelect * {
    color: var(--text) !important;
    opacity: 1 !important;
}

div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"],
div[data-baseweb="menu"] ul,
ul[role="listbox"] {
    background: #ffffff !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
}

div[data-baseweb="popover"] *,
div[data-baseweb="menu"] *,
ul[role="listbox"] *,
li[role="option"],
li[role="option"] *,
div[role="option"],
div[role="option"] * {
    background: #ffffff !important;
    color: var(--text) !important;
    opacity: 1 !important;
}

li[role="option"]:hover,
li[role="option"]:hover *,
div[role="option"]:hover,
div[role="option"]:hover * {
    background: var(--amber-soft) !important;
    color: var(--title) !important;
}

/* File uploader */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div {
    background: #ffffff !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] *,
[data-testid="stFileUploaderDropzone"] span,
[data-testid="stFileUploaderDropzone"] small,
[data-testid="stFileUploaderDropzone"] p,
[data-testid="stFileUploaderFile"] * {
    color: var(--text) !important;
    background: transparent !important;
    opacity: 1 !important;
}

[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploaderDropzone"] button * {
    background: var(--amber) !important;
    color: #ffffff !important;
    border-color: var(--amber) !important;
}

/* Metrics / dataframe / expander */
[data-testid="metric-container"],
[data-testid="stDataFrame"],
[data-testid="stTable"],
[data-testid="stExpander"] {
    background: #ffffff !important;
    color: var(--text) !important;
    border: 1px solid var(--line) !important;
    border-radius: 9px !important;
}

[data-testid="metric-container"] {
    padding: 14px 15px !important;
}

[data-testid="metric-container"] *,
[data-testid="stDataFrame"] *,
[data-testid="stTable"] *,
[data-testid="stExpander"] * {
    color: var(--text) !important;
    opacity: 1 !important;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: var(--title) !important;
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] *,
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * {
    color: var(--muted) !important;
}

/* Custom cards */
.info-card {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-top: 3px solid var(--amber) !important;
    border-radius: 11px !important;
    padding: 18px 20px !important;
    margin-bottom: 16px !important;
    min-height: 128px !important;
    box-shadow: 0 3px 10px rgba(70, 43, 16, 0.035) !important;
}

.info-card h3 {
    margin-top: 0 !important;
    margin-bottom: 9px !important;
    font-size: 1.04rem !important;
    color: var(--title) !important;
}

.info-card p {
    color: var(--muted) !important;
    margin-bottom: 0 !important;
    line-height: 1.48 !important;
    font-size: 0.96rem !important;
}

.header-card {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-left: 5px solid var(--amber) !important;
    border-radius: 12px !important;
    padding: 22px 24px !important;
    margin-bottom: 18px !important;
    box-shadow: 0 3px 12px rgba(70, 43, 16, 0.04) !important;
}

.header-title {
    font-size: 1.75rem !important;
    font-weight: 700 !important;
    color: var(--title) !important;
    margin: 0 0 8px 0 !important;
}

.header-subtitle {
    font-size: 0.98rem !important;
    color: var(--muted) !important;
    margin: 0 !important;
    line-height: 1.55 !important;
}

.hash-box {
    background: #fff7ed !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    padding: 9px 10px !important;
    font-family: Consolas, "Courier New", monospace !important;
    font-size: 13px !important;
    word-break: break-all !important;
    color: var(--title) !important;
}

.small-muted {
    color: var(--muted) !important;
    font-size: 0.92rem !important;
}

[data-testid="stAlert"],
[data-testid="stAlert"] * {
    color: var(--text) !important;
    opacity: 1 !important;
}

.stProgress > div > div > div {
    background: var(--amber) !important;
}

/* Keep Streamlit / Material icons as icons, not text like keyboard_arrow_right */
span.material-symbols-rounded,
span.material-symbols-outlined,
span.material-icons,
span[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    font-weight: normal !important;
    font-style: normal !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    word-wrap: normal !important;
    direction: ltr !important;
    color: inherit !important;
}

/* Sidebar menu: small pill-style options */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] {
    gap: 4px !important;
}

[data-testid="stSidebar"] .stRadio label {
    width: fit-content !important;
    min-width: 130px !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    padding: 6px 12px !important;
    margin-bottom: 5px !important;
    transition: 0.12s ease-in-out !important;
}

[data-testid="stSidebar"] .stRadio label:hover {
    background: #fff1d6 !important;
    border-color: #eadfd1 !important;
}

[data-testid="stSidebar"] .stRadio label:has(input:checked) {
    background: #fed7aa !important;
    border-color: #c56a12 !important;
    font-weight: 650 !important;
}

/* hide the default radio circle so it looks more like a menu */
[data-testid="stSidebar"] .stRadio label > div:first-child {
    display: none !important;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# Blockchain simulation
# ------------------------------------------------------------

class Block:
    def __init__(self, index, claim_hash, fraud_score, decision, previous_hash, source="Manual"):
        self.index = index
        self.claim_hash = claim_hash
        self.fraud_score = round(float(fraud_score), 4)
        self.decision = decision
        self.source = source
        self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        self.previous_hash = previous_hash
        self.block_hash = self.calculate_hash()

    def calculate_hash(self):
        block_data = {
            "index": self.index,
            "claim_hash": self.claim_hash,
            "fraud_score": self.fraud_score,
            "decision": self.decision,
            "source": self.source,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = []
        self.create_genesis_block()

    def create_genesis_block(self):
        self.chain.append(
            Block(
                index=0,
                claim_hash="GENESIS_BLOCK",
                fraud_score=0.0,
                decision="GENESIS",
                previous_hash="0",
                source="System"
            )
        )

    def get_latest_block(self):
        return self.chain[-1]

    def add_record(self, claim_data, fraud_score, source="Manual"):
        claim_string = json.dumps(claim_data, sort_keys=True)
        claim_hash = hashlib.sha256(claim_string.encode()).hexdigest()
        decision = "Fraudulent" if fraud_score >= 0.5 else "Legitimate"

        block = Block(
            index=len(self.chain),
            claim_hash=claim_hash,
            fraud_score=fraud_score,
            decision=decision,
            previous_hash=self.get_latest_block().block_hash,
            source=source
        )

        self.chain.append(block)
        return block

    def verify_chain(self):
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i - 1]

            if current_block.previous_hash != previous_block.block_hash:
                return False, f"Previous hash mismatch at block {i}"

            if current_block.calculate_hash() != current_block.block_hash:
                return False, f"Block hash mismatch at block {i}"

        return True, "All stored blocks are valid"


# ------------------------------------------------------------
# Load model files
# ------------------------------------------------------------

@st.cache_resource
def load_resources():
    model = load_model("ann_model.keras")
    scaler = joblib.load("scaler.pkl")
    preprocess_info = joblib.load("preprocess_info.pkl")
    return model, scaler, preprocess_info


try:
    model, scaler, preprocess_info = load_resources()
except Exception:
    st.error("Required model files are missing. Please run main_fixed.py and train_ann_fixed.py first.")
    st.stop()


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

if "blockchain" not in st.session_state:
    st.session_state.blockchain = Blockchain()

if "total_claims" not in st.session_state:
    st.session_state.total_claims = 0

if "fraud_claims" not in st.session_state:
    st.session_state.fraud_claims = 0

if "sample_type" not in st.session_state:
    st.session_state.sample_type = "normal"

bc = st.session_state.blockchain


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

def risk_level(score):
    if score >= 0.70:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def preprocess_claims(input_df):
    df = input_df.copy()

    df = df.drop(
        columns=["Provider_ID", "Claim_ID", "Claim_Submission_Date", "Is_Fraud"],
        errors="ignore"
    )

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
    df["cost_outlier_flag"] = (
        df["Claim_Amount"] > claim_q3 + 1.5 * claim_iqr
    ).astype(int)
    df["high_claim_frequency"] = (
        df["Number_of_Claims_Per_Provider_Monthly"] > high_claim_threshold
    ).astype(int)

    df = pd.get_dummies(df)
    df = df.reindex(columns=preprocess_info["feature_columns"], fill_value=0)

    return scaler.transform(df)


def run_prediction(input_df):
    processed_data = preprocess_claims(input_df)
    scores = model.predict(processed_data, verbose=0).flatten()
    decisions = ["Fraudulent" if s >= 0.5 else "Legitimate" for s in scores]
    risks = [risk_level(s) for s in scores]
    return scores, decisions, risks


def make_claim_dict(
    patient_age,
    patient_gender,
    diagnosis_code,
    procedure_code,
    claim_amount,
    approved_amount,
    insurance_type,
    days_between,
    monthly_claims,
    provider_specialty,
    patient_state,
    claim_status,
    length_of_stay,
    visit_type,
    chronic_condition,
    prior_visits
):
    return {
        "Patient_Age": patient_age,
        "Patient_Gender": patient_gender,
        "Diagnosis_Code": diagnosis_code,
        "Procedure_Code": procedure_code,
        "Claim_Amount": claim_amount,
        "Approved_Amount": approved_amount,
        "Insurance_Type": insurance_type,
        "Days_Between_Service_and_Claim": days_between,
        "Number_of_Claims_Per_Provider_Monthly": monthly_claims,
        "Provider_Specialty": provider_specialty,
        "Patient_State": patient_state,
        "Claim_Status": claim_status,
        "Length_of_Stay": length_of_stay,
        "Visit_Type": visit_type,
        "Chronic_Condition_Flag": chronic_condition,
        "Prior_Visits_12m": prior_visits
    }


def show_block(block):
    st.write(f"**Block index:** {block.index}")
    st.write(f"**Timestamp:** {block.timestamp}")
    st.write(f"**Decision:** {block.decision}")
    st.write(f"**Fraud score:** {block.fraud_score}")
    st.write(f"**Source:** {block.source}")

    st.write("Claim hash")
    st.markdown(f'<div class="hash-box">{block.claim_hash}</div>', unsafe_allow_html=True)

    st.write("Previous hash")
    st.markdown(f'<div class="hash-box">{block.previous_hash}</div>', unsafe_allow_html=True)

    st.write("Block hash")
    st.markdown(f'<div class="hash-box">{block.block_hash}</div>', unsafe_allow_html=True)


def ledger_dataframe():
    rows = []
    for block in bc.chain:
        rows.append({
            "Index": block.index,
            "Timestamp": block.timestamp,
            "Decision": block.decision,
            "Fraud Score": block.fraud_score,
            "Source": block.source,
            "Claim Hash": block.claim_hash,
            "Previous Hash": block.previous_hash,
            "Block Hash": block.block_hash
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("Project Menu")
st.sidebar.caption("Healthcare claim review prototype")

page = st.sidebar.radio(
    "Go to",
    [
        "Home",
        "Single Claim",
        "Bulk Upload",
        "OCR Scanner",
        "Blockchain",
        "Model Results",
        "About"
    ]
)

st.sidebar.divider()
valid, _ = bc.verify_chain()
st.sidebar.write("**Current session**")
st.sidebar.write(f"Claims processed: {st.session_state.total_claims}")
st.sidebar.write(f"Fraud flagged: {st.session_state.fraud_claims}")
st.sidebar.write(f"Ledger blocks: {len(bc.chain)}")
st.sidebar.write(f"Ledger status: {'Valid' if valid else 'Invalid'}")


# ------------------------------------------------------------
# Page: Home
# ------------------------------------------------------------

if page == "Home":

    st.markdown("""
    <div class="header-card">
        <p class="header-title">Healthcare Fraud Detection System</p>
        <p class="header-subtitle">
        A final year project prototype for reviewing healthcare claims, generating
        fraud predictions, and keeping audit records for verification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="info-card">
        <h3>Single Claim</h3>
        <p>Enter one claim record and view the model prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
        <h3>Bulk Upload</h3>
        <p>Upload a CSV file and screen multiple claims together.</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="info-card">
        <h3>Blockchain Ledger</h3>
        <p>Store prediction records and verify hash integrity.</p>
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Session overview")

    total = st.session_state.total_claims
    fraud = st.session_state.fraud_claims
    rate = (fraud / total * 100) if total else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Claims Processed", total)
    m2.metric("Fraud Flagged", fraud)
    m3.metric("Fraud Rate", f"{rate:.1f}%")
    m4.metric("Ledger Blocks", len(bc.chain))

    st.divider()

    st.markdown("""
    <div class="info-card">
        <h3>Suggested demo flow</h3>
        <p>
        1. Run one normal and one suspicious claim in <b>Single Claim</b>.<br>
        2. Upload a CSV in <b>Bulk Upload</b>.<br>
        3. Show the stored records in <b>Blockchain</b>.<br>
        4. Open <b>Model Results</b> to show evaluation evidence.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------
# Page: Single Claim
# ------------------------------------------------------------

elif page == "Single Claim":

    st.title("Single Claim Prediction")
    st.write("Fill in a claim record and run the ANN model.")

    with st.container(border=True):
        st.write("Example values")
        c1, c2 = st.columns(2)
        if c1.button("Use normal example"):
            st.session_state.sample_type = "normal"
            st.rerun()
        if c2.button("Use suspicious example"):
            st.session_state.sample_type = "suspicious"
            st.rerun()

    if st.session_state.sample_type == "suspicious":
        default = {
            "claim_id": "CLM999",
            "provider_id": "PRV888",
            "patient_age": 67,
            "patient_gender": "Male",
            "claim_amount": 18000.0,
            "approved_amount": 5000.0,
            "insurance_type": "Private",
            "claim_status": "Pending",
            "diagnosis_code": "I25.10",
            "procedure_code": "99285",
            "provider_specialty": "Cardiology",
            "patient_state": "CA",
            "days_between": 1,
            "monthly_claims": 45,
            "length_of_stay": 12,
            "visit_type": "Emergency",
            "chronic_condition": 1,
            "prior_visits": 8
        }
    else:
        default = {
            "claim_id": "CLM001",
            "provider_id": "PRV001",
            "patient_age": 45,
            "patient_gender": "Female",
            "claim_amount": 4200.0,
            "approved_amount": 3900.0,
            "insurance_type": "Government",
            "claim_status": "Approved",
            "diagnosis_code": "E11.9",
            "procedure_code": "36415",
            "provider_specialty": "General Practice",
            "patient_state": "CA",
            "days_between": 7,
            "monthly_claims": 8,
            "length_of_stay": 2,
            "visit_type": "Outpatient",
            "chronic_condition": 0,
            "prior_visits": 2
        }

    with st.form("claim_form"):

        st.subheader("Claim details")

        col1, col2, col3 = st.columns(3)

        with col1:
            claim_id = st.text_input("Claim ID", value=default["claim_id"])
            provider_id = st.text_input("Provider ID", value=default["provider_id"])
            patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=default["patient_age"])
            patient_gender = st.selectbox("Patient Gender", ["Male", "Female"], index=["Male", "Female"].index(default["patient_gender"]))

        with col2:
            claim_amount = st.number_input("Claim Amount", min_value=0.0, value=default["claim_amount"])
            approved_amount = st.number_input("Approved Amount", min_value=0.0, value=default["approved_amount"])
            insurance_type = st.selectbox("Insurance Type", ["Private", "Government", "Medicaid", "Self-Pay"], index=["Private", "Government", "Medicaid", "Self-Pay"].index(default["insurance_type"]))
            claim_status = st.selectbox("Claim Status", ["Approved", "Pending", "Rejected"], index=["Approved", "Pending", "Rejected"].index(default["claim_status"]))

        with col3:
            diagnosis_code = st.text_input("Diagnosis Code", value=default["diagnosis_code"])
            procedure_code = st.text_input("Procedure Code", value=default["procedure_code"])
            provider_specialty = st.selectbox(
                "Provider Specialty",
                ["Cardiology", "General Practice", "Orthopedics", "Neurology", "Oncology", "Radiology"],
                index=["Cardiology", "General Practice", "Orthopedics", "Neurology", "Oncology", "Radiology"].index(default["provider_specialty"])
            )
            patient_state = st.text_input("Patient State", value=default["patient_state"])

        col4, col5, col6 = st.columns(3)

        with col4:
            days_between = st.number_input("Days Between Service and Claim", min_value=0, value=default["days_between"])

        with col5:
            monthly_claims = st.number_input("Claims Per Provider Monthly", min_value=0, value=default["monthly_claims"])

        with col6:
            length_of_stay = st.number_input("Length of Stay", min_value=0, value=default["length_of_stay"])

        col7, col8, col9 = st.columns(3)

        with col7:
            visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "Emergency"], index=["Inpatient", "Outpatient", "Emergency"].index(default["visit_type"]))

        with col8:
            chronic_condition = st.selectbox("Chronic Condition", [0, 1], index=[0, 1].index(default["chronic_condition"]), format_func=lambda x: "Yes" if x == 1 else "No")

        with col9:
            prior_visits = st.number_input("Prior Visits in 12 Months", min_value=0, value=default["prior_visits"])

        submitted = st.form_submit_button("Run Prediction")

    if submitted:
        claim_data = make_claim_dict(
            patient_age,
            patient_gender,
            diagnosis_code,
            procedure_code,
            claim_amount,
            approved_amount,
            insurance_type,
            days_between,
            monthly_claims,
            provider_specialty,
            patient_state,
            claim_status,
            length_of_stay,
            visit_type,
            chronic_condition,
            prior_visits
        )

        scores, decisions, risks = run_prediction(pd.DataFrame([claim_data]))

        score = float(scores[0])
        decision = decisions[0]
        risk = risks[0]

        block = bc.add_record(
            {
                "Claim_ID": claim_id,
                "Provider_ID": provider_id,
                "Fraud_Score": round(score, 4),
                "Decision": decision
            },
            score,
            source="Single Claim"
        )

        st.session_state.total_claims += 1
        if decision == "Fraudulent":
            st.session_state.fraud_claims += 1

        st.divider()
        st.subheader("Prediction result")

        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Probability", f"{score * 100:.2f}%")
        r2.metric("Decision", decision)
        r3.metric("Risk Level", risk)

        st.progress(min(score, 1.0))

        with st.expander("Blockchain record created", expanded=True):
            show_block(block)


# ------------------------------------------------------------
# Page: Bulk Upload
# ------------------------------------------------------------

elif page == "Bulk Upload":

    st.title("Bulk CSV Prediction")
    st.write("Upload a CSV file containing multiple healthcare claim records.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        raw_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded data preview")
        st.dataframe(raw_df.head(), use_container_width=True)

        if st.button("Run Bulk Prediction"):
            scores, decisions, risks = run_prediction(raw_df)

            results_df = raw_df.copy()
            results_df["Fraud_Probability"] = [round(float(score), 4) for score in scores]
            results_df["Fraud_Probability_Percentage"] = [f"{float(score) * 100:.2f}%" for score in scores]
            results_df["Decision"] = decisions
            results_df["Risk_Level"] = risks

            for i, score in enumerate(scores):
                claim_id = str(raw_df["Claim_ID"].iloc[i]) if "Claim_ID" in raw_df.columns else f"ROW_{i}"
                provider_id = str(raw_df["Provider_ID"].iloc[i]) if "Provider_ID" in raw_df.columns else "UNKNOWN"

                bc.add_record(
                    {
                        "Claim_ID": claim_id,
                        "Provider_ID": provider_id,
                        "Fraud_Score": round(float(score), 4),
                        "Decision": decisions[i]
                    },
                    float(score),
                    source="Bulk CSV"
                )

            total = len(results_df)
            fraud_count = decisions.count("Fraudulent")
            legitimate_count = decisions.count("Legitimate")

            st.session_state.total_claims += total
            st.session_state.fraud_claims += fraud_count

            st.divider()
            st.subheader("Bulk prediction summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Claims", total)
            c2.metric("Fraudulent Claims", fraud_count)
            c3.metric("Legitimate Claims", legitimate_count)

            st.subheader("Prediction results")
            st.dataframe(results_df, use_container_width=True)

            st.download_button(
                label="Download Prediction Results",
                data=results_df.to_csv(index=False),
                file_name="fraud_prediction_results.csv",
                mime="text/csv"
            )


# ------------------------------------------------------------
# Page: OCR Scanner
# ------------------------------------------------------------

elif page == "OCR Scanner":

    st.title("OCR Document Scanner")

    st.write(
        "This page loads the optional OCR module if `document_scanner.py` is available in the project folder."
    )

    try:
        from document_scanner import render_document_scanner
        render_document_scanner(model, scaler, bc)
    except ImportError:
        st.warning("document_scanner.py was not found in this folder.")
        st.info(
            "For the demo, keep this page only if the OCR scanner file is completed and tested."
        )
    except Exception as error:
        st.error(f"OCR module error: {error}")


# ------------------------------------------------------------
# Page: Blockchain
# ------------------------------------------------------------

elif page == "Blockchain":

    st.title("Blockchain Records")

    st.write(
        "This page shows the blockchain-style records created during the current session."
    )

    is_valid, message = bc.verify_chain()
    if is_valid:
        st.success(f"Integrity check passed: {message}")
    else:
        st.error(f"Integrity check failed: {message}")

    st.subheader("Ledger table")
    st.dataframe(ledger_dataframe(), use_container_width=True)

    st.divider()
    st.subheader("View a block")

    block_index = st.number_input(
        "Block index",
        min_value=0,
        max_value=len(bc.chain) - 1,
        value=0
    )

    selected_block = bc.chain[block_index]
    show_block(selected_block)

    if st.button("Verify Selected Block"):
        recalculated_hash = selected_block.calculate_hash()

        if recalculated_hash == selected_block.block_hash:
            st.success("Selected block is valid. The recalculated hash matches the stored hash.")
        else:
            st.error("Selected block is invalid. The recalculated hash does not match the stored hash.")

        st.write("Recalculated hash")
        st.markdown(f'<div class="hash-box">{recalculated_hash}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Page: Model Results
# ------------------------------------------------------------

elif page == "Model Results":

    st.title("Model Results")

    st.write("This page displays evaluation outputs generated during model training.")

    try:
        summary = pd.read_csv("full_model_summary.csv")
        st.subheader("Model comparison")
        st.dataframe(summary, use_container_width=True)
    except FileNotFoundError:
        st.info("full_model_summary.csv not found. Run train_ann_fixed.py first.")

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        try:
            st.image("confusion_matrix_ANN.png", caption="ANN Confusion Matrix", use_container_width=True)
        except Exception:
            st.info("confusion_matrix_ANN.png not found.")

    with col2:
        try:
            st.image("ann_training_history.png", caption="ANN Training History", use_container_width=True)
        except Exception:
            st.info("ann_training_history.png not found.")

    with col3:
        try:
            st.image("roc_curve_ANN.png", caption="ANN ROC Curve", use_container_width=True)
        except Exception:
            st.info("roc_curve_ANN.png not found.")


# ------------------------------------------------------------
# Page: About
# ------------------------------------------------------------

elif page == "About":

    st.title("About This Prototype")

    st.write("""
    This system is developed as a final year project prototype for healthcare fraud detection.
    It uses an Artificial Neural Network model to classify healthcare insurance claims as
    legitimate or potentially fraudulent.
    """)

    st.subheader("Main functions")

    st.write("""
    - Single claim prediction
    - Bulk CSV prediction
    - OCR-assisted document scanning
    - Model performance viewing
    - Blockchain-style record storage
    - Hash-based integrity verification
    """)

    st.subheader("Limitations")

    st.write("""
    The system is not connected to real hospital or insurance databases. The blockchain
    component is a prototype simulation that demonstrates hashing, block linking, and
    tamper-evident record keeping. It is not a deployed distributed blockchain network.
    """)
