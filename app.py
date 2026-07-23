# app_polished.py
# Streamlit FYP prototype — polished final version
# Run with: streamlit run app_polished.py

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import joblib
import html
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()


# ------------------------------------------------------------
# Page setup
# ------------------------------------------------------------

st.set_page_config(
    page_title="Healthcare Fraud Detection System",
    page_icon="🏥",
    layout="wide"
)


# ------------------------------------------------------------
# Styling
# ------------------------------------------------------------

st.markdown("""
<style>
/* ============================================================
   MediGuard — Warm Light Theme v3
   White + cream + amber accent. Enhanced polish.
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
    --red-soft: #fef2f2;
    --green: #166534;
    --green-soft: #f0fdf4;
}

/* Hide Streamlit chrome */
header[data-testid="stHeader"] {
    background: transparent !important;
    height: 0rem !important;
}
#MainMenu, footer { visibility: hidden !important; }

/* Overall app */
.stApp {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
}

.block-container {
    max-width: 1140px;
    padding-top: 1.4rem;
    padding-bottom: 2.4rem;
}

/* ── Typography ── */
h1, h2, h3 {
    color: var(--title) !important;
    font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
    letter-spacing: -0.01em !important;
}
h1 { font-size: 1.75rem !important; font-weight: 700 !important; }
h2 { font-size: 1.28rem !important; font-weight: 650 !important; }
h3 { font-size: 1.05rem !important; font-weight: 650 !important; }
p, li, label { color: var(--text) !important; font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif !important; }

/* ── Sidebar ── */
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
[data-testid="stSidebar"] h3 { color: var(--title) !important; }

/* Sidebar radio as pill menu */
[data-testid="stSidebar"] .stRadio [role="radiogroup"] { gap: 4px !important; }
[data-testid="stSidebar"] .stRadio label {
    width: fit-content !important;
    min-width: 140px !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 999px !important;
    padding: 6px 14px !important;
    margin-bottom: 5px !important;
    transition: all 0.12s ease-in-out !important;
    font-size: 0.93rem !important;
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
/* Hide default radio circle */
[data-testid="stSidebar"] .stRadio label > div:first-child { display: none !important; }

/* Sidebar stat items */
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    color: var(--text) !important;
}

/* ── Buttons ── */
.stButton > button,
.stDownloadButton > button,
[data-testid="stFormSubmitButton"] > button {
    background: var(--amber) !important;
    color: #ffffff !important;
    border: 1px solid var(--amber) !important;
    border-radius: 7px !important;
    font-weight: 600 !important;
    min-height: 38px !important;
    box-shadow: 0 1px 3px rgba(197, 106, 18, 0.18) !important;
    transition: all 0.12s ease !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {
    background: var(--amber-dark) !important;
    border-color: var(--amber-dark) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 6px rgba(154, 79, 14, 0.28) !important;
}
.stButton > button *,
.stDownloadButton > button *,
[data-testid="stFormSubmitButton"] > button * { color: #ffffff !important; }

/* ── Inputs ── */
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
.stNumberInput button,
.stNumberInput button * {
    background: #fff7ed !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}

/* ── Selectboxes ── */
div[data-baseweb="select"],
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    color: var(--text) !important;
    border-color: var(--line) !important;
}
div[data-baseweb="select"] *,
.stSelectbox *,
.stMultiSelect * { color: var(--text) !important; opacity: 1 !important; }

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

/* ── File uploader ── */
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

/* ── Metrics ── */
[data-testid="metric-container"] {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    padding: 16px 18px !important;
    box-shadow: 0 2px 8px rgba(70, 43, 16, 0.045) !important;
}
[data-testid="metric-container"] * { color: var(--text) !important; opacity: 1 !important; }
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * { color: var(--title) !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * { color: var(--muted) !important; font-size: 0.82rem !important; }
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] * { color: var(--muted) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
}
[data-testid="stExpander"] * { color: var(--text) !important; opacity: 1 !important; }

/* ── Dataframe / table: FORCE LIGHT THEME ── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
[data-testid="stDataFrame"] iframe {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    overflow: hidden !important;
}

[data-testid="stTable"],
[data-testid="stTable"] table {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    border-collapse: separate !important;
    border-spacing: 0 !important;
    width: 100% !important;
    color: var(--text) !important;
}
[data-testid="stTable"] th {
    background: var(--panel-soft) !important;
    color: var(--title) !important;
    font-weight: 650 !important;
    font-size: 0.83rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid var(--line) !important;
}
[data-testid="stTable"] td {
    background: #ffffff !important;
    color: var(--text) !important;
    font-size: 0.9rem !important;
    padding: 9px 14px !important;
    border-bottom: 1px solid var(--line) !important;
}
[data-testid="stTable"] tr:last-child td { border-bottom: none !important; }
[data-testid="stTable"] tr:hover td { background: var(--amber-soft) !important; }

/* Alert boxes */
[data-testid="stAlert"],
[data-testid="stAlert"] * { color: var(--text) !important; opacity: 1 !important; }

/* Progress bar */
.stProgress > div > div > div { background: var(--amber) !important; }

/* Material icons fix */
span.material-symbols-rounded,
span.material-symbols-outlined,
span.material-icons,
span[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    font-weight: normal !important; font-style: normal !important; line-height: 1 !important;
    letter-spacing: normal !important; text-transform: none !important;
    white-space: nowrap !important; word-wrap: normal !important;
    direction: ltr !important; color: inherit !important;
}

/* ── Custom components ── */

.info-card {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-top: 3px solid var(--amber) !important;
    border-radius: 11px !important;
    padding: 18px 20px !important;
    margin-bottom: 16px !important;
    min-height: 128px !important;
    box-shadow: 0 3px 10px rgba(70, 43, 16, 0.035) !important;
    transition: box-shadow 0.15s ease !important;
}
.info-card:hover {
    box-shadow: 0 5px 16px rgba(70, 43, 16, 0.07) !important;
}
.info-card h3 { margin-top: 0 !important; margin-bottom: 9px !important; font-size: 1.04rem !important; color: var(--title) !important; }
.info-card p { color: var(--muted) !important; margin-bottom: 0 !important; line-height: 1.48 !important; font-size: 0.96rem !important; }

.header-card {
    background: linear-gradient(135deg, #ffffff 0%, #fff7ed 100%) !important;
    border: 1px solid var(--line) !important;
    border-left: 5px solid var(--amber) !important;
    border-radius: 12px !important;
    padding: 24px 28px !important;
    margin-bottom: 20px !important;
    box-shadow: 0 4px 16px rgba(70, 43, 16, 0.05) !important;
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
    line-height: 1.6 !important;
}

/* Result banners */
.result-fraud {
    background: var(--red-soft) !important;
    border: 1px solid #fecaca !important;
    border-left: 5px solid var(--red) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    margin: 12px 0 !important;
}
.result-legit {
    background: var(--green-soft) !important;
    border: 1px solid #bbf7d0 !important;
    border-left: 5px solid var(--green) !important;
    border-radius: 10px !important;
    padding: 16px 20px !important;
    margin: 12px 0 !important;
}
.result-fraud p, .result-fraud span { color: var(--red) !important; font-weight: 600 !important; }
.result-legit p, .result-legit span { color: var(--green) !important; font-weight: 600 !important; }

.hash-box {
    background: #fff7ed !important;
    border: 1px solid var(--line) !important;
    border-radius: 6px !important;
    padding: 10px 12px !important;
    font-family: Consolas, "Courier New", monospace !important;
    font-size: 12.5px !important;
    word-break: break-all !important;
    color: var(--title) !important;
    line-height: 1.5 !important;
}

.small-muted {
    color: var(--muted) !important;
    font-size: 0.92rem !important;
}

.section-divider {
    border: none !important;
    border-top: 1px solid var(--line) !important;
    margin: 20px 0 !important;
}

/* Stat badge in sidebar */
.sidebar-stat {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 8px !important;
    padding: 10px 13px !important;
    margin-bottom: 6px !important;
    font-size: 0.86rem !important;
}


/* ── Home-page visual upgrade ── */
.hero-wrap {
    position: relative;
    background:
        radial-gradient(circle at top right, rgba(197, 106, 18, 0.18), transparent 32%),
        linear-gradient(135deg, #ffffff 0%, #fff7ed 58%, #fff1d6 100%);
    border: 1px solid var(--line);
    border-left: 6px solid var(--amber);
    border-radius: 18px;
    padding: 28px 30px;
    margin-bottom: 22px;
    box-shadow: 0 8px 28px rgba(70, 43, 16, 0.08);
}
.hero-kicker {
    display: inline-block;
    color: var(--amber-dark) !important;
    background: #fff1d6;
    border: 1px solid #fed7aa;
    border-radius: 999px;
    padding: 5px 12px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    color: var(--title) !important;
    font-size: 2.05rem !important;
    line-height: 1.18 !important;
    font-weight: 760 !important;
    letter-spacing: -0.035em !important;
    margin: 0 0 10px 0 !important;
}
.hero-subtitle {
    color: var(--muted) !important;
    font-size: 1.02rem !important;
    line-height: 1.62 !important;
    max-width: 820px;
    margin: 0 0 16px 0 !important;
}
.hero-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin-top: 12px;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255, 255, 255, 0.76);
    border: 1px solid var(--line);
    border-radius: 999px;
    color: var(--text) !important;
    padding: 7px 11px;
    font-size: 0.84rem;
    font-weight: 600;
}

.module-card {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 18px 18px 16px 18px;
    min-height: 168px;
    box-shadow: 0 4px 14px rgba(70, 43, 16, 0.045);
    transition: all 0.16s ease;
    margin-bottom: 8px;
}
.module-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 22px rgba(70, 43, 16, 0.085);
    border-color: #d8b894;
}
.module-icon {
    width: 42px;
    height: 42px;
    border-radius: 13px;
    background: #fff1d6;
    border: 1px solid #fed7aa;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.32rem;
    margin-bottom: 12px;
}
.module-title {
    color: var(--title) !important;
    font-size: 1.04rem !important;
    font-weight: 720 !important;
    margin: 0 0 7px 0 !important;
}
.module-text {
    color: var(--muted) !important;
    font-size: 0.92rem !important;
    line-height: 1.48 !important;
    margin: 0 !important;
}
.demo-flow {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 15px;
    padding: 18px 20px;
    box-shadow: 0 4px 14px rgba(70, 43, 16, 0.04);
}
.demo-flow h3 {
    margin-top: 0 !important;
}
.demo-step {
    display: flex;
    gap: 11px;
    align-items: flex-start;
    margin: 9px 0;
}
.demo-num {
    min-width: 26px;
    height: 26px;
    border-radius: 999px;
    background: #fff1d6;
    border: 1px solid #fed7aa;
    color: var(--amber-dark);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    font-weight: 750;
    font-size: 0.82rem;
}
.demo-step span:last-child {
    color: var(--muted);
    line-height: 1.45;
}

</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------
# ------------------------------------------------------------
# Import final blockchain implementation
# ------------------------------------------------------------

try:
    from blockchain import Blockchain, CHAIN_FILE
except Exception as error:
    st.error(
        "Could not import blockchain.py. Please make sure blockchain.py is in the same folder as this app. "
        f"Error: {error}"
    )
    st.stop()


# Load model files
# ------------------------------------------------------------

@st.cache_resource
def load_resources():
    baseline_models = joblib.load("baseline_models.pkl")

    if not isinstance(baseline_models, dict):
        raise TypeError("baseline_models.pkl should contain a dictionary of trained models.")

    if "XGBoost" not in baseline_models:
        raise KeyError("baseline_models.pkl does not contain the key 'XGBoost'.")

    model = baseline_models["XGBoost"]
    scaler = joblib.load("scaler.pkl")
    preprocess_info = joblib.load("preprocess_info.pkl")

    required_keys = [
        "numeric_cols",
        "categorical_cols",
        "numeric_means",
        "categorical_modes",
        "claim_q3",
        "claim_iqr",
        "high_claim_threshold",
        "feature_columns",
    ]
    missing_keys = [key for key in required_keys if key not in preprocess_info]
    if missing_keys:
        raise KeyError(f"preprocess_info.pkl is missing required keys: {missing_keys}")

    return model, scaler, preprocess_info


try:
    model, scaler, preprocess_info = load_resources()
except Exception as error:
    st.error("Required model files are missing or invalid. Please run the training scripts first to generate baseline_models.pkl, scaler.pkl, and preprocess_info.pkl.")
    st.exception(error)
    st.stop()


# ------------------------------------------------------------
# Session state
# ------------------------------------------------------------

# Replace old/simple in-memory blockchain objects if they exist from a previous app version.
if (
    "blockchain" not in st.session_state
    or not hasattr(st.session_state.blockchain, "verify_integrity")
    or not hasattr(st.session_state.blockchain, "add_record")
):
    st.session_state.blockchain = Blockchain(chain_file=CHAIN_FILE)
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

def safe_text(value) -> str:
    """Escape values before injecting them into custom HTML."""
    return html.escape(str(value))


def short_hash(value, length=18) -> str:
    value = str(value)
    return value[:length] + "…" if len(value) > length else value


def get_block_hash(block) -> str:
    """Support both blockchain.py blocks (.hash) and older app blocks (.block_hash)."""
    return getattr(block, "hash", getattr(block, "block_hash", "N/A"))


def compute_block_hash(block) -> str:
    """Recalculate block hash using whichever method the block provides."""
    if hasattr(block, "compute_hash"):
        return block.compute_hash()
    if hasattr(block, "calculate_hash"):
        return block.calculate_hash()
    return "N/A"


def verify_blockchain(blockchain):
    """Use blockchain.py integrity verification, with fallback for old app objects."""
    if hasattr(blockchain, "verify_integrity"):
        return blockchain.verify_integrity()
    if hasattr(blockchain, "verify_chain"):
        return blockchain.verify_chain()
    return False, "Blockchain object does not provide an integrity verification method."


def ledger_counts(blockchain):
    """Calculate counts from the blockchain ledger itself."""
    records = [block for block in blockchain.chain if getattr(block, "decision", "") != "GENESIS"]
    total_records = len(records)
    fraud_records = sum(1 for block in records if getattr(block, "decision", "") == "Fraudulent")
    fraud_rate = (fraud_records / total_records * 100) if total_records else 0.0
    return total_records, fraud_records, fraud_rate


def risk_level(score):
    if score >= 0.70:
        return "High"
    if score >= 0.50:
        return "Medium"
    return "Low"


def preprocess_claims(input_df):
    df = input_df.copy()
    df = df.drop(columns=["Provider_ID", "Claim_ID", "Claim_Submission_Date", "Is_Fraud"], errors="ignore")

    numeric_cols = preprocess_info["numeric_cols"]
    categorical_cols = preprocess_info["categorical_cols"]
    numeric_means = preprocess_info["numeric_means"]
    categorical_modes = preprocess_info["categorical_modes"]

    for col in numeric_cols:
        if col not in df.columns:
            df[col] = numeric_means[col]
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(numeric_means[col])

    for col in categorical_cols:
        if col not in df.columns:
            df[col] = categorical_modes[col]
        else:
            df[col] = df[col].fillna(categorical_modes[col]).astype(str)

    claim_q3 = preprocess_info["claim_q3"]
    claim_iqr = preprocess_info["claim_iqr"]
    high_claim_threshold = preprocess_info["high_claim_threshold"]

    df["claim_to_cost_ratio"] = df["Claim_Amount"] / (df["Approved_Amount"] + 1)
    df["cost_outlier_flag"] = (df["Claim_Amount"] > claim_q3 + 1.5 * claim_iqr).astype(int)
    df["high_claim_frequency"] = (df["Number_of_Claims_Per_Provider_Monthly"] > high_claim_threshold).astype(int)

    df = pd.get_dummies(df)
    df = df.reindex(columns=preprocess_info["feature_columns"], fill_value=0)

    return scaler.transform(df)


def run_prediction(input_df):
    processed_data = preprocess_claims(input_df)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(processed_data)[:, 1]
    else:
        scores = model.predict(processed_data)

    decisions = ["Fraudulent" if float(s) >= 0.5 else "Legitimate" for s in scores]
    risks = [risk_level(float(s)) for s in scores]
    return scores, decisions, risks


def make_claim_dict(patient_age, patient_gender, diagnosis_code, procedure_code,
                    claim_amount, approved_amount, insurance_type, days_between,
                    monthly_claims, provider_specialty, patient_state, claim_status,
                    length_of_stay, visit_type, chronic_condition, prior_visits):
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
    """Display user-relevant audit information without exposing implementation metadata."""
    block_hash = get_block_hash(block)
    nonce = getattr(block, "nonce", "N/A")

    col_a, col_b = st.columns(2)
    with col_a:
        st.write(f"**Block index:** {block.index}")
        st.write(f"**Timestamp:** {block.timestamp}")
        st.write(f"**Decision:** {block.decision}")
    with col_b:
        st.write(f"**Fraud score:** {block.fraud_score}")
        st.write(f"**Source:** {block.source}")
        st.write(f"**Nonce:** {nonce}")

    st.write("**Claim hash**")
    st.markdown(f'<div class="hash-box">{safe_text(block.claim_hash)}</div>', unsafe_allow_html=True)
    st.write("**Previous hash**")
    st.markdown(f'<div class="hash-box">{safe_text(block.previous_hash)}</div>', unsafe_allow_html=True)
    st.write("**Block hash**")
    st.markdown(f'<div class="hash-box">{safe_text(block_hash)}</div>', unsafe_allow_html=True)


# ----------------------------------------------------------------
# FIX 1: ledger_html now includes its own self-contained <style>
# so it renders correctly inside components.v1.html() iframe
# ----------------------------------------------------------------
def ledger_html():
    rows_html = ""
    for block in bc.chain:
        if block.decision == "GENESIS":
            badge = '<span class="badge-genesis">GENESIS</span>'
        elif block.decision == "Fraudulent":
            badge = '<span class="badge-fraud">Fraudulent</span>'
        else:
            badge = '<span class="badge-legit">Legitimate</span>'

        block_hash = get_block_hash(block)
        nonce = getattr(block, "nonce", "N/A")

        rows_html += f"""
        <tr>
            <td>{safe_text(block.index)}</td>
            <td>{safe_text(block.timestamp)}</td>
            <td>{badge}</td>
            <td>{safe_text(block.fraud_score)}</td>
            <td>{safe_text(block.source)}</td>
            <td>{safe_text(nonce)}</td>
            <td class="mono" title="{safe_text(block.claim_hash)}">{safe_text(short_hash(block.claim_hash))}</td>
            <td class="mono" title="{safe_text(block.previous_hash)}">{safe_text(short_hash(block.previous_hash))}</td>
            <td class="mono" title="{safe_text(block_hash)}">{safe_text(short_hash(block_hash))}</td>
        </tr>
        """

    # Self-contained HTML with inline CSS — required for components.v1.html()
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            background: #fffaf3;
            padding: 4px;
        }}
        .ledger-wrap {{
            overflow-x: auto;
            border-radius: 10px;
            border: 1px solid #eadfd1;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            background: #ffffff;
        }}
        thead tr {{
            background: #fff7ed;
        }}
        th {{
            color: #1f160f;
            font-weight: 700;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            padding: 10px 14px;
            border-bottom: 1px solid #eadfd1;
            text-align: left;
            white-space: nowrap;
        }}
        td {{
            color: #2b2118;
            padding: 9px 14px;
            border-bottom: 1px solid #f3ede5;
            vertical-align: middle;
            background: #ffffff;
            white-space: nowrap;
        }}
        tbody tr:last-child td {{ border-bottom: none; }}
        tbody tr:hover td {{ background: #fff1d6; }}
        .badge-fraud {{
            display: inline-block;
            background: #fef2f2;
            color: #b91c1c;
            border: 1px solid #fecaca;
            border-radius: 999px;
            padding: 2px 9px;
            font-size: 0.76rem;
            font-weight: 600;
        }}
        .badge-legit {{
            display: inline-block;
            background: #f0fdf4;
            color: #166534;
            border: 1px solid #bbf7d0;
            border-radius: 999px;
            padding: 2px 9px;
            font-size: 0.76rem;
            font-weight: 600;
        }}
        .badge-genesis {{
            display: inline-block;
            background: #fff1d6;
            color: #9a4f0e;
            border: 1px solid #fed7aa;
            border-radius: 999px;
            padding: 2px 9px;
            font-size: 0.76rem;
            font-weight: 600;
        }}
        .mono {{
            font-family: Consolas, "Courier New", monospace;
            font-size: 0.74rem;
            color: #6f6258;
        }}
    </style>
    </head>
    <body>
        <div class="ledger-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Index</th>
                        <th>Timestamp</th>
                        <th>Decision</th>
                        <th>Fraud Score</th>
                        <th>Source</th>
                        <th>Nonce</th>
                        <th>Claim Hash</th>
                        <th>Previous Hash</th>
                        <th>Block Hash</th>
                    </tr>
                </thead>
                <tbody>
                    {rows_html}
                </tbody>
            </table>
        </div>
    </body>
    </html>
    """


def ledger_dataframe():
    rows = []
    for block in bc.chain:
        rows.append({
            "Index": block.index,
            "Timestamp": block.timestamp,
            "Decision": block.decision,
            "Fraud Score": block.fraud_score,
            "Source": block.source,
            "Nonce": getattr(block, "nonce", "N/A"),
            "Claim Hash": block.claim_hash,
            "Previous Hash": block.previous_hash,
            "Block Hash": get_block_hash(block)
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------
# FIX 2: helper to render a pandas DataFrame as a styled HTML
# table via components.v1.html() — avoids the blank iframe bug
# ----------------------------------------------------------------
def dataframe_html(df, max_rows=None, highlight_fraud=False):
    """
    Render a DataFrame as a styled HTML table to avoid blank iframe/theme issues.

    If highlight_fraud=True and a Decision column exists:
    - Fraudulent rows are displayed with a soft red background.
    - Legitimate rows are displayed with a soft green background.
    - The Decision and Risk_Level cells are displayed as colored badges.
    """
    display_df = df.head(max_rows) if max_rows else df

    header_cells = "".join(f"<th>{safe_text(col)}</th>" for col in display_df.columns)

    body_rows = ""
    for _, row in display_df.iterrows():
        decision_value = str(row.get("Decision", "")).strip()

        if highlight_fraud and decision_value == "Fraudulent":
            row_class = "fraud-row"
        elif highlight_fraud and decision_value == "Legitimate":
            row_class = "legit-row"
        else:
            row_class = ""

        cells = ""
        for col in display_df.columns:
            value = row[col]
            value_text = str(value).strip()

            if highlight_fraud and col == "Decision" and value_text == "Fraudulent":
                cell_value = '<span class="decision-badge fraud-badge">Fraudulent</span>'
            elif highlight_fraud and col == "Decision" and value_text == "Legitimate":
                cell_value = '<span class="decision-badge legit-badge">Legitimate</span>'
            elif highlight_fraud and col == "Risk_Level" and value_text == "High":
                cell_value = '<span class="risk-badge high-risk">High</span>'
            elif highlight_fraud and col == "Risk_Level" and value_text == "Medium":
                cell_value = '<span class="risk-badge medium-risk">Medium</span>'
            elif highlight_fraud and col == "Risk_Level" and value_text == "Low":
                cell_value = '<span class="risk-badge low-risk">Low</span>'
            else:
                cell_value = safe_text(value)

            cells += f"<td>{cell_value}</td>"

        body_rows += f'<tr class="{row_class}">{cells}</tr>'

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;
            background: #fffaf3;
            padding: 4px;
        }}
        .wrap {{
            overflow-x: auto;
            border-radius: 10px;
            border: 1px solid #eadfd1;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.86rem;
            background: #ffffff;
        }}
        thead tr {{ background: #fff7ed; }}
        th {{
            color: #1f160f;
            font-weight: 700;
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            padding: 10px 14px;
            border-bottom: 1px solid #eadfd1;
            text-align: left;
            white-space: nowrap;
        }}
        td {{
            color: #2b2118;
            padding: 9px 14px;
            border-bottom: 1px solid #f3ede5;
            background: #ffffff;
            vertical-align: middle;
            white-space: nowrap;
        }}
        tbody tr:last-child td {{ border-bottom: none; }}
        tbody tr:hover td {{ background: #fff1d6; }}

        /* Highlighted Bulk Upload prediction rows */
        tr.fraud-row td {{
            background: #fef2f2;
            border-bottom: 1px solid #fecaca;
        }}
        tr.fraud-row:hover td {{
            background: #fee2e2;
        }}
        tr.legit-row td {{
            background: #f0fdf4;
            border-bottom: 1px solid #bbf7d0;
        }}
        tr.legit-row:hover td {{
            background: #dcfce7;
        }}

        .decision-badge,
        .risk-badge {{
            display: inline-block;
            border-radius: 999px;
            padding: 3px 10px;
            font-size: 0.76rem;
            font-weight: 700;
            line-height: 1.25;
        }}
        .fraud-badge {{
            background: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }}
        .legit-badge {{
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }}
        .high-risk {{
            background: #fee2e2;
            color: #b91c1c;
            border: 1px solid #fecaca;
        }}
        .medium-risk {{
            background: #fff1d6;
            color: #9a4f0e;
            border: 1px solid #fed7aa;
        }}
        .low-risk {{
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
        }}
    </style>
    </head>
    <body>
        <div class="wrap">
            <table>
                <thead><tr>{header_cells}</tr></thead>
                <tbody>{body_rows}</tbody>
            </table>
        </div>
    </body>
    </html>
    """

def estimate_table_height(n_rows, row_px=38, header_px=42, padding=16, max_px=600):
    """Compute a sensible iframe height for a given number of rows."""
    return min(header_px + n_rows * row_px + padding, max_px)


# ------------------------------------------------------------
# Navigation and display helpers
# ------------------------------------------------------------

PAGES = [
    "Home",
    "Single Claim",
    "Bulk Upload",
    "OCR Scanner",
    "Blockchain",
    "Model Results",
    "About",
]

if "current_page" not in st.session_state:
    st.session_state.current_page = "Home"


def go_to(page_name):
    """Navigate between app sections from Home buttons and keep sidebar radio in sync."""
    st.session_state.current_page = page_name
    st.rerun()


def display_optional_image(file_name, caption):
    """Display a result image if it exists; otherwise show a friendly message."""
    if Path(file_name).exists():
        st.image(file_name, caption=caption, use_container_width=True)
    else:
        st.info(f"{file_name} not found.")


# ------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------

st.sidebar.title("🏥 MediGuard")
st.sidebar.caption("Healthcare claim review prototype")

selected_page = st.sidebar.radio(
    "Go to",
    PAGES,
    index=PAGES.index(st.session_state.current_page),
)

if selected_page != st.session_state.current_page:
    st.session_state.current_page = selected_page
    st.rerun()

page = st.session_state.current_page

st.sidebar.divider()
valid, _ = verify_blockchain(bc)
total, fraud, rate = ledger_counts(bc)

st.sidebar.write("**Session stats**")
st.sidebar.markdown(f"""
<div style="font-size:0.87rem; line-height:2; color:#2b2118;">
🔢 Claims processed: <b>{total}</b><br>
🚨 Fraud flagged: <b>{fraud}</b><br>
📊 Fraud rate: <b>{rate:.1f}%</b><br>
🔗 Ledger blocks: <b>{len(bc.chain)}</b><br>

✅ Ledger status: <b>{'Valid' if valid else '⚠️ Invalid'}</b>
</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("Demo controls"):
    st.caption("Use this only before a fresh demo if you want to clear previous blockchain records.")
    reset_confirmed = st.checkbox("I want to reset the demo ledger")

    if st.button("Reset ledger", disabled=not reset_confirmed, use_container_width=True):
        try:
            Path(CHAIN_FILE).unlink(missing_ok=True)
            st.session_state.blockchain = Blockchain(chain_file=CHAIN_FILE)
            st.session_state.current_page = "Home"
            st.success("Demo ledger reset.")
            st.rerun()
        except Exception as error:
            st.error(f"Could not reset ledger: {error}")


# ------------------------------------------------------------
# Page: Home
# ------------------------------------------------------------

if page == "Home":

    st.markdown(f"""
    <div class="hero-wrap">
        <div class="hero-kicker">Final Year Project Prototype</div>
        <p class="hero-title">Healthcare Fraud Detection System</p>
        <p class="hero-subtitle">
        Review healthcare claims using the system's fixed AI fraud-screening model and store
        prediction outcomes in a tamper-evident blockchain-style audit ledger.
        </p>
        <div class="hero-badges">
            <span class="hero-badge">🧠 AI Fraud Screening</span>
            <span class="hero-badge">🔐 SHA-256 Hashing</span>
            <span class="hero-badge">⛓️ Proof-of-Work Ledger</span>
            <span class="hero-badge">📄 OCR-assisted Input</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Main menu")
    st.caption("Choose a module to start your demo flow.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">🔍</div>
            <p class="module-title">Single Claim</p>
            <p class="module-text">Enter one claim record and generate a fraud probability, decision, risk level, and blockchain record.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Single Claim", key="home_single", use_container_width=True):
            go_to("Single Claim")

    with col2:
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">📂</div>
            <p class="module-title">Bulk Upload</p>
            <p class="module-text">Upload a CSV file, screen multiple claims, view summary metrics, and download prediction results.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Bulk Upload", key="home_bulk", use_container_width=True):
            go_to("Bulk Upload")

    with col3:
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">📄</div>
            <p class="module-title">OCR Scanner</p>
            <p class="module-text">Upload claim documents and review extracted fields before running the fraud prediction workflow.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open OCR Scanner", key="home_ocr", use_container_width=True):
            go_to("OCR Scanner")

    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">⛓️</div>
            <p class="module-title">Blockchain</p>
            <p class="module-text">Inspect stored blocks, nonce values, hash links, and chain integrity status.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Blockchain", key="home_blockchain", use_container_width=True):
            go_to("Blockchain")

    with col5:
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">📊</div>
            <p class="module-title">Model Results</p>
            <p class="module-text">Review model-comparison evidence, evaluation metrics, and training output charts.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Model Results", key="home_results", use_container_width=True):
            go_to("Model Results")

    with col6:
        st.markdown("""
        <div class="module-card">
            <div class="module-icon">ℹ️</div>
            <p class="module-title">About</p>
            <p class="module-text">Summarise system functions, technical choices, limitations, and prototype scope for assessment.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open About", key="home_about", use_container_width=True):
            go_to("About")

    st.divider()

    st.subheader("Live session overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Claims Processed", total)
    m2.metric("Fraud Flagged", fraud)
    m3.metric("Fraud Rate", f"{rate:.1f}%")
    m4.metric("Ledger Blocks", len(bc.chain))

    st.divider()

    left, right = st.columns([1.15, 0.85])

    with left:
        st.markdown("""
        <div class="demo-flow">
            <h3>💡 Suggested demo flow</h3>
            <div class="demo-step"><span class="demo-num">1</span><span>Run the normal and suspicious examples in <b>Single Claim</b>.</span></div>
            <div class="demo-step"><span class="demo-num">2</span><span>Upload a CSV in <b>Bulk Upload</b> and download the prediction results.</span></div>
            <div class="demo-step"><span class="demo-num">3</span><span>Use <b>OCR Scanner</b> to show document-assisted claim extraction.</span></div>
            <div class="demo-step"><span class="demo-num">4</span><span>Open <b>Blockchain</b> to verify hash-chain integrity and inspect a block.</span></div>
            <div class="demo-step"><span class="demo-num">5</span><span>Open <b>Model Results</b> to explain how the fixed deployed model was selected.</span></div>
        </div>
        """, unsafe_allow_html=True)

    with right:
        is_valid, status_message = verify_blockchain(bc)
        st.markdown("""
        <div class="demo-flow">
            <h3>✅ Integrity status</h3>
        </div>
        """, unsafe_allow_html=True)
        if is_valid:
            st.success(status_message)
        else:
            st.error(status_message)


# ------------------------------------------------------------
# Page: Single Claim
# ------------------------------------------------------------

elif page == "Single Claim":

    st.title("Single Claim Prediction")
    st.write("Fill in a claim record and run the system's fixed fraud-detection model.")


    if st.session_state.sample_type == "suspicious":
        default = {
            "claim_id": "CLM999", "provider_id": "PRV888",
            "patient_age": 67, "patient_gender": "Male",
            "claim_amount": 18000.0, "approved_amount": 5000.0,
            "insurance_type": "Private", "claim_status": "Pending",
            "diagnosis_code": "I25.10", "procedure_code": "99285",
            "provider_specialty": "Cardiology", "patient_state": "CA",
            "days_between": 1, "monthly_claims": 45, "length_of_stay": 12,
            "visit_type": "Emergency", "chronic_condition": 1, "prior_visits": 8
        }
    else:
        default = {
            "claim_id": "CLM001", "provider_id": "PRV001",
            "patient_age": 45, "patient_gender": "Female",
            "claim_amount": 4200.0, "approved_amount": 3900.0,
            "insurance_type": "Government", "claim_status": "Approved",
            "diagnosis_code": "E11.9", "procedure_code": "36415",
            "provider_specialty": "General Practice", "patient_state": "CA",
            "days_between": 7, "monthly_claims": 8, "length_of_stay": 2,
            "visit_type": "Outpatient", "chronic_condition": 0, "prior_visits": 2
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
            insurance_type = st.selectbox("Insurance Type", ["Private", "Government", "Medicaid", "Self-Pay"],
                                          index=["Private", "Government", "Medicaid", "Self-Pay"].index(default["insurance_type"]))
            claim_status = st.selectbox("Claim Status", ["Approved", "Pending", "Rejected"],
                                        index=["Approved", "Pending", "Rejected"].index(default["claim_status"]))

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
            visit_type = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "Emergency"],
                                      index=["Inpatient", "Outpatient", "Emergency"].index(default["visit_type"]))
        with col8:
            chronic_condition = st.selectbox("Chronic Condition", [0, 1],
                                             index=[0, 1].index(default["chronic_condition"]),
                                             format_func=lambda x: "Yes" if x == 1 else "No")
        with col9:
            prior_visits = st.number_input("Prior Visits in 12 Months", min_value=0, value=default["prior_visits"])

            submitted = st.form_submit_button(
            "🔍 Run Prediction",
            use_container_width=True
        )

    # Example buttons are now displayed at the bottom of the form
    with st.container(border=True):
        st.write("**Load example claim data**")
        st.caption(
            "Select an example to automatically populate the claim fields above."
        )

        example_col1, example_col2 = st.columns(2)

        if example_col1.button(
            "✅ Use Normal Example",
            key="single_claim_normal_example",
            use_container_width=True
        ):
            st.session_state.sample_type = "normal"
            st.rerun()

        if example_col2.button(
            "🚨 Use Suspicious Example",
            key="single_claim_suspicious_example",
            use_container_width=True
        ):
            st.session_state.sample_type = "suspicious"
            st.rerun()

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

    if submitted:
        claim_data = make_claim_dict(
            patient_age, patient_gender, diagnosis_code, procedure_code,
            claim_amount, approved_amount, insurance_type, days_between,
            monthly_claims, provider_specialty, patient_state, claim_status,
            length_of_stay, visit_type, chronic_condition, prior_visits
        )

        scores, decisions, risks = run_prediction(pd.DataFrame([claim_data]))
        score = float(scores[0])
        decision = decisions[0]
        risk = risks[0]

        block = bc.add_record(
            {"Claim_ID": claim_id, "Provider_ID": provider_id,
             "Fraud_Score": round(score, 4), "Decision": decision},
            score, source="Single Claim"
        )


        st.divider()
        st.subheader("Prediction result")

        if decision == "Fraudulent":
            st.markdown(f'<div class="result-fraud"><p>🚨 FRAUDULENT — This claim has been flagged. Fraud probability: {score * 100:.2f}% · Risk level: {risk}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-legit"><p>✅ LEGITIMATE — This claim appears valid. Fraud probability: {score * 100:.2f}% · Risk level: {risk}</p></div>', unsafe_allow_html=True)

        r1, r2, r3 = st.columns(3)
        r1.metric("Fraud Probability", f"{score * 100:.2f}%")
        r2.metric("Decision", decision)
        r3.metric("Risk Level", risk)

        st.progress(min(score, 1.0))

        with st.expander("⛓️ Blockchain record created", expanded=True):
            show_block(block)


# ------------------------------------------------------------
# Page: Bulk Upload
# ------------------------------------------------------------

elif page == "Bulk Upload":

    st.title("Bulk CSV Prediction")
    st.write("Upload a CSV file containing multiple healthcare claim records.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as error:
            st.error(f"Could not read CSV file: {error}")
            st.stop()

        st.subheader("Uploaded data preview")

        # FIX: Use components.v1.html() with self-contained styled table
        # instead of st.dataframe() which renders blank due to theme conflicts
        preview_rows = min(5, len(raw_df))
        preview_height = estimate_table_height(preview_rows)
        components.html(dataframe_html(raw_df, max_rows=5), height=preview_height, scrolling=False)

        if st.button("▶️ Run Bulk Prediction"):
            scores, decisions, risks = run_prediction(raw_df)

            results_df = raw_df.copy()
            results_df["Fraud_Probability"] = [round(float(score), 4) for score in scores]
            results_df["Fraud_Probability_%"] = [f"{float(score) * 100:.2f}%" for score in scores]
            results_df["Decision"] = decisions
            results_df["Risk_Level"] = risks

            for i, score in enumerate(scores):
                claim_id = str(raw_df["Claim_ID"].iloc[i]) if "Claim_ID" in raw_df.columns else f"ROW_{i}"
                provider_id = str(raw_df["Provider_ID"].iloc[i]) if "Provider_ID" in raw_df.columns else "UNKNOWN"
                bc.add_record(
                    {"Claim_ID": claim_id, "Provider_ID": provider_id,
                     "Fraud_Score": round(float(score), 4), "Decision": decisions[i]},
                    float(score), source="Bulk CSV"
                )

            total_b = len(results_df)
            fraud_count = decisions.count("Fraudulent")
            legitimate_count = decisions.count("Legitimate")


            st.divider()
            st.subheader("Bulk prediction summary")

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Claims", total_b)
            c2.metric("Fraudulent Claims", fraud_count)
            c3.metric("Legitimate Claims", legitimate_count)

            st.subheader("Prediction results")

            # FIX: Use components.v1.html() for results table too
            results_height = estimate_table_height(len(results_df))
            components.html(dataframe_html(results_df, highlight_fraud=True), height=results_height, scrolling=True)

            st.download_button(
                label="⬇️ Download Prediction Results",
                data=results_df.to_csv(index=False),
                file_name="fraud_prediction_results.csv",
                mime="text/csv"
            )


# ------------------------------------------------------------
# Page: OCR Scanner
# ------------------------------------------------------------

elif page == "OCR Scanner":

    # The scanner module renders its own heading, compact OCR tip,
    # upload control, review form, and prediction workflow.
    # Keeping the guidance inside one component avoids duplicate headings
    # and prevents a large warning banner from appearing above the scanner.
    try:
        from document_scanner import render_document_scanner
        render_document_scanner(model, scaler, bc, preprocess_info=preprocess_info)
    except TypeError:
        # Backward compatibility for older document_scanner.py versions with only 3 parameters.
        try:
            from document_scanner import render_document_scanner
            render_document_scanner(model, scaler, bc)
        except Exception as error:
            st.error(f"OCR module error: {error}")
    except ImportError:
        st.warning("document_scanner.py was not found in this folder.")
        st.info("For the demo, keep this page only if the OCR scanner file is completed and tested.")
    except Exception as error:
        st.error(f"OCR module error: {error}")


# ------------------------------------------------------------
# Page: Blockchain
# ------------------------------------------------------------

elif page == "Blockchain":

    st.title("Blockchain Records")
    st.write("Records created during the current session, stored as a SHA-256 linked hash chain.")

    is_valid, message = verify_blockchain(bc)
    if is_valid:
        st.success(f"✅ Integrity check passed — {message}")
    else:
        st.error(f"❌ Integrity check failed — {message}")

    b1, b2, b3, b4 = st.columns(4)
    b1.metric("Ledger Blocks", len(bc.chain))
    b2.metric("Stored Records", total)
    b3.metric("Fraudulent Records", fraud)
    b4.metric("Fraud Rate", f"{rate:.1f}%")

    st.subheader("Ledger table")

    # FIX: Use components.v1.html() with self-contained styled HTML
    # instead of st.markdown(unsafe_allow_html=True) which was rendering raw HTML source
    ledger_height = estimate_table_height(len(bc.chain), max_px=500)
    components.html(ledger_html(), height=ledger_height, scrolling=True)

    st.download_button(
        label="⬇️ Download Ledger CSV",
        data=ledger_dataframe().to_csv(index=False),
        file_name="blockchain_ledger.csv",
        mime="text/csv"
    )

    st.divider()
    st.subheader("Inspect a block")

    block_index = st.number_input(
        "Block index",
        min_value=0,
        max_value=len(bc.chain) - 1,
        value=0
    )

    selected_block = bc.chain[block_index]

    with st.container(border=True):
        show_block(selected_block)

    if st.button("🔍 Verify Selected Block"):
        recalculated_hash = compute_block_hash(selected_block)
        stored_hash = get_block_hash(selected_block)
        if recalculated_hash == stored_hash:
            st.success("✅ Block is valid. The recalculated hash matches the stored hash.")
        else:
            st.error("❌ Block is invalid. The recalculated hash does not match the stored hash.")

        st.write("**Recalculated hash**")
        st.markdown(f'<div class="hash-box">{safe_text(recalculated_hash)}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------
# Page: Model Results
# ------------------------------------------------------------

elif page == "Model Results":

    st.title("Model Results")
    st.write("Evaluation outputs generated during model development. This page is informational: the deployed model is fixed in the backend and cannot be changed by users.")

    st.info("The deployed classifier is fixed by the developer. This page explains the evaluation evidence; it does not provide a model-selection control.")

    try:
        summary = pd.read_csv("full_model_summary.csv")
        st.subheader("Model comparison")
        # FIX: Use components.v1.html() for the model summary table as well
        summary_height = estimate_table_height(len(summary))
        components.html(dataframe_html(summary), height=summary_height, scrolling=False)
    except FileNotFoundError:
        st.info("full_model_summary.csv not found. Run main_fixed.py and train_ann_fixed.py first.")

    st.divider()

    st.subheader("Core deployed-model evaluation charts")
    col1, col2, col3 = st.columns(3)
    with col1:
        display_optional_image("confusion_matrix_XGBoost.png", "Deployed Model Confusion Matrix (XGBoost)")
    with col2:
        display_optional_image("model_comparison.png", "Model Comparison")
    with col3:
        display_optional_image("roc_curve_comparison.png", "ROC Curve Comparison")

    st.divider()
    st.subheader("Data preparation and ANN comparison charts")
    col4, col5, col6 = st.columns(3)
    with col4:
        display_optional_image("smote_comparison.png", "Class Balance Before and After SMOTE")
    with col5:
        display_optional_image("confusion_matrix_ANN.png", "ANN Confusion Matrix")
    with col6:
        display_optional_image("ann_training_history.png", "ANN Training History")

    col7, col8, col9 = st.columns(3)
    with col7:
        display_optional_image("roc_curve_ANN.png", "ANN ROC Curve")
    with col8:
        display_optional_image("cross_validation_recall.png", "Cross-Validation Recall")
    with col9:
        display_optional_image("confusion_matrix_Random Forest.png", "Random Forest Confusion Matrix")


# ------------------------------------------------------------
# Page: About
# ------------------------------------------------------------

elif page == "About":

    st.title("About This Prototype")

    st.markdown("""
    <div class="info-card">
        <h3>🏥 MediGuard — Healthcare Fraud Detection System</h3>
        <p>
        This system is developed as a final year project prototype. It uses a fixed backend
        fraud-detection classifier to classify healthcare insurance claims as legitimate or
        potentially fraudulent. The model configuration is controlled by the system developer
        and is not selectable by end users. Detailed model-comparison evidence is available on
        the Model Results page for transparency and assessment.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Main functions")
        st.write("""
        - Single claim prediction with result banners
        - Bulk CSV prediction with downloadable report
        - OCR-assisted document scanning
        - Model performance comparison and charts
        - Blockchain-style audit record storage
        - SHA-256 hash-based integrity verification
        - Nonce-based Proof-of-Work demonstration
        """)

    with col2:
        st.subheader("Limitations")
        st.write("""
        The system is not connected to real hospital or insurance databases.
        The blockchain component is a prototype simulation demonstrating
        hashing, block linking, nonce-based Proof-of-Work, and tamper-evident record keeping.
        It is not a deployed distributed blockchain network.
        """)

# End of app_polished.py
