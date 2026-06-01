import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import hashlib
import json
from datetime import datetime, timezone

import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="MediGuard — Healthcare Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────
# COLOUR PALETTE (change here to retheme)
# BG       : #FFFFFF  pure white main area
# SURFACE  : #F8FAFC  very light grey cards
# BORDER   : #E2E8F0  soft grey borders
# TEXT     : #FFFFFF  near-black body text
# MUTED    : #64748B  secondary text
# ACCENT   : #0F766E  teal — buttons, links
# DANGER   : #DC2626  red — fraud flag
# WARN     : #D97706  amber — medium risk
# SUCCESS  : #16A34A  green — legitimate
# SIDEBAR  : #1E293B  dark slate sidebar
# ─────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ══════════════════════════════
   FULL LIGHT MODE RESET
   ══════════════════════════════ */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > section,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main, .main .block-container {
    background-color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important;
    color: #0F172A !important;
}

/* ══════════════════════════════
   ALL BODY TEXT
   ══════════════════════════════ */
p, span, li, h1, h2, h3, h4, h5, div, label, a {
    color: #0F172A !important;
    font-family: 'Inter', sans-serif !important;
}

/* ══════════════════════════════
   SIDEBAR — polished dark
   ══════════════════════════════ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #0F172A !important;
    color: #F8FAFC !important;
    position: sticky !important;
    top: 0 !important;
    height: calc(100vh - 0px) !important;
    overflow-y: auto !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}

/* Force sidebar text contrast */
[data-testid="stSidebar"] *,
[data-testid="stSidebar"] *::placeholder,
[data-testid="stSidebar"] *::selection {
    color: #F8FAFC !important;
    font-family: 'Inter', sans-serif !important;
}

/* Sidebar headings and labels */
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4,
[data-testid="stSidebar"] .stMarkdown h5,
[data-testid="stSidebar"] .stMarkdown h6,
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown span,
[data-testid="stSidebar"] .stMarkdown div {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stRadio label span,
[data-testid="stSidebar"] .stRadio label div,
[data-testid="stSidebar"] .stRadio label strong {
    color: #F8FAFC !important;
}

[data-testid="stSidebar"] .stCaption p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #CBD5E1 !important;
}

[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}

/* Sidebar inputs and controls */
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input,
[data-testid="stSidebar"] .stTextArea textarea,
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #111827 !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255,255,255,0.12) !important;
}
[data-testid="stSidebar"] .stTextInput input::placeholder,
[data-testid="stSidebar"] .stNumberInput input::placeholder,
[data-testid="stSidebar"] .stTextArea textarea::placeholder {
    color: rgba(248,248,252,0.65) !important;
}
[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] .stDownloadButton > button,
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button {
    background: #0F766E !important;
    color: #FFFFFF !important;
    border: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] [data-testid="stFormSubmitButton"] > button:hover,
[data-testid="stSidebar"] .stDownloadButton > button:hover {
    background: #0D5C56 !important;
}

/* Sidebar radio buttons */
[data-testid="stSidebar"] .stRadio {
    color: #F8FAFC !important;
}

/* ══════════════════════════════
   FORM LABELS
   ══════════════════════════════ */
.stTextInput > label p,
.stNumberInput > label p,
.stSelectbox > label p,
.stFileUploader > label p,
.stTextArea > label p {
    color: #374151 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════
   TEXT INPUTS
   ══════════════════════════════ */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #FFFFFF !important;
    color: #1E293B !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    padding: 9px 12px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #0F766E !important;
    box-shadow: 0 0 0 3px rgba(15,118,110,0.10) !important;
    outline: none !important;
}

/* ══════════════════════════════
   SELECTBOX — closed + dropdown
   ══════════════════════════════ */
.stSelectbox > div > div {
    background: #FFFFFF !important;
    border: 1.5px solid #CBD5E1 !important;
    border-radius: 8px !important;
    color: #1E293B !important;
}
.stSelectbox > div > div > div,
.stSelectbox span,
.stSelectbox p {
    color: #1E293B !important;
    background: transparent !important;
}
/* Dropdown panel */
[data-baseweb="popover"],
[data-baseweb="popover"] *,
[data-baseweb="menu"],
[data-baseweb="menu"] *,
ul[role="listbox"],
ul[role="listbox"] * {
    background: #FFFFFF !important;
    color: #1E293B !important;
    font-family: 'Inter', sans-serif !important;
}
[data-baseweb="menu"] li,
[data-baseweb="menu"] [role="option"],
[data-baseweb="select"] li {
    background: #FFFFFF !important;
    color: #1E293B !important;
    font-size: 14px !important;
    padding: 9px 14px !important;
}
[data-baseweb="menu"] li:hover,
[data-baseweb="menu"] [role="option"]:hover {
    background: #F0FDFA !important;
    color: #0F766E !important;
}
[data-baseweb="menu"] [role="option"][aria-selected="true"] {
    background: #CCFBF1 !important;
    color: #0F766E !important;
    font-weight: 600 !important;
}

/* ══════════════════════════════
   DATAFRAME
   ══════════════════════════════ */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div {
    background: #FFFFFF !important;
    border-radius: 10px !important;
    border: 1px solid #E2E8F0 !important;
    overflow: hidden !important;
}
table.dataframe {
    background: #FFFFFF !important;
    color: #1E293B !important;
    border-collapse: collapse !important;
    width: 100% !important;
    font-size: 13px !important;
}
table.dataframe th {
    background: #F8FAFC !important;
    color: #475569 !important;
    font-weight: 700 !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    padding: 11px 14px !important;
    border-bottom: 2px solid #E2E8F0 !important;
    text-align: left !important;
}
table.dataframe td {
    background: #FFFFFF !important;
    color: #1E293B !important;
    padding: 10px 14px !important;
    border-bottom: 1px solid #F1F5F9 !important;
}
table.dataframe tr:hover td {
    background: #F8FAFC !important;
}

/* ══════════════════════════════
   BUTTONS
   ══════════════════════════════ */
.stButton > button,
[data-testid="stFormSubmitButton"] > button,
.stDownloadButton > button {
    background: #0F766E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
    transition: background 0.15s ease !important;
    letter-spacing: 0.1px !important;
}
.stButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover,
.stDownloadButton > button:hover {
    background: #0D5C56 !important;
    color: #FFFFFF !important;
}
.stButton > button p,
.stButton > button span,
[data-testid="stFormSubmitButton"] > button p,
.stDownloadButton > button p {
    color: #FFFFFF !important;
}

/* ══════════════════════════════
   NUMBER INPUT STEPPERS
   ══════════════════════════════ */
.stNumberInput button {
    background: #F8FAFC !important;
    color: #475569 !important;
    border: 1px solid #CBD5E1 !important;
}

/* ══════════════════════════════
   FILE UPLOADER
   ══════════════════════════════ */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] *,
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section > div,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div {
    background: #FFFFFF !important;
    color: #1E293B !important;
}
[data-testid="stFileUploader"] {
    border: 2px dashed #94A3B8 !important;
    border-radius: 10px !important;
    padding: 8px !important;
}
[data-testid="stFileUploaderDropzone"] {
    border: none !important;
}
[data-testid="stFileUploaderDropzone"] span {
    color: #64748B !important;
    font-size: 14px !important;
}
[data-testid="stFileUploaderDropzone"] small {
    color: #94A3B8 !important;
}
[data-testid="stFileUploaderDropzone"] svg {
    fill: #94A3B8 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #0F766E !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    padding: 8px 16px !important;
}
[data-testid="stFileUploaderDropzone"] button *,
[data-testid="stFileUploaderDropzone"] button p {
    color: #FFFFFF !important;
    background: transparent !important;
}
[data-testid="stFileUploaderFile"] {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
}
[data-testid="stFileUploaderFile"] * {
    color: #1E293B !important;
    background: transparent !important;
}

/* ══════════════════════════════
   ALERTS
   ══════════════════════════════ */
[data-testid="stAlert"] {
    border-radius: 8px !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span {
    color: #1E293B !important;
}

/* ══════════════════════════════
   EXPANDER
   ══════════════════════════════ */
[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #1E293B !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] summary p,
.streamlit-expanderHeader p {
    color: #1E293B !important;
}
[data-testid="stExpander"] > div:last-child {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-top: none !important;
    border-radius: 0 0 8px 8px !important;
}

/* ══════════════════════════════
   TABS
   ══════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #64748B !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    background: #0F766E !important;
    color: #FFFFFF !important;
}
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] span {
    color: #FFFFFF !important;
}

/* ══════════════════════════════
   RADIO (main page)
   ══════════════════════════════ */
.stRadio label p { color: #1E293B !important; }

/* ══════════════════════════════
   METRIC CARDS (main area)
   ══════════════════════════════ */
[data-testid="metric-container"] {
    background: #F8FAFC !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 10px !important;
    padding: 16px !important;
    box-shadow: none !important;
}
[data-testid="stMetricValue"] {
    color: #1E293B !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #64748B !important;
    font-size: 13px !important;
}

/* ══════════════════════════════
   MISC
   ══════════════════════════════ */
.stCheckbox label p     { color: #1E293B !important; }
.element-container p    { color: #1E293B !important; }
.stSpinner > div        { color: #0F766E !important; }
.stProgress > div > div { background: #0F766E !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #F1F5F9; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #94A3B8; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def card(title, value, subtitle="", color="#0F766E", icon=""):
    st.markdown(f"""
    <div style="background:#F8FAFC; border-radius:10px; padding:18px 20px;
                border:1px solid #E2E8F0; margin-bottom:6px;">
        <p style="margin:0 0 4px; font-size:12px; color:#64748B;
                  font-weight:600;">{icon} {title}</p>
        <p style="margin:0 0 4px; font-size:24px; font-weight:700;
                  color:{color}; line-height:1.2;">{value}</p>
        <p style="margin:0; font-size:12px; color:#94A3B8;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)


def section_header(title, subtitle=""):
    sub = f'<p style="margin:3px 0 0; color:#64748B; font-size:13px;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="margin:0 0 20px;">
        <h2 style="margin:0; color:#1E293B; font-size:20px; font-weight:700;">{title}</h2>
        {sub}
    </div>
    """, unsafe_allow_html=True)


def divider():
    st.markdown("<hr style='border:none; border-top:1px solid #E2E8F0; margin:16px 0;'>",
                unsafe_allow_html=True)


def label(text):
    st.markdown(f"""<p style="font-size:11px; font-weight:700; color:#94A3B8;
        letter-spacing:0.7px; text-transform:uppercase; margin:0 0 8px;">{text}</p>""",
        unsafe_allow_html=True)


def risk_badge(score):
    if score > 0.7:   return ("High Risk",   "#DC2626", "#FEF2F2")
    elif score > 0.3: return ("Medium Risk", "#D97706", "#FFFBEB")
    else:             return ("Low Risk",    "#16A34A", "#F0FDF4")


def plot_style(ax, title):
    ax.set_title(title, fontsize=12, fontweight='600', color='#1E293B', pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#E2E8F0')
    ax.spines['bottom'].set_color('#E2E8F0')
    ax.tick_params(colors='#64748B', labelsize=10)
    ax.set_facecolor('#FFFFFF')
    ax.figure.set_facecolor('#FFFFFF')


# ─────────────────────────────────────────
# BLOCKCHAIN
# ─────────────────────────────────────────
class Block:
    def __init__(self, index, claim_hash, fraud_score,
                 decision, previous_hash, model_version="ANN_v1.0"):
        self.index         = index
        self.claim_hash    = claim_hash
        self.fraud_score   = round(float(fraud_score), 4)
        self.decision      = decision
        self.timestamp     = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.model_version = model_version
        self.previous_hash = previous_hash
        self.hash          = self.compute_hash()

    def compute_hash(self):
        block_string = json.dumps({
            "index": self.index, "claim_hash": self.claim_hash,
            "fraud_score": self.fraud_score, "decision": self.decision,
            "timestamp": self.timestamp, "model_version": self.model_version,
            "previous_hash": self.previous_hash
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        self.chain.append(Block(
            index=0, claim_hash="GENESIS", fraud_score=0.0,
            decision="GENESIS", previous_hash="0"
        ))

    def get_last_block(self):
        return self.chain[-1]

    def add_record(self, claim_data: dict, fraud_score: float):
        claim_hash = hashlib.sha256(
            json.dumps(claim_data, sort_keys=True).encode()
        ).hexdigest()
        decision = "Fraudulent" if fraud_score > 0.5 else "Legitimate"
        new_block = Block(
            index=len(self.chain), claim_hash=claim_hash,
            fraud_score=fraud_score, decision=decision,
            previous_hash=self.get_last_block().hash
        )
        self.chain.append(new_block)
        return new_block

    def verify_integrity(self):
        for i in range(1, len(self.chain)):
            curr, prev = self.chain[i], self.chain[i - 1]
            if curr.previous_hash != prev.hash:
                return False, f"Chain broken at block {i}"
            recalc = hashlib.sha256(json.dumps({
                "index": curr.index, "claim_hash": curr.claim_hash,
                "fraud_score": curr.fraud_score, "decision": curr.decision,
                "timestamp": curr.timestamp, "model_version": curr.model_version,
                "previous_hash": curr.previous_hash
            }, sort_keys=True).encode()).hexdigest()
            if recalc != curr.hash:
                return False, f"Tampered block at {i}"
        return True, "All blocks verified — chain is intact"


# ─────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    model  = load_model("ann_model.keras")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_resources()

for key, val in [("blockchain", Blockchain()), ("total_processed", 0), ("total_fraud", 0)]:
    if key not in st.session_state:
        st.session_state[key] = val

bc = st.session_state.blockchain


# ─────────────────────────────────────────
# PREPROCESSING
# ─────────────────────────────────────────
def get_ref_columns():
    ref = pd.read_csv("healthcare_fraud_detection.csv")
    ref = ref.drop(['Provider_ID','Claim_ID','Claim_Submission_Date','Is_Fraud'], axis=1)
    ref['claim_to_cost_ratio'] = 0
    ref['cost_outlier_flag']   = 0
    ref['high_claim_frequency']= 0
    return pd.get_dummies(ref).columns

REF_COLS = get_ref_columns()

def preprocess_single(d):
    df = pd.DataFrame([d])
    df['claim_to_cost_ratio']  = df['Claim_Amount'] / (df['Approved_Amount'] + 1)
    df['cost_outlier_flag']    = 0
    df['high_claim_frequency'] = int(df['Number_of_Claims_Per_Provider_Monthly'].values[0] > 10)
    df = pd.get_dummies(df)
    return scaler.transform(df.reindex(columns=REF_COLS, fill_value=0))

def preprocess_batch(df_raw):
    df = df_raw.copy()
    if 'Is_Fraud' in df.columns: df = df.drop('Is_Fraud', axis=1)
    ids = df[['Claim_ID','Provider_ID']].copy() if 'Claim_ID' in df.columns else pd.DataFrame()
    df  = df.drop(columns=['Provider_ID','Claim_ID','Claim_Submission_Date'], errors='ignore')
    for c in df.select_dtypes(include='number').columns:  df[c] = df[c].fillna(df[c].mean())
    for c in df.select_dtypes(include='object').columns:  df[c] = df[c].fillna(df[c].mode()[0])
    df['claim_to_cost_ratio']  = df['Claim_Amount'] / (df['Approved_Amount'] + 1)
    Q1, Q3 = df['Claim_Amount'].quantile(0.25), df['Claim_Amount'].quantile(0.75)
    df['cost_outlier_flag']    = (df['Claim_Amount'] > Q3 + 1.5*(Q3-Q1)).astype(int)
    df['high_claim_frequency'] = (df['Number_of_Claims_Per_Provider_Monthly'] >
                                  df['Number_of_Claims_Per_Provider_Monthly'].quantile(0.90)).astype(int)
    df = pd.get_dummies(df)
    return scaler.transform(df.reindex(columns=REF_COLS, fill_value=0)), ids


# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    # Logo — use write() so global <p> CSS override doesn't swallow the colour
    st.write("🛡️  **MediGuard**")
    st.write("Fraud Detection System")

    st.markdown("---")

    page = st.radio("nav", [
        "🏠  Dashboard",
        "📋  Submit Claim",
        "📂  Bulk Analysis",
        "🔗  Blockchain Ledger",
        "📄  Document Scanner"
    ], label_visibility="collapsed")

    st.markdown("---")

    is_valid, _ = bc.verify_integrity()

    st.write("**SYSTEM STATUS**")
    st.write(f"Blocks: **{len(bc.chain)}**")
    st.write(f"Processed: **{st.session_state.total_processed}**")
    st.write(f"Flagged: **{st.session_state.total_fraud}**")
    st.write(f"Chain: **{'✅ OK' if is_valid else '❌ ERR'}**")
    st.caption(f"ANN v1.0  ·  {datetime.now().strftime('%d %b %Y')}")


# ══════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════
if page == "🏠  Dashboard":

    section_header("Dashboard",
                   "Healthcare claims fraud detection — neural network + blockchain audit trail.")

    c1, c2, c3, c4 = st.columns(4)
    with c1: card("Claims Processed", str(st.session_state.total_processed), "This session", "#0F766E", "📊")
    with c2: card("Fraud Flagged",    str(st.session_state.total_fraud),     "Above threshold", "#DC2626", "🚨")
    with c3:
        rate = (f"{st.session_state.total_fraud/st.session_state.total_processed*100:.1f}%"
                if st.session_state.total_processed > 0 else "—")
        card("Fraud Rate", rate, "Of all processed", "#D97706", "📈")
    with c4: card("Chain Blocks", str(len(bc.chain)), "Immutable records", "#16A34A", "🔗")

    divider()
    label("How it works")

    st.markdown("""
    <div style="background:#F8FAFC; border-radius:10px; padding:20px 24px;
                border:1px solid #E2E8F0; margin-bottom:16px;">
        <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:16px; text-align:center;">
            <div>
                <div style="font-size:24px; margin-bottom:8px;">📥</div>
                <p style="font-weight:600; color:#1E293B; font-size:13px; margin:0 0 3px;">1. Submit</p>
                <p style="color:#64748B; font-size:12px; margin:0; line-height:1.5;">
                    Enter a claim or upload a CSV batch
                </p>
            </div>
            <div>
                <div style="font-size:24px; margin-bottom:8px;">🤖</div>
                <p style="font-weight:600; color:#1E293B; font-size:13px; margin:0 0 3px;">2. Analyse</p>
                <p style="color:#64748B; font-size:12px; margin:0; line-height:1.5;">
                    Neural network scores fraud probability
                </p>
            </div>
            <div>
                <div style="font-size:24px; margin-bottom:8px;">🔗</div>
                <p style="font-weight:600; color:#1E293B; font-size:13px; margin:0 0 3px;">3. Record</p>
                <p style="color:#64748B; font-size:12px; margin:0; line-height:1.5;">
                    Decision hashed and stored on-chain
                </p>
            </div>
            <div>
                <div style="font-size:24px; margin-bottom:8px;">✅</div>
                <p style="font-weight:600; color:#1E293B; font-size:13px; margin:0 0 3px;">4. Audit</p>
                <p style="color:#64748B; font-size:12px; margin:0; line-height:1.5;">
                    Verify any record by block hash
                </p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    divider()
    label("Model Performance Summary")

    try:
        summary = pd.read_csv("full_model_summary.csv")
        styled = (summary.style
            .highlight_max(subset=['Accuracy','Recall (Fraud)','F1 (Fraud)','ROC-AUC'],
                           color='#CCFBF1')
            .format({'Accuracy':'{:.4f}','Recall (Fraud)':'{:.4f}',
                     'F1 (Fraud)':'{:.4f}','ROC-AUC':'{:.4f}'})
            .set_table_styles([
                {'selector':'th','props':[
                    ('background','#F8FAFC'),('color','#475569'),
                    ('font-weight','700'),('font-size','11px'),
                    ('text-transform','uppercase'),('letter-spacing','0.5px'),
                    ('padding','11px 14px'),('border-bottom','2px solid #E2E8F0'),
                    ('text-align','left')]},
                {'selector':'td','props':[
                    ('background','#FFFFFF'),('color','#1E293B'),
                    ('padding','10px 14px'),
                    ('border-bottom','1px solid #F1F5F9'),('font-size','13px')]},
            ]))
        st.write(styled.to_html(), unsafe_allow_html=True)
    except:
        st.info("Run ann_model.py first to generate model summary.")


# ══════════════════════════════════════════
# PAGE 2 — SUBMIT CLAIM
# ══════════════════════════════════════════
elif page == "📋  Submit Claim":

    section_header("Submit a Claim",
                   "Enter claim details to run fraud analysis and log the result.")

    with st.form("claim_form"):
        label("Claim Details")
        col1, col2, col3 = st.columns(3)

        with col1:
            claim_id       = st.text_input("Claim ID",    value="CLM-0001")
            provider_id    = st.text_input("Provider ID", value="PRV-0001")
            patient_age    = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
            patient_gender = st.selectbox("Patient Gender", ["Male","Female"])
            insurance_type = st.selectbox("Insurance Type", ["Private","Government","Medicaid","Self-Pay"])

        with col2:
            claim_amount    = st.number_input("Claim Amount ($)",    min_value=0.0, value=5000.0, step=100.0)
            approved_amount = st.number_input("Approved Amount ($)", min_value=0.0, value=4500.0, step=100.0)
            claim_status    = st.selectbox("Claim Status", ["Approved","Pending","Rejected"])
            days_between    = st.number_input("Days Between Service and Claim", min_value=0, value=5)
            num_claims      = st.number_input("Claims Per Provider Monthly",    min_value=0, value=10)

        with col3:
            diagnosis_code     = st.text_input("Diagnosis Code", value="I25.10")
            procedure_code     = st.text_input("Procedure Code", value="36415")
            provider_specialty = st.selectbox("Provider Specialty",
                                   ["Cardiology","General Practice","Orthopedics",
                                    "Neurology","Oncology","Radiology"])
            patient_state      = st.text_input("Patient State", value="CA")
            visit_type         = st.selectbox("Visit Type", ["Inpatient","Outpatient","Emergency"])

        col4, col5 = st.columns(2)
        with col4:
            length_of_stay = st.number_input("Length of Stay (days)", min_value=0, value=3)
        with col5:
            prior_visits      = st.number_input("Prior Visits (12 months)", min_value=0, value=2)
            chronic_condition = st.selectbox("Chronic Condition", [0,1],
                                  format_func=lambda x: "Yes" if x else "No")

        submitted = st.form_submit_button("Run Fraud Analysis", use_container_width=True)

    if submitted:
        input_dict = {
            "Patient_Age": patient_age, "Patient_Gender": patient_gender,
            "Diagnosis_Code": diagnosis_code, "Procedure_Code": procedure_code,
            "Claim_Amount": claim_amount, "Approved_Amount": approved_amount,
            "Insurance_Type": insurance_type,
            "Days_Between_Service_and_Claim": days_between,
            "Number_of_Claims_Per_Provider_Monthly": num_claims,
            "Provider_Specialty": provider_specialty, "Patient_State": patient_state,
            "Claim_Status": claim_status, "Length_of_Stay": length_of_stay,
            "Visit_Type": visit_type, "Chronic_Condition_Flag": chronic_condition,
            "Prior_Visits_12m": prior_visits
        }

        with st.spinner("Analysing..."):
            X           = preprocess_single(input_dict)
            fraud_score = float(model.predict(X, verbose=0).flatten()[0])
            decision    = "Fraudulent" if fraud_score > 0.5 else "Legitimate"
            risk_lbl, risk_col, _ = risk_badge(fraud_score)
            block = bc.add_record(
                {"claim_id": claim_id, "provider_id": provider_id,
                 "fraud_score": round(fraud_score, 4)}, fraud_score)
            st.session_state.total_processed += 1
            if decision == "Fraudulent": st.session_state.total_fraud += 1

        divider()
        label("Result")

        r1, r2, r3 = st.columns(3)
        dec_col = "#DC2626" if decision == "Fraudulent" else "#16A34A"
        with r1: card("Fraud Score",  f"{fraud_score*100:.1f}%", "Model output", "#0F766E", "🎯")
        with r2: card("Decision",     decision, "Threshold: 0.50", dec_col,
                      "🚨" if decision == "Fraudulent" else "✅")
        with r3: card("Risk Level",   risk_lbl, "", risk_col, "⚠️")

        # Score bar
        bar_col = "#DC2626" if fraud_score > 0.7 else ("#D97706" if fraud_score > 0.3 else "#16A34A")
        st.markdown(f"""
        <div style="background:#F8FAFC; border-radius:10px; padding:18px 20px;
                    border:1px solid #E2E8F0; margin:12px 0;">
            <div style="display:flex; justify-content:space-between; margin-bottom:8px;">
                <span style="font-size:12px; font-weight:600; color:#64748B;">Fraud Probability</span>
                <span style="font-size:14px; font-weight:700; color:{bar_col};">{fraud_score*100:.1f}%</span>
            </div>
            <div style="background:#E2E8F0; border-radius:6px; height:8px; overflow:hidden;">
                <div style="width:{fraud_score*100:.1f}%; height:100%; background:{bar_col};
                            border-radius:6px;"></div>
            </div>
            <div style="display:flex; justify-content:space-between; margin-top:4px;">
                <span style="font-size:11px; color:#94A3B8;">Low (0%)</span>
                <span style="font-size:11px; color:#94A3B8;">High (100%)</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        divider()
        label("Blockchain Record")

        # Light-themed blockchain record box
        dec_txt_col = "#DC2626" if decision == "Fraudulent" else "#16A34A"
        rows_html = ""
        for k, v, vc in [
            ("Block",       f"#{block.index}",       "#1E293B"),
            ("Timestamp",   block.timestamp,          "#1E293B"),
            ("Decision",    block.decision,            dec_txt_col),
            ("Score",       str(block.fraud_score),   "#1E293B"),
            ("Claim Hash",  block.claim_hash,         "#475569"),
            ("Block Hash",  block.hash,               "#475569"),
            ("Prev Hash",   block.previous_hash,      "#475569"),
        ]:
            rows_html += f"""
            <tr>
                <td style="padding:7px 12px; color:#64748B; font-weight:600;
                           white-space:nowrap; width:110px;">{k}</td>
                <td style="padding:7px 12px; color:{vc}; word-break:break-all;">{v}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#F8FAFC; border-radius:10px; border:1px solid #E2E8F0;
                    overflow:hidden; font-family:'Courier New',monospace; font-size:12.5px;">
            <table style="width:100%; border-collapse:collapse;">
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """, unsafe_allow_html=True)

        st.success(f"Logged to blockchain — Block #{block.index}")


# ══════════════════════════════════════════
# PAGE 3 — BULK ANALYSIS
# ══════════════════════════════════════════
elif page == "📂  Bulk Analysis":

    section_header("Bulk Claim Analysis",
                   "Upload a CSV to screen multiple claims at once.")

    uploaded = st.file_uploader("Upload claims CSV file", type=["csv"],
                                help="Must match training dataset columns.")

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"File loaded — {len(df_raw):,} claims ready.")

        with st.expander("Preview first 5 rows"):
            st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("Run Detection on All Claims", use_container_width=True):
            with st.spinner("Processing..."):
                X_scaled, _   = preprocess_batch(df_raw)
                fraud_probs   = model.predict(X_scaled, verbose=0).flatten()
                decisions     = ["Fraudulent" if p > 0.5 else "Legitimate" for p in fraud_probs]
                risk_levels   = [risk_badge(p)[0] for p in fraud_probs]

                results_df = df_raw.copy()
                results_df['Fraud_Score'] = [f"{p*100:.1f}%" for p in fraud_probs]
                results_df['Decision']    = decisions
                results_df['Risk_Level']  = risk_levels

                for i in range(len(fraud_probs)):
                    cid = str(df_raw['Claim_ID'].iloc[i])    if 'Claim_ID'    in df_raw.columns else str(i)
                    pid = str(df_raw['Provider_ID'].iloc[i]) if 'Provider_ID' in df_raw.columns else "N/A"
                    bc.add_record({"claim_id": cid, "provider_id": pid,
                                   "fraud_score": round(float(fraud_probs[i]), 4)},
                                  float(fraud_probs[i]))

                fraud_count = decisions.count("Fraudulent")
                legit_count = decisions.count("Legitimate")
                total       = len(decisions)
                high_risk   = sum(1 for p in fraud_probs if p > 0.7)

                st.session_state.total_processed += total
                st.session_state.total_fraud     += fraud_count

            divider()
            k1, k2, k3, k4, k5 = st.columns(5)
            with k1: card("Total",      f"{total:,}",                    "",            "#0F766E", "📊")
            with k2: card("Fraudulent", str(fraud_count),               "",            "#DC2626", "🚨")
            with k3: card("Legitimate", str(legit_count),               "",            "#16A34A", "✅")
            with k4: card("High Risk",  str(high_risk),                 ">70%",        "#DC2626", "🔴")
            with k5: card("Fraud Rate", f"{fraud_count/total*100:.1f}%","",            "#D97706", "📈")

            divider()
            ch1, ch2 = st.columns(2)

            with ch1:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(fraud_probs, bins=30, color='#0F766E', edgecolor='white', alpha=0.85)
                ax.axvline(x=0.5, color='#DC2626', linestyle='--', linewidth=1.5, label='Threshold (0.5)')
                ax.set_xlabel("Fraud Probability", color='#64748B', fontsize=11)
                ax.set_ylabel("Claims",            color='#64748B', fontsize=11)
                ax.legend(fontsize=10)
                plot_style(ax, "Fraud Score Distribution")
                plt.tight_layout(); st.pyplot(fig)

            with ch2:
                fig, ax = plt.subplots(figsize=(6, 4))
                wedges, texts, autotexts = ax.pie(
                    [fraud_count, legit_count],
                    labels=["Fraudulent","Legitimate"],
                    colors=["#DC2626","#0F766E"],
                    autopct='%1.1f%%', startangle=90,
                    wedgeprops=dict(width=0.55)
                )
                for t in texts:     t.set_color('#1E293B'); t.set_fontsize(12)
                for a in autotexts: a.set_color('white');   a.set_fontweight('bold')
                ax.set_title("Fraudulent vs Legitimate", fontsize=12,
                             fontweight='600', color='#1E293B', pad=12)
                fig.set_facecolor('#FFFFFF')
                plt.tight_layout(); st.pyplot(fig)

            divider()
            label("Prediction Results")

            display_cols = [c for c in
                ['Claim_ID','Provider_ID','Claim_Amount','Fraud_Score','Decision','Risk_Level']
                if c in results_df.columns]
            st.dataframe(results_df[display_cols], use_container_width=True, height=300)

            st.download_button("Download Results CSV",
                               data=results_df.to_csv(index=False),
                               file_name="fraud_detection_results.csv",
                               mime="text/csv", use_container_width=True)
            st.success(f"All {total:,} claims processed and logged to blockchain.")


# ══════════════════════════════════════════
# PAGE 4 — BLOCKCHAIN LEDGER
# ══════════════════════════════════════════
elif page == "🔗  Blockchain Ledger":

    section_header("Blockchain Ledger",
                   "Immutable record of every claim decision.")

    is_valid, message = bc.verify_integrity()
    if is_valid:
        st.success(f"✅ Chain Integrity Verified — {message}")
    else:
        st.error(f"❌ Chain Compromised — {message}")

    b1, b2, b3 = st.columns(3)
    with b1: card("Total Blocks",    str(len(bc.chain)),      "Including genesis",  "#0F766E", "🧱")
    with b2: card("Claims Recorded", str(len(bc.chain) - 1), "Excludes genesis",   "#0F766E", "📋")
    with b3: card("Chain Status",    "Valid" if is_valid else "Compromised",
                  "Real-time check", "#16A34A" if is_valid else "#DC2626", "🛡️")

    divider()
    label("Look Up Block")

    col1, col2 = st.columns([4, 1])
    with col1:
        block_index = st.number_input("Block index", min_value=0,
                                      max_value=max(0, len(bc.chain)-1),
                                      value=0, label_visibility="collapsed")
    with col2:
        do_search = st.button("Retrieve", use_container_width=True)

    if do_search:
        blk = bc.chain[block_index]
        dec_col = "#DC2626" if blk.decision == "Fraudulent" else "#16A34A"
        rows_html = ""
        for k, v, vc in [
            ("Block",       str(blk.index),       "#1E293B"),
            ("Timestamp",   blk.timestamp,        "#1E293B"),
            ("Decision",    blk.decision,          dec_col),
            ("Score",       str(blk.fraud_score), "#1E293B"),
            ("Model",       blk.model_version,    "#1E293B"),
            ("Claim Hash",  blk.claim_hash,       "#475569"),
            ("Block Hash",  blk.hash,             "#475569"),
            ("Prev Hash",   blk.previous_hash,    "#475569"),
        ]:
            rows_html += f"""
            <tr>
                <td style="padding:7px 12px; color:#64748B; font-weight:600;
                           white-space:nowrap; width:110px;">{k}</td>
                <td style="padding:7px 12px; color:{vc}; word-break:break-all;">{v}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#F8FAFC; border-radius:10px; border:1px solid #E2E8F0;
                    overflow:hidden; font-family:'Courier New',monospace;
                    font-size:12.5px; margin-top:10px;">
            <table style="width:100%; border-collapse:collapse;">{rows_html}</table>
        </div>
        """, unsafe_allow_html=True)

    divider()
    label("Transaction History")

    rows = [{"Index": b.index, "Timestamp": b.timestamp, "Decision": b.decision,
             "Score": b.fraud_score,
             "Claim Hash": b.claim_hash[:22]+"...",
             "Block Hash":  b.hash[:22]+"..."}
            for b in bc.chain]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)

    divider()

    try:
        summary = pd.read_csv("full_model_summary.csv")
        label("Model Comparison")
        fig, ax = plt.subplots(figsize=(10, 4))
        x, w = np.arange(len(summary)), 0.25
        ax.bar(x-w, summary['Accuracy'],       w, label='Accuracy',       color='#0F766E', alpha=0.85)
        ax.bar(x,   summary['Recall (Fraud)'], w, label='Recall (Fraud)', color='#0D9488', alpha=0.85)
        ax.bar(x+w, summary['F1 (Fraud)'],     w, label='F1 (Fraud)',     color='#5EEAD4', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(summary['Model'], rotation=10)
        ax.set_ylim(0.85, 1.02); ax.legend(fontsize=11)
        plot_style(ax, "Accuracy vs Recall vs F1")
        plt.tight_layout(); st.pyplot(fig)
    except:
        st.info("Run ann_model.py to generate model summary.")


# ══════════════════════════════════════════
# PAGE 5 — DOCUMENT SCANNER
# ══════════════════════════════════════════
elif page == "📄  Document Scanner":
    try:
        from document_scanner import render_document_scanner
        render_document_scanner(model, scaler, bc)
    except ImportError:
        section_header("Document Scanner", "OCR-based claim document analysis")
        st.error("document_scanner.py not found in the project folder.")
    except Exception as e:
        st.error(f"Document scanner error: {str(e)}")
