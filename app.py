import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    page_title="Healthcare Fraud Detection",
    page_icon="🏥",
    layout="wide"
)

# ─────────────────────────────────────────
# BLOCKCHAIN CLASSES (copied from blockchain.py)
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
            "index":          self.index,
            "claim_hash":     self.claim_hash,
            "fraud_score":    self.fraud_score,
            "decision":       self.decision,
            "timestamp":      self.timestamp,
            "model_version":  self.model_version,
            "previous_hash":  self.previous_hash
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()


class Blockchain:
    def __init__(self):
        self.chain = []
        self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = Block(
            index=0,
            claim_hash="GENESIS",
            fraud_score=0.0,
            decision="GENESIS",
            previous_hash="0"
        )
        self.chain.append(genesis)

    def get_last_block(self):
        return self.chain[-1]

    def add_record(self, claim_data: dict, fraud_score: float):
        claim_string = json.dumps(claim_data, sort_keys=True)
        claim_hash   = hashlib.sha256(claim_string.encode()).hexdigest()
        decision     = "Fraudulent" if fraud_score > 0.5 else "Legitimate"
        new_block    = Block(
            index         = len(self.chain),
            claim_hash    = claim_hash,
            fraud_score   = fraud_score,
            decision      = decision,
            previous_hash = self.get_last_block().hash
        )
        self.chain.append(new_block)
        return new_block

    def verify_integrity(self):
        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]
            if current.previous_hash != previous.hash:
                return False, f"Chain broken at block {i}"
            recalculated = hashlib.sha256(json.dumps({
                "index":         current.index,
                "claim_hash":    current.claim_hash,
                "fraud_score":   current.fraud_score,
                "decision":      current.decision,
                "timestamp":     current.timestamp,
                "model_version": current.model_version,
                "previous_hash": current.previous_hash
            }, sort_keys=True).encode()).hexdigest()
            if recalculated != current.hash:
                return False, f"Tampered block at {i}"
        return True, "Chain integrity verified — all blocks valid"


# ─────────────────────────────────────────
# LOAD MODEL AND SCALER (cached)
# ─────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = load_model("ann_model.h5")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_resources()

# ─────────────────────────────────────────
# SESSION STATE — keep blockchain alive
# ─────────────────────────────────────────
if "blockchain" not in st.session_state:
    st.session_state.blockchain = Blockchain()

if "history" not in st.session_state:
    st.session_state.history = []

bc = st.session_state.blockchain

# ─────────────────────────────────────────
# PREPROCESSING FUNCTION
# ─────────────────────────────────────────
def preprocess_single(input_dict):
    df = pd.DataFrame([input_dict])

    df['claim_to_cost_ratio'] = df['Claim_Amount'] / (df['Approved_Amount'] + 1)

    q1  = df['Claim_Amount'].quantile(0.25) if len(df) > 1 else 0
    q3  = df['Claim_Amount'].quantile(0.75) if len(df) > 1 else df['Claim_Amount'].values[0]
    iqr = q3 - q1
    df['cost_outlier_flag'] = (df['Claim_Amount'] > q3 + 1.5 * iqr).astype(int)

    df['high_claim_frequency'] = (
        df['Number_of_Claims_Per_Provider_Monthly'] > 10
    ).astype(int)

    df = pd.get_dummies(df)

    # Load reference columns from training
    ref_df = pd.read_csv("healthcare_fraud_detection.csv")
    ref_df = ref_df.drop(['Provider_ID', 'Claim_ID',
                          'Claim_Submission_Date', 'Is_Fraud'], axis=1)
    ref_df['claim_to_cost_ratio']   = 0
    ref_df['cost_outlier_flag']     = 0
    ref_df['high_claim_frequency']  = 0
    ref_encoded = pd.get_dummies(ref_df)

    df = df.reindex(columns=ref_encoded.columns, fill_value=0)
    df_scaled = scaler.transform(df)
    return df_scaled


def preprocess_batch(df_raw):
    df = df_raw.copy()

    if 'Is_Fraud' in df.columns:
        df = df.drop('Is_Fraud', axis=1)

    drop_cols = [c for c in ['Provider_ID', 'Claim_ID',
                              'Claim_Submission_Date'] if c in df.columns]
    ids = df[['Claim_ID', 'Provider_ID']].copy() if \
          'Claim_ID' in df.columns else pd.DataFrame()
    df  = df.drop(columns=drop_cols, errors='ignore')

    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['claim_to_cost_ratio'] = df['Claim_Amount'] / (df['Approved_Amount'] + 1)
    Q1  = df['Claim_Amount'].quantile(0.25)
    Q3  = df['Claim_Amount'].quantile(0.75)
    IQR = Q3 - Q1
    df['cost_outlier_flag']    = (df['Claim_Amount'] > Q3 + 1.5 * IQR).astype(int)
    df['high_claim_frequency'] = (
        df['Number_of_Claims_Per_Provider_Monthly'] >
        df['Number_of_Claims_Per_Provider_Monthly'].quantile(0.90)
    ).astype(int)

    df = pd.get_dummies(df)

    ref_df = pd.read_csv("healthcare_fraud_detection.csv")
    ref_df = ref_df.drop(['Provider_ID', 'Claim_ID',
                           'Claim_Submission_Date', 'Is_Fraud'], axis=1)
    ref_df['claim_to_cost_ratio']  = 0
    ref_df['cost_outlier_flag']    = 0
    ref_df['high_claim_frequency'] = 0
    ref_encoded = pd.get_dummies(ref_df)

    df = df.reindex(columns=ref_encoded.columns, fill_value=0)
    df_scaled = scaler.transform(df)
    return df_scaled, ids


# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/hospital.png", width=60)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🏠 Home",
    "📋 Submit Single Claim",
    "📂 Upload CSV",
    "🔗 Blockchain Verification"
])

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Blocks in chain:** {len(bc.chain)}")
st.sidebar.markdown(f"**Claims processed:** {len(bc.chain) - 1}")


# ═══════════════════════════════════════════
# PAGE 1: HOME
# ═══════════════════════════════════════════
if page == "🏠 Home":
    st.title("🏥 Healthcare Fraud Detection System")
    st.markdown("**AI + Blockchain powered fraud detection dashboard**")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("### 📋 Single Claim\nAnalyse one claim manually using structured input fields.")
    with col2:
        st.info("### 📂 Bulk CSV\nUpload a CSV file to analyse multiple claims at once.")
    with col3:
        st.info("### 🔗 Blockchain\nVerify claim integrity and view the audit trail.")

    st.markdown("---")
    st.subheader("How it works")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("**Step 1**\n\nSubmit claim data via form or CSV upload")
    with c2:
        st.markdown("**Step 2**\n\nANN model analyses patterns and scores fraud probability")
    with c3:
        st.markdown("**Step 3**\n\nResult is hashed and stored on the blockchain ledger")
    with c4:
        st.markdown("**Step 4**\n\nAuditors can verify any record using the block hash")

    st.markdown("---")
    st.subheader("Model Performance Summary")
    try:
        summary = pd.read_csv("full_model_summary.csv")
        st.dataframe(summary, use_container_width=True)
    except:
        st.warning("Run ann_model.py first to generate full_model_summary.csv")


# ═══════════════════════════════════════════
# PAGE 2: SINGLE CLAIM SUBMISSION
# ═══════════════════════════════════════════
elif page == "📋 Submit Single Claim":
    st.title("📋 Single Claim Submission")
    st.markdown("Enter claim details below to analyse fraud probability.")
    st.markdown("---")

    with st.form("claim_form"):
        col1, col2 = st.columns(2)

        with col1:
            claim_id    = st.text_input("Claim ID",    value="CLM-0001")
            provider_id = st.text_input("Provider ID", value="PRV-0001")
            patient_age = st.number_input("Patient Age", min_value=0, max_value=120, value=45)
            patient_gender = st.selectbox("Patient Gender", ["Male", "Female"])
            diagnosis_code = st.text_input("Diagnosis Code", value="D001")
            procedure_code = st.text_input("Procedure Code", value="P001")
            insurance_type = st.selectbox("Insurance Type", ["Private", "Government"])

        with col2:
            claim_amount    = st.number_input("Claim Amount ($)",    min_value=0.0, value=5000.0)
            approved_amount = st.number_input("Approved Amount ($)", min_value=0.0, value=4500.0)
            claim_status    = st.selectbox("Claim Status", ["Approved", "Pending", "Rejected"])
            days_between    = st.number_input("Days Between Service and Claim", min_value=0, value=5)
            num_claims      = st.number_input("Number of Claims Per Provider Monthly", min_value=0, value=10)
            length_of_stay  = st.number_input("Length of Stay (days)", min_value=0, value=3)
            prior_visits    = st.number_input("Prior Visits (12 months)", min_value=0, value=2)

        col3, col4 = st.columns(2)
        with col3:
            provider_specialty = st.selectbox("Provider Specialty",
                ["General Practice", "Cardiology", "Orthopedics", "Neurology", "Oncology"])
            patient_state      = st.text_input("Patient State", value="CA")
        with col4:
            visit_type         = st.selectbox("Visit Type", ["Inpatient", "Outpatient", "Emergency"])
            chronic_condition  = st.selectbox("Chronic Condition Flag", [0, 1])

        submitted = st.form_submit_button("🔍 Analyse Claim", use_container_width=True)

    if submitted:
        input_dict = {
            "Patient_Age":                         patient_age,
            "Patient_Gender":                      patient_gender,
            "Diagnosis_Code":                      diagnosis_code,
            "Procedure_Code":                      procedure_code,
            "Claim_Amount":                        claim_amount,
            "Approved_Amount":                     approved_amount,
            "Insurance_Type":                      insurance_type,
            "Days_Between_Service_and_Claim":      days_between,
            "Number_of_Claims_Per_Provider_Monthly": num_claims,
            "Provider_Specialty":                  provider_specialty,
            "Patient_State":                       patient_state,
            "Claim_Status":                        claim_status,
            "Length_of_Stay":                      length_of_stay,
            "Visit_Type":                          visit_type,
            "Chronic_Condition_Flag":              chronic_condition,
            "Prior_Visits_12m":                    prior_visits
        }

        with st.spinner("Analysing claim..."):
            X_scaled     = preprocess_single(input_dict)
            fraud_score  = float(model.predict(X_scaled, verbose=0).flatten()[0])
            decision     = "Fraudulent" if fraud_score > 0.5 else "Legitimate"

            claim_data   = {"claim_id": claim_id,
                            "provider_id": provider_id,
                            "fraud_score": round(fraud_score, 4)}
            block        = bc.add_record(claim_data, fraud_score)

        st.markdown("---")
        st.subheader("Analysis Result")

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
                st.metric("Risk Level", "🔴 High Risk")
            elif fraud_score > 0.3:
                st.metric("Risk Level", "🟡 Medium Risk")
            else:
                st.metric("Risk Level", "🟢 Low Risk")

        st.markdown("---")
        st.subheader("Blockchain Record")
        st.code(f"""
Block Index    : {block.index}
Timestamp      : {block.timestamp}
Decision       : {block.decision}
Fraud Score    : {block.fraud_score}
Claim Hash     : {block.claim_hash}
Block Hash     : {block.hash}
Previous Hash  : {block.previous_hash}
        """)

        st.session_state.history.append({
            "Claim ID":     claim_id,
            "Provider ID":  provider_id,
            "Fraud Score":  f"{fraud_score*100:.1f}%",
            "Decision":     decision,
            "Block Hash":   block.hash[:20] + "..."
        })
        st.success(f"Claim logged to blockchain. Block #{block.index}")


# ═══════════════════════════════════════════
# PAGE 3: BULK CSV UPLOAD
# ═══════════════════════════════════════════
elif page == "📂 Upload CSV":
    st.title("📂 Bulk CSV Upload")
    st.markdown("Upload a CSV file containing multiple claims for batch analysis.")
    st.markdown("---")

    uploaded = st.file_uploader("Upload claims CSV", type=["csv"])

    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.success(f"File loaded: {len(df_raw)} claims found")
        st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("🔍 Run Fraud Detection", use_container_width=True):
            with st.spinner("Processing all claims..."):
                X_scaled, ids = preprocess_batch(df_raw)
                fraud_probs   = model.predict(X_scaled, verbose=0).flatten()
                decisions     = ["Fraudulent" if p > 0.5 else "Legitimate"
                                 for p in fraud_probs]

                risk_levels = []
                for p in fraud_probs:
                    if p > 0.7:   risk_levels.append("🔴 High")
                    elif p > 0.3: risk_levels.append("🟡 Medium")
                    else:         risk_levels.append("🟢 Low")

                results_df = df_raw.copy()
                results_df['Fraud_Score'] = [f"{p*100:.1f}%" for p in fraud_probs]
                results_df['Decision']    = decisions
                results_df['Risk_Level']  = risk_levels

                # Log to blockchain
                for i in range(len(fraud_probs)):
                    cid = str(df_raw['Claim_ID'].iloc[i]) \
                          if 'Claim_ID' in df_raw.columns else str(i)
                    pid = str(df_raw['Provider_ID'].iloc[i]) \
                          if 'Provider_ID' in df_raw.columns else "N/A"
                    bc.add_record(
                        {"claim_id": cid, "provider_id": pid,
                         "fraud_score": round(float(fraud_probs[i]), 4)},
                        float(fraud_probs[i])
                    )

            st.markdown("---")
            st.subheader("Results Summary")

            fraud_count = decisions.count("Fraudulent")
            legit_count = decisions.count("Legitimate")
            total       = len(decisions)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Claims",  total)
            c2.metric("Fraudulent",    fraud_count)
            c3.metric("Legitimate",    legit_count)
            c4.metric("Fraud Rate",    f"{fraud_count/total*100:.1f}%")

            st.markdown("---")
            st.subheader("Prediction Results")
            st.dataframe(results_df[['Claim_ID', 'Provider_ID',
                                      'Claim_Amount', 'Fraud_Score',
                                      'Decision', 'Risk_Level']
                                     if 'Claim_ID' in results_df.columns
                                     else results_df],
                         use_container_width=True)

            # Fraud distribution chart
            st.markdown("---")
            st.subheader("Fraud Score Distribution")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

            ax1.hist(fraud_probs, bins=30, color='steelblue', edgecolor='white')
            ax1.set_xlabel("Fraud Probability Score")
            ax1.set_ylabel("Number of Claims")
            ax1.set_title("Distribution of Fraud Scores")
            ax1.axvline(x=0.5, color='red', linestyle='--', label='Threshold (0.5)')
            ax1.legend()

            ax2.pie([fraud_count, legit_count],
                    labels=["Fraudulent", "Legitimate"],
                    colors=["#e74c3c", "#2ecc71"],
                    autopct='%1.1f%%', startangle=90)
            ax2.set_title("Fraud vs Legitimate")

            plt.tight_layout()
            st.pyplot(fig)

            # Download button
            csv_out = results_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Results CSV",
                data=csv_out,
                file_name="fraud_detection_results.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.success(f"All {total} claims logged to blockchain.")


# ═══════════════════════════════════════════
# PAGE 4: BLOCKCHAIN VERIFICATION
# ═══════════════════════════════════════════
elif page == "🔗 Blockchain Verification":
    st.title("🔗 Blockchain Verification")
    st.markdown("Verify chain integrity and inspect individual block records.")
    st.markdown("---")

    # Integrity check
    st.subheader("Chain Integrity Check")
    is_valid, message = bc.verify_integrity()

    if is_valid:
        st.success(f"✅ {message}")
    else:
        st.error(f"❌ {message}")

    col1, col2 = st.columns(2)
    col1.metric("Total Blocks",     len(bc.chain))
    col2.metric("Claims Recorded",  len(bc.chain) - 1)

    st.markdown("---")

    # Search by block index
    st.subheader("Search Block by Index")
    block_index = st.number_input("Enter Block Index",
                                   min_value=0,
                                   max_value=max(0, len(bc.chain) - 1),
                                   value=0)

    if st.button("🔍 Retrieve Block", use_container_width=True):
        block = bc.chain[block_index]
        st.code(f"""
Block Index    : {block.index}
Timestamp      : {block.timestamp}
Decision       : {block.decision}
Fraud Score    : {block.fraud_score}
Model Version  : {block.model_version}
Claim Hash     : {block.claim_hash}
Block Hash     : {block.hash}
Previous Hash  : {block.previous_hash}
        """)

    st.markdown("---")

    # Full transaction history
    st.subheader("Transaction History")
    rows = []
    for block in bc.chain:
        rows.append({
            "Index":          block.index,
            "Timestamp":      block.timestamp,
            "Decision":       block.decision,
            "Fraud Score":    block.fraud_score,
            "Claim Hash":     block.claim_hash[:20] + "...",
            "Block Hash":     block.hash[:20] + "...",
        })
    chain_df = pd.DataFrame(rows)
    st.dataframe(chain_df, use_container_width=True)

    # Model comparison chart
    st.markdown("---")
    st.subheader("Model Performance Comparison")
    try:
        summary = pd.read_csv("full_model_summary.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        x     = np.arange(len(summary))
        width = 0.2
        ax.bar(x - width, summary['Accuracy'],       width, label='Accuracy')
        ax.bar(x,         summary['Recall (Fraud)'], width, label='Recall (Fraud)')
        ax.bar(x + width, summary['F1 (Fraud)'],     width, label='F1 (Fraud)')
        ax.set_xticks(x)
        ax.set_xticklabels(summary['Model'], rotation=15)
        ax.set_ylim(0.8, 1.05)
        ax.set_title("Model Comparison")
        ax.legend()
        plt.tight_layout()
        st.pyplot(fig)
    except:
        st.warning("Run ann_model.py first to generate full_model_summary.csv")