import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime, timezone

import tensorflow as tf
from tensorflow.keras.models import load_model

# ─────────────────────────────────────────
# BLOCK CLASS
# ─────────────────────────────────────────
class Block:
    def __init__(self, index, claim_hash, fraud_score,
                 decision, previous_hash, model_version="ANN_v1.0"):

        self.index        = index
        self.claim_hash   = claim_hash
        self.fraud_score  = round(float(fraud_score), 4)
        self.decision     = decision
        self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.model_version = model_version
        self.previous_hash = previous_hash
        self.hash         = self.compute_hash()

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


# ─────────────────────────────────────────
# BLOCKCHAIN CLASS
# ─────────────────────────────────────────
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
        print("Genesis block created.")

    def get_last_block(self):
        return self.chain[-1]

    def add_record(self, claim_data: dict, fraud_score: float):
        # Hash the claim data (never store raw sensitive data)
        claim_string = json.dumps(claim_data, sort_keys=True)
        claim_hash   = hashlib.sha256(claim_string.encode()).hexdigest()

        decision = "Fraudulent" if fraud_score > 0.5 else "Legitimate"

        new_block = Block(
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

            # Check previous hash linkage
            if current.previous_hash != previous.hash:
                return False, f"Chain broken at block {i} — previous hash mismatch"

            # Recompute hash and compare
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
                return False, f"Tampered block detected at block {i}"

        return True, "Chain integrity verified — all blocks valid"

    def get_chain_as_dataframe(self):
        rows = []
        for block in self.chain:
            rows.append({
                "Block Index":    block.index,
                "Timestamp":      block.timestamp,
                "Decision":       block.decision,
                "Fraud Score":    block.fraud_score,
                "Claim Hash":     block.claim_hash[:20] + "...",
                "Block Hash":     block.hash[:20] + "...",
                "Previous Hash":  block.previous_hash[:20] + "..."
            })
        return pd.DataFrame(rows)


# ─────────────────────────────────────────
# STEP 1: Load ANN model and scaler
# ─────────────────────────────────────────
print("\nLoading ANN model and scaler...")
model  = load_model("ann_model.keras")
scaler = joblib.load("scaler.pkl")
print("Model and scaler loaded successfully.")

# ─────────────────────────────────────────
# STEP 2: Load dataset and preprocess
# ─────────────────────────────────────────
df = pd.read_csv("healthcare_fraud_detection.csv")

df = df.drop(['Claim_Submission_Date'], axis=1)

# Save IDs for reference before dropping
claim_ids    = df['Claim_ID'].tolist()
provider_ids = df['Provider_ID'].tolist()

df = df.drop(['Provider_ID', 'Claim_ID'], axis=1)

# Missing value imputation
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# Feature engineering
df['claim_to_cost_ratio'] = df['Claim_Amount'] / (df['Approved_Amount'] + 1)

Q1  = df['Claim_Amount'].quantile(0.25)
Q3  = df['Claim_Amount'].quantile(0.75)
IQR = Q3 - Q1
df['cost_outlier_flag'] = (df['Claim_Amount'] > Q3 + 1.5 * IQR).astype(int)

df['high_claim_frequency'] = (
    df['Number_of_Claims_Per_Provider_Monthly'] >
    df['Number_of_Claims_Per_Provider_Monthly'].quantile(0.90)
).astype(int)

# One-hot encode
df_encoded = pd.get_dummies(df)

# Separate features
X = df_encoded.drop('Is_Fraud', axis=1)

# Scale
X_scaled = scaler.transform(X)

print(f"\nDataset ready. Total claims to process: {len(X_scaled)}")

# ─────────────────────────────────────────
# STEP 3: Run ANN predictions
# ─────────────────────────────────────────
print("\nRunning ANN fraud predictions...")
fraud_probs = model.predict(X_scaled, verbose=0).flatten()
print(f"Predictions complete. Sample scores: {fraud_probs[:5].round(4)}")

# ─────────────────────────────────────────
# STEP 4: Log every prediction to blockchain
# ─────────────────────────────────────────
print("\nLogging predictions to blockchain...")
bc = Blockchain()

for i in range(len(X_scaled)):
    claim_data = {
        "claim_id":    claim_ids[i],
        "provider_id": provider_ids[i],
        "fraud_score": round(float(fraud_probs[i]), 4)
    }
    bc.add_record(claim_data, float(fraud_probs[i]))

    # Print progress every 1000 records
    if (i + 1) % 1000 == 0:
        print(f"  Logged {i + 1} / {len(X_scaled)} claims...")

print(f"\nAll {len(X_scaled)} claims logged to blockchain.")
print(f"Total blocks in chain: {len(bc.chain)} (includes genesis block)")

# ─────────────────────────────────────────
# STEP 5: Verify blockchain integrity
# ─────────────────────────────────────────
print("\n" + "="*55)
print("BLOCKCHAIN INTEGRITY CHECK")
print("="*55)
is_valid, message = bc.verify_integrity()
print(f"Result : {message}")
print(f"Valid  : {is_valid}")

# ─────────────────────────────────────────
# STEP 6: Tamper detection test
# ─────────────────────────────────────────
print("\n" + "="*55)
print("TAMPER DETECTION TEST")
print("="*55)
print("Simulating tampering on block 3...")

# Rebuild a small clean chain for tamper test
bc_test = Blockchain()
for i in range(10):
    claim_data = {
        "claim_id":    claim_ids[i],
        "provider_id": provider_ids[i],
        "fraud_score": round(float(fraud_probs[i]), 4)
    }
    bc_test.add_record(claim_data, float(fraud_probs[i]))

# Verify before tampering
is_valid_before, msg_before = bc_test.verify_integrity()
print(f"Before tamper — Valid: {is_valid_before} | {msg_before}")

# Tamper by changing the previous_hash directly (breaks the chain link)
bc_test.chain[3].previous_hash = "0000000000000000tampered0000000000000000"

# Verify after tampering
is_valid_after, msg_after = bc_test.verify_integrity()
print(f"After tamper  — Valid: {is_valid_after} | {msg_after}")

if not is_valid_after:
    print("Tamper detected correctly! Blockchain is secure.")
else:
    print("WARNING: Tamper not detected")

# ─────────────────────────────────────────
# STEP 7: Print blockchain summary table
# ─────────────────────────────────────────
print("\n" + "="*55)
print("BLOCKCHAIN LEDGER — FIRST 10 BLOCKS")
print("="*55)

# Rebuild clean chain for display
bc_clean = Blockchain()
for i in range(min(10, len(X_scaled))):
    claim_data = {
        "claim_id":    claim_ids[i],
        "provider_id": provider_ids[i],
        "fraud_score": round(float(fraud_probs[i]), 4)
    }
    bc_clean.add_record(claim_data, float(fraud_probs[i]))

chain_df = bc_clean.get_chain_as_dataframe()
print(chain_df.to_string(index=False))
chain_df.to_csv("blockchain_ledger_sample.csv", index=False)
print("\nSaved: blockchain_ledger_sample.csv")

# ─────────────────────────────────────────
# STEP 8: Fraud summary statistics
# ─────────────────────────────────────────
print("\n" + "="*55)
print("FRAUD DETECTION SUMMARY")
print("="*55)

decisions   = ["Fraudulent" if p > 0.5 else "Legitimate" for p in fraud_probs]
fraud_count = decisions.count("Fraudulent")
legit_count = decisions.count("Legitimate")
total       = len(decisions)

print(f"Total claims processed : {total}")
print(f"Flagged as Fraudulent  : {fraud_count} ({fraud_count/total*100:.1f}%)")
print(f"Flagged as Legitimate  : {legit_count} ({legit_count/total*100:.1f}%)")
print(f"\nHigh risk claims (score > 0.8) : "
      f"{sum(1 for p in fraud_probs if p > 0.8)}")
print(f"Medium risk (0.5 - 0.8)        : "
      f"{sum(1 for p in fraud_probs if 0.5 <= p <= 0.8)}")
print(f"Low risk (score < 0.5)         : "
      f"{sum(1 for p in fraud_probs if p < 0.5)}")

print("\nDone! Run app.py next (Streamlit dashboard).")