# blockchain.py
# Blockchain-style audit ledger for the FYP prototype.
#
# Purpose:
# - Store fraud prediction records in a tamper-evident chain
# - Use SHA-256 hashing, previous-hash linking, and lightweight Proof-of-Work
# - Persist records to blockchain_data.json
#
# Note:
# This is a lightweight blockchain-style audit ledger, not a deployed
# distributed blockchain network.

import hashlib
import json
import joblib
import os
import pandas as pd
from datetime import datetime, timezone


CHAIN_FILE = "blockchain_data.json"
DIFFICULTY = 3
MODEL_VERSION = "XGBoost_v1.0"


# ============================================================
# BLOCK CLASS
# ============================================================

class Block:
    def __init__(
        self,
        index,
        claim_hash,
        fraud_score,
        decision,
        previous_hash,
        model_version=MODEL_VERSION,
        source="System",
        nonce=0,
        timestamp=None
    ):
        self.index = index
        self.claim_hash = claim_hash
        self.fraud_score = round(float(fraud_score), 4)
        self.decision = decision
        self.timestamp = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.model_version = model_version
        self.source = source
        self.previous_hash = previous_hash
        self.nonce = nonce
        self.hash = self.compute_hash()

    def compute_hash(self):
        block_data = {
            "index": self.index,
            "claim_hash": self.claim_hash,
            "fraud_score": self.fraud_score,
            "decision": self.decision,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "source": self.source,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce
        }

        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    def mine(self, difficulty=DIFFICULTY):
        target = "0" * difficulty

        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash = self.compute_hash()

        return self.hash

    def to_dict(self):
        return {
            "index": self.index,
            "claim_hash": self.claim_hash,
            "fraud_score": self.fraud_score,
            "decision": self.decision,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
            "source": self.source,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash
        }

    @classmethod
    def from_dict(cls, data):
        block = cls(
            index=data["index"],
            claim_hash=data["claim_hash"],
            fraud_score=data["fraud_score"],
            decision=data["decision"],
            previous_hash=data["previous_hash"],
            model_version=data.get("model_version", MODEL_VERSION),
            source=data.get("source", "System"),
            nonce=data.get("nonce", 0),
            timestamp=data.get("timestamp")
        )

        block.hash = data["hash"]
        return block


# ============================================================
# BLOCKCHAIN CLASS
# ============================================================

class Blockchain:
    def __init__(self, difficulty=DIFFICULTY, chain_file=CHAIN_FILE):
        self.difficulty = difficulty
        self.chain_file = chain_file
        self.chain = []

        if not self._load_chain():
            self._create_genesis_block()

    def _create_genesis_block(self):
        genesis = Block(
            index=0,
            claim_hash="GENESIS",
            fraud_score=0.0,
            decision="GENESIS",
            previous_hash="0",
            model_version=MODEL_VERSION,
            source="System"
        )

        genesis.mine(self.difficulty)
        self.chain.append(genesis)
        self._save_chain()

    def get_last_block(self):
        return self.chain[-1]

    def add_record(self, claim_data: dict, fraud_score: float, source="Prediction"):
        claim_string = json.dumps(claim_data, sort_keys=True)
        claim_hash = hashlib.sha256(claim_string.encode()).hexdigest()

        decision = "Fraudulent" if fraud_score >= 0.5 else "Legitimate"

        new_block = Block(
            index=len(self.chain),
            claim_hash=claim_hash,
            fraud_score=fraud_score,
            decision=decision,
            previous_hash=self.get_last_block().hash,
            model_version=MODEL_VERSION,
            source=source
        )

        new_block.mine(self.difficulty)

        self.chain.append(new_block)
        self._save_chain()

        return new_block

    def verify_integrity(self):
        target = "0" * self.difficulty

        for i, block in enumerate(self.chain):
            recalculated_hash = block.compute_hash()

            if recalculated_hash != block.hash:
                return False, f"Tampered block detected at block {i}"

            if not block.hash.startswith(target):
                return False, f"Block {i} failed Proof-of-Work check"

            if i > 0:
                previous_block = self.chain[i - 1]

                if block.previous_hash != previous_block.hash:
                    return False, f"Chain broken at block {i} — previous hash mismatch"

        return True, "Chain integrity verified — all blocks valid"

    def _save_chain(self):
        try:
            with open(self.chain_file, "w") as file:
                json.dump([block.to_dict() for block in self.chain], file, indent=2)
        except Exception as error:
            print(f"Warning: could not save chain — {error}")

    def _load_chain(self):
        if not os.path.exists(self.chain_file):
            return False

        try:
            with open(self.chain_file, "r") as file:
                data = json.load(file)

            self.chain = [Block.from_dict(item) for item in data]
            return True

        except Exception as error:
            print(f"Warning: could not load chain ({error}). Starting fresh.")
            return False

    def get_chain_as_dataframe(self):
        rows = []

        for block in self.chain:
            rows.append({
                "Block Index": block.index,
                "Timestamp": block.timestamp,
                "Decision": block.decision,
                "Fraud Score": block.fraud_score,
                "Model Version": block.model_version,
                "Source": block.source,
                "Nonce": block.nonce,
                "Claim Hash": block.claim_hash[:20] + "...",
                "Block Hash": block.hash[:20] + "...",
                "Previous Hash": block.previous_hash[:20] + "..."
            })

        return pd.DataFrame(rows)


# ============================================================
# PREPROCESSING HELPER FOR STANDALONE TESTING
# ============================================================

def preprocess_claims(input_df, scaler, preprocess_info):
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


# ============================================================
# STANDALONE SCRIPT
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("BLOCKCHAIN LEDGER TEST WITH XGBOOST MODEL")
    print("=" * 60)

    print("\nLoading XGBoost model, scaler, and preprocessing info...")

    baseline_models = joblib.load("baseline_models.pkl")
    model = baseline_models["XGBoost"]
    scaler = joblib.load("scaler.pkl")
    preprocess_info = joblib.load("preprocess_info.pkl")

    print("XGBoost model, scaler, and preprocessing info loaded successfully.")

    print("\nLoading dataset...")
    df = pd.read_csv("healthcare_fraud_detection.csv")

    claim_ids = df["Claim_ID"].tolist() if "Claim_ID" in df.columns else [f"CLM-{i}" for i in range(len(df))]
    provider_ids = df["Provider_ID"].tolist() if "Provider_ID" in df.columns else [f"PRV-{i}" for i in range(len(df))]

    X_scaled = preprocess_claims(df, scaler, preprocess_info)

    print(f"Dataset ready. Total claims: {len(X_scaled)}")

    print("\nRunning XGBoost fraud predictions...")
    fraud_probs = model.predict_proba(X_scaled)[:, 1]

    print(f"Sample scores: {fraud_probs[:5].round(4)}")

    print("\nLogging sample predictions to blockchain...")
    bc = Blockchain()

    sample_size = min(20, len(X_scaled))

    for i in range(sample_size):
        claim_data = {
            "claim_id": str(claim_ids[i]),
            "provider_id": str(provider_ids[i]),
            "fraud_score": round(float(fraud_probs[i]), 4),
            "model_version": MODEL_VERSION
        }

        bc.add_record(
            claim_data=claim_data,
            fraud_score=float(fraud_probs[i]),
            source="Standalone XGBoost Test"
        )

    print(f"Logged {sample_size} sample claims.")
    print(f"Total blocks in chain: {len(bc.chain)}")

    print("\n" + "=" * 60)
    print("BLOCKCHAIN INTEGRITY CHECK")
    print("=" * 60)

    is_valid, message = bc.verify_integrity()
    print(f"Valid  : {is_valid}")
    print(f"Result : {message}")

    print("\n" + "=" * 60)
    print("TAMPER DETECTION TEST")
    print("=" * 60)

    test_chain_file = "test_chain.json"

    if os.path.exists(test_chain_file):
        os.remove(test_chain_file)

    bc_test = Blockchain(chain_file=test_chain_file)

    for i in range(5):
        claim_data = {
            "claim_id": str(claim_ids[i]),
            "provider_id": str(provider_ids[i]),
            "fraud_score": round(float(fraud_probs[i]), 4),
            "model_version": MODEL_VERSION
        }

        bc_test.add_record(
            claim_data=claim_data,
            fraud_score=float(fraud_probs[i]),
            source="Tamper Test"
        )

    before_valid, before_msg = bc_test.verify_integrity()
    print(f"Before tamper — Valid: {before_valid} | {before_msg}")

    bc_test.chain[3].previous_hash = "tampered_hash"

    after_valid, after_msg = bc_test.verify_integrity()
    print(f"After tamper  — Valid: {after_valid} | {after_msg}")

    if not after_valid:
        print("Tamper detected correctly.")
    else:
        print("Warning: tamper was not detected.")

    if os.path.exists(test_chain_file):
        os.remove(test_chain_file)

    print("\n" + "=" * 60)
    print("BLOCKCHAIN LEDGER SAMPLE")
    print("=" * 60)

    chain_df = bc.get_chain_as_dataframe().head(10)
    print(chain_df.to_string(index=False))

    chain_df.to_csv("blockchain_ledger_sample.csv", index=False)
    print("\nSaved: blockchain_ledger_sample.csv")

    print("\nDone.")