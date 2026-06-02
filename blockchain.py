import hashlib
import json
import joblib
import numpy as np
import pandas as pd
import os
from datetime import datetime, timezone
import tensorflow as tf
from tensorflow.keras.models import load_model

CHAIN_FILE = "blockchain_data.json"   # where the chain is saved to disk
DIFFICULTY  = 3                        # PoW — hash must start with this many zeros


# ─────────────────────────────────────────
# BLOCK CLASS
# ─────────────────────────────────────────
class Block:
    def __init__(self, index, claim_hash, fraud_score,
                 decision, previous_hash,
                 model_version="ANN_v1.0",
                 nonce=0, timestamp=None):
        self.index          = index
        self.claim_hash     = claim_hash
        self.fraud_score    = round(float(fraud_score), 4)
        self.decision       = decision
        self.timestamp      = timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self.model_version  = model_version
        self.previous_hash  = previous_hash
        self.nonce          = nonce          # ← NEW: Proof-of-Work counter
        self.hash           = self.compute_hash()

    # ── Core hash (includes nonce so PoW changes the hash) ──────────
    def compute_hash(self):
        block_string = json.dumps({
            "index":         self.index,
            "claim_hash":    self.claim_hash,
            "fraud_score":   self.fraud_score,
            "decision":      self.decision,
            "timestamp":     self.timestamp,
            "model_version": self.model_version,
            "previous_hash": self.previous_hash,
            "nonce":         self.nonce        # ← included in hash
        }, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()

    # ── Proof of Work ────────────────────────────────────────────────
    def mine(self, difficulty=DIFFICULTY):
        """
        Keep incrementing nonce until the hash starts with
        `difficulty` leading zeros — this is Proof of Work.
        """
        target = "0" * difficulty
        while not self.hash.startswith(target):
            self.nonce += 1
            self.hash   = self.compute_hash()
        return self.hash

    # ── Serialise to dict (for JSON persistence) ────────────────────
    def to_dict(self):
        return {
            "index":         self.index,
            "claim_hash":    self.claim_hash,
            "fraud_score":   self.fraud_score,
            "decision":      self.decision,
            "timestamp":     self.timestamp,
            "model_version": self.model_version,
            "previous_hash": self.previous_hash,
            "nonce":         self.nonce,
            "hash":          self.hash
        }

    # ── Rebuild from dict (loading from disk) ───────────────────────
    @classmethod
    def from_dict(cls, d):
        block = cls(
            index         = d["index"],
            claim_hash    = d["claim_hash"],
            fraud_score   = d["fraud_score"],
            decision      = d["decision"],
            previous_hash = d["previous_hash"],
            model_version = d.get("model_version", "ANN_v1.0"),
            nonce         = d.get("nonce", 0),
            timestamp     = d.get("timestamp")
        )
        block.hash = d["hash"]   # restore the already-mined hash
        return block


# ─────────────────────────────────────────
# BLOCKCHAIN CLASS
# ─────────────────────────────────────────
class Blockchain:
    def __init__(self, difficulty=DIFFICULTY, chain_file=CHAIN_FILE):
        self.difficulty  = difficulty
        self.chain_file  = chain_file
        self.chain       = []

        # Try to load existing chain from disk first
        if not self._load_chain():
            self._create_genesis_block()

    # ── Genesis block ────────────────────────────────────────────────
    def _create_genesis_block(self):
        genesis = Block(
            index         = 0,
            claim_hash    = "GENESIS",
            fraud_score   = 0.0,
            decision      = "GENESIS",
            previous_hash = "0"
        )
        genesis.mine(self.difficulty)   # mine the genesis block too
        self.chain.append(genesis)
        self._save_chain()
        print(f"Genesis block created and mined. Hash: {genesis.hash[:20]}...")

    # ── Helpers ──────────────────────────────────────────────────────
    def get_last_block(self):
        return self.chain[-1]

    # ── Add a new record (mine it first) ─────────────────────────────
    def add_record(self, claim_data: dict, fraud_score: float):
        claim_string = json.dumps(claim_data, sort_keys=True)
        claim_hash   = hashlib.sha256(claim_string.encode()).hexdigest()
        decision     = "Fraudulent" if fraud_score > 0.5 else "Legitimate"

        new_block = Block(
            index         = len(self.chain),
            claim_hash    = claim_hash,
            fraud_score   = fraud_score,
            decision      = decision,
            previous_hash = self.get_last_block().hash
        )

        # ── Proof of Work ──────────────────────────────────────
        new_block.mine(self.difficulty)
        # ──────────────────────────────────────────────────────

        self.chain.append(new_block)
        self._save_chain()        # persist every new block to disk
        return new_block

    # ── Verify integrity ─────────────────────────────────────────────
    def verify_integrity(self):
        target = "0" * self.difficulty

        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]

            # 1. Previous hash linkage
            if current.previous_hash != previous.hash:
                return False, f"Chain broken at block {i} — previous hash mismatch"

            # 2. Recompute hash and compare
            recalculated = current.compute_hash()
            if recalculated != current.hash:
                return False, f"Tampered block detected at block {i}"

            # 3. Proof-of-Work check — hash must still meet difficulty
            if not current.hash.startswith(target):
                return False, f"Block {i} failed Proof-of-Work check"

        return True, "Chain integrity verified — all blocks valid"

    # ── Persistence: save chain to JSON ──────────────────────────────
    def _save_chain(self):
        try:
            with open(self.chain_file, "w") as f:
                json.dump([b.to_dict() for b in self.chain], f, indent=2)
        except Exception as e:
            print(f"Warning: could not save chain — {e}")

    # ── Persistence: load chain from JSON ────────────────────────────
    def _load_chain(self):
        if not os.path.exists(self.chain_file):
            return False
        try:
            with open(self.chain_file, "r") as f:
                data = json.load(f)
            self.chain = [Block.from_dict(d) for d in data]
            print(f"Chain loaded from disk — {len(self.chain)} blocks.")
            return True
        except Exception as e:
            print(f"Warning: could not load chain ({e}). Starting fresh.")
            return False

    # ── DataFrame view (used by app.py) ──────────────────────────────
    def get_chain_as_dataframe(self):
        rows = []
        for block in self.chain:
            rows.append({
                "Block Index":   block.index,
                "Timestamp":     block.timestamp,
                "Decision":      block.decision,
                "Fraud Score":   block.fraud_score,
                "Nonce":         block.nonce,
                "Claim Hash":    block.claim_hash[:20] + "...",
                "Block Hash":    block.hash[:20] + "...",
                "Previous Hash": block.previous_hash[:20] + "..."
            })
        return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════
#  STANDALONE SCRIPT — run blockchain.py directly to process
#  all claims from the dataset (same as before, now with PoW)
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":

    # ── Load ANN model and scaler ────────────────────────────────────
    print("\nLoading ANN model and scaler...")
    model  = load_model("ann_model.keras")
    scaler = joblib.load("scaler.pkl")
    print("Model and scaler loaded successfully.")

    # ── Load and preprocess dataset ──────────────────────────────────
    df = pd.read_csv("healthcare_fraud_detection.csv")
    df = df.drop(['Claim_Submission_Date'], axis=1)

    claim_ids    = df['Claim_ID'].tolist()
    provider_ids = df['Provider_ID'].tolist()

    df = df.drop(['Provider_ID', 'Claim_ID'], axis=1)

    for col in df.select_dtypes(include='number').columns:
        df[col] = df[col].fillna(df[col].mean())
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    df['claim_to_cost_ratio']   = df['Claim_Amount'] / (df['Approved_Amount'] + 1)
    Q1  = df['Claim_Amount'].quantile(0.25)
    Q3  = df['Claim_Amount'].quantile(0.75)
    IQR = Q3 - Q1
    df['cost_outlier_flag']     = (df['Claim_Amount'] > Q3 + 1.5 * IQR).astype(int)
    df['high_claim_frequency']  = (
        df['Number_of_Claims_Per_Provider_Monthly'] >
        df['Number_of_Claims_Per_Provider_Monthly'].quantile(0.90)
    ).astype(int)

    df_encoded = pd.get_dummies(df)
    X          = df_encoded.drop('Is_Fraud', axis=1)
    X_scaled   = scaler.transform(X)

    print(f"\nDataset ready. Total claims: {len(X_scaled)}")

    # ── Run ANN predictions ──────────────────────────────────────────
    print("\nRunning ANN fraud predictions...")
    fraud_probs = model.predict(X_scaled, verbose=0).flatten()
    print(f"Sample scores: {fraud_probs[:5].round(4)}")

    # ── Log to blockchain (with Proof of Work) ───────────────────────
    print("\nLogging predictions to blockchain (Proof of Work active)...")
    bc = Blockchain()

    for i in range(len(X_scaled)):
        claim_data = {
            "claim_id":    claim_ids[i],
            "provider_id": provider_ids[i],
            "fraud_score": round(float(fraud_probs[i]), 4)
        }
        bc.add_record(claim_data, float(fraud_probs[i]))
        if (i + 1) % 1000 == 0:
            print(f"  Logged {i + 1} / {len(X_scaled)} claims...")

    print(f"\nAll {len(X_scaled)} claims logged.")
    print(f"Total blocks in chain: {len(bc.chain)} (includes genesis)")

    # ── Verify integrity ─────────────────────────────────────────────
    print("\n" + "="*55)
    print("BLOCKCHAIN INTEGRITY CHECK")
    print("="*55)
    is_valid, message = bc.verify_integrity()
    print(f"Result : {message}")
    print(f"Valid  : {is_valid}")

    # ── Tamper detection test ─────────────────────────────────────────
    print("\n" + "="*55)
    print("TAMPER DETECTION TEST")
    print("="*55)
    bc_test = Blockchain(chain_file="test_chain.json")
    for i in range(10):
        claim_data = {
            "claim_id":    claim_ids[i],
            "provider_id": provider_ids[i],
            "fraud_score": round(float(fraud_probs[i]), 4)
        }
        bc_test.add_record(claim_data, float(fraud_probs[i]))

    is_valid_before, msg_before = bc_test.verify_integrity()
    print(f"Before tamper — Valid: {is_valid_before} | {msg_before}")

    bc_test.chain[3].previous_hash = "0000000000000000tampered0000000000000000"
    is_valid_after, msg_after = bc_test.verify_integrity()
    print(f"After tamper  — Valid: {is_valid_after} | {msg_after}")

    if not is_valid_after:
        print("Tamper detected correctly! Blockchain is secure.")
    else:
        print("WARNING: Tamper not detected")

    # Clean up test file
    if os.path.exists("test_chain.json"):
        os.remove("test_chain.json")

    # ── Print blockchain summary table ────────────────────────────────
    print("\n" + "="*55)
    print("BLOCKCHAIN LEDGER — FIRST 10 BLOCKS")
    print("="*55)
    chain_df = bc.get_chain_as_dataframe().head(11)
    print(chain_df.to_string(index=False))
    chain_df.to_csv("blockchain_ledger_sample.csv", index=False)
    print("\nSaved: blockchain_ledger_sample.csv")

    # ── Fraud summary ─────────────────────────────────────────────────
    print("\n" + "="*55)
    print("FRAUD DETECTION SUMMARY")
    print("="*55)
    decisions   = ["Fraudulent" if p > 0.5 else "Legitimate" for p in fraud_probs]
    fraud_count = decisions.count("Fraudulent")
    legit_count = decisions.count("Legitimate")
    total       = len(decisions)

    print(f"Total claims processed   : {total}")
    print(f"Flagged as Fraudulent    : {fraud_count} ({fraud_count/total*100:.1f}%)")
    print(f"Flagged as Legitimate    : {legit_count} ({legit_count/total*100:.1f}%)")
    print(f"High risk (score > 0.8)  : {sum(1 for p in fraud_probs if p > 0.8)}")
    print(f"Medium risk (0.5 – 0.8)  : {sum(1 for p in fraud_probs if 0.5 <= p <= 0.8)}")
    print(f"Low risk (score < 0.5)   : {sum(1 for p in fraud_probs if p < 0.5)}")

    print("\nDone! Run app.py next (Streamlit dashboard).")