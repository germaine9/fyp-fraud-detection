import joblib
import numpy as np
import pandas as pd
import hashlib
import json
from datetime import datetime, timezone

print("="*55)
print("SYSTEM VERIFICATION TESTS")
print("="*55)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS — {name}")
        passed += 1
    else:
        print(f"  FAIL — {name}")
        failed += 1

# ── Test 1: Dataset loads correctly
try:
    df = pd.read_csv("healthcare_fraud_detection.csv")
    test("Dataset loads successfully",        True)
    test("Dataset has 10000 rows",            df.shape[0] == 10000)
    test("Is_Fraud column exists",            'Is_Fraud' in df.columns)
    test("Fraud rate between 5-15%",          0.05 < df['Is_Fraud'].mean() < 0.15)
except Exception as e:
    test(f"Dataset load — {e}",               False)

# ── Test 2: Scaler loads correctly
try:
    scaler = joblib.load("scaler.pkl")
    test("Scaler loads successfully",         True)
    test("Scaler is StandardScaler",
         "StandardScaler" in str(type(scaler)))
except Exception as e:
    test(f"Scaler load — {e}",               False)

# ── Test 3: ANN model loads correctly
try:
    from tensorflow.keras.models import load_model
    model = load_model("ann_model.keras")
    test("ANN model loads successfully",      True)
    test("ANN output shape is (None, 1)",
         model.output_shape == (None, 1))
except Exception as e:
    test(f"ANN model load — {e}",            False)

# ── Test 4: Baseline models load correctly
try:
    baseline = joblib.load("baseline_models.pkl")
    test("Baseline models load successfully", True)
    test("Contains 4 models",                len(baseline) == 4)
    for name in ["Logistic Regression",
                 "Decision Tree",
                 "Random Forest", "XGBoost"]:
        test(f"Model exists: {name}",         name in baseline)
except Exception as e:
    test(f"Baseline models load — {e}",      False)

# ── Test 5: Blockchain works correctly
try:
    import sys
    sys.path.insert(0, '.')
    from blockchain import Block, Blockchain

    bc = Blockchain()
    test("Blockchain initialises",            len(bc.chain) == 1)
    test("Genesis block created",
         bc.chain[0].decision == "GENESIS")

    block = bc.add_record(
        {"claim_id": "TEST-001", "amount": 1000},
        0.82
    )
    test("Block added successfully",          len(bc.chain) == 2)
    test("Fraud decision correct",
         block.decision == "Fraudulent")
    test("Hash is 64 characters",            len(block.hash) == 64)

    is_valid, msg = bc.verify_integrity()
    test("Chain integrity valid",             is_valid)

    bc.chain[1].previous_hash = "tampered"
    is_valid2, _ = bc.verify_integrity()
    test("Tamper detection works",            not is_valid2)

except Exception as e:
    test(f"Blockchain test — {e}",           False)

# ── Test 6: SHA-256 hashing works
try:
    data        = json.dumps({"test": "data"}, sort_keys=True)
    hash1       = hashlib.sha256(data.encode()).hexdigest()
    hash2       = hashlib.sha256(data.encode()).hexdigest()
    test("SHA-256 produces consistent hash",  hash1 == hash2)
    test("SHA-256 hash is 64 chars",          len(hash1) == 64)

    data2       = json.dumps({"test": "different"}, sort_keys=True)
    hash3       = hashlib.sha256(data2.encode()).hexdigest()
    test("Different data produces diff hash", hash1 != hash3)
except Exception as e:
    test(f"Hashing test — {e}",              False)

# ── Final results
print("\n" + "="*55)
print(f"RESULTS: {passed} passed, {failed} failed")
print("="*55)

if failed == 0:
    print("All tests passed. System is ready.")
else:
    print(f"Fix the {failed} failing test(s) before submission.")