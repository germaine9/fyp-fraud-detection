# test_system_fixed.py
# System verification tests for the FYP prototype.
#
# Run with:
#     python test_system_fixed.py
#
# Purpose:
# - Check that key files can be loaded
# - Check dataset structure
# - Check trained models
# - Check blockchain integrity and tamper detection
# - Avoid touching the real blockchain_data.json file

import os
import sys
import json
import hashlib
import joblib
import pandas as pd


print("=" * 60)
print("SYSTEM VERIFICATION TESTS")
print("=" * 60)

passed = 0
failed = 0


def test(name, condition):
    """Simple test result printer."""
    global passed, failed

    if condition:
        print(f"PASS — {name}")
        passed += 1
    else:
        print(f"FAIL — {name}")
        failed += 1


# ============================================================
# TEST 1: Dataset
# ============================================================

try:
    df = pd.read_csv("healthcare_fraud_detection.csv")

    test("Dataset loads successfully", True)
    test("Dataset is not empty", df.shape[0] > 0)
    test("Dataset has Is_Fraud column", "Is_Fraud" in df.columns)

    if "Is_Fraud" in df.columns:
        fraud_rate = df["Is_Fraud"].mean()
        test("Fraud rate is within reasonable range", 0 < fraud_rate < 1)

    required_columns = [
        "Claim_Amount",
        "Approved_Amount",
        "Number_of_Claims_Per_Provider_Monthly"
    ]

    for col in required_columns:
        test(f"Required column exists: {col}", col in df.columns)

except Exception as error:
    test(f"Dataset test error: {error}", False)


# ============================================================
# TEST 2: Preprocessing files
# ============================================================

try:
    preprocess_info = joblib.load("preprocess_info.pkl")

    test("preprocess_info.pkl loads successfully", True)
    test("preprocess_info contains feature_columns", "feature_columns" in preprocess_info)
    test("preprocess_info contains numeric_cols", "numeric_cols" in preprocess_info)
    test("preprocess_info contains categorical_cols", "categorical_cols" in preprocess_info)
    test("preprocess_info contains high_claim_threshold", "high_claim_threshold" in preprocess_info)

    if "feature_columns" in preprocess_info:
        test("Feature column list is not empty", len(preprocess_info["feature_columns"]) > 0)

except Exception as error:
    test(f"preprocess_info load error: {error}", False)


# ============================================================
# TEST 3: Scaler
# ============================================================

try:
    scaler = joblib.load("scaler.pkl")

    test("scaler.pkl loads successfully", True)
    test("Scaler is StandardScaler", "StandardScaler" in str(type(scaler)))

except Exception as error:
    test(f"Scaler load error: {error}", False)


# ============================================================
# TEST 4: ANN model
# ============================================================

try:
    from tensorflow.keras.models import load_model

    model = load_model("ann_model.keras")

    test("ANN model loads successfully", True)
    test("ANN output shape is one output neuron", model.output_shape[-1] == 1)

except Exception as error:
    test(f"ANN model load error: {error}", False)


# ============================================================
# TEST 5: Baseline models
# ============================================================

try:
    baseline_models = joblib.load("baseline_models.pkl")

    expected_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost"
    ]

    test("baseline_models.pkl loads successfully", True)
    test("Baseline model file contains dictionary", isinstance(baseline_models, dict))
    test("Contains 4 baseline models", len(baseline_models) == 4)

    for name in expected_models:
        test(f"Baseline model exists: {name}", name in baseline_models)

except Exception as error:
    test(f"Baseline models load error: {error}", False)


# ============================================================
# TEST 6: Model summary files
# ============================================================

summary_files = [
    "model_summary.csv",
    "ann_model_summary.csv",
    "full_model_summary.csv",
    "ann_classification_report.csv"
]

for file_name in summary_files:
    try:
        summary_df = pd.read_csv(file_name)
        test(f"{file_name} loads successfully", True)
        test(f"{file_name} is not empty", not summary_df.empty)
    except Exception as error:
        test(f"{file_name} load error: {error}", False)


# ============================================================
# TEST 7: Output image files
# ============================================================

image_files = [
    "confusion_matrix_ANN.png",
    "ann_training_history.png",
    "roc_curve_ANN.png"
]

for file_name in image_files:
    test(f"Image file exists: {file_name}", os.path.exists(file_name))


# ============================================================
# TEST 8: Blockchain
# ============================================================

try:
    sys.path.insert(0, ".")
    from blockchain import Blockchain

    test_chain_file = "test_system_chain.json"

    # Make sure this test does not use or modify the real blockchain_data.json
    if os.path.exists(test_chain_file):
        os.remove(test_chain_file)

    bc = Blockchain(chain_file=test_chain_file)

    test("Blockchain initialises with genesis block", len(bc.chain) == 1)
    test("Genesis block decision is GENESIS", bc.chain[0].decision == "GENESIS")

    block = bc.add_record(
        {"claim_id": "TEST-001", "amount": 1000},
        0.82
    )

    test("New block added successfully", len(bc.chain) == 2)
    test("Fraud decision is correct", block.decision == "Fraudulent")
    test("Block hash is 64 characters", len(block.hash) == 64)
    test("Block has nonce attribute", hasattr(block, "nonce"))

    if hasattr(bc, "difficulty"):
        test("Proof-of-Work hash meets difficulty", block.hash.startswith("0" * bc.difficulty))

    is_valid, message = bc.verify_integrity()
    test("Blockchain integrity check passes", is_valid)

    # Tamper test
    original_previous_hash = bc.chain[1].previous_hash
    bc.chain[1].previous_hash = "tampered_hash"

    is_valid_after_tamper, _ = bc.verify_integrity()
    test("Tamper detection works", not is_valid_after_tamper)

    # Restore and clean up
    bc.chain[1].previous_hash = original_previous_hash

    if os.path.exists(test_chain_file):
        os.remove(test_chain_file)

except Exception as error:
    test(f"Blockchain test error: {error}", False)


# ============================================================
# TEST 9: SHA-256 hashing
# ============================================================

try:
    data_1 = json.dumps({"test": "data"}, sort_keys=True)
    data_2 = json.dumps({"test": "different"}, sort_keys=True)

    hash_1 = hashlib.sha256(data_1.encode()).hexdigest()
    hash_2 = hashlib.sha256(data_1.encode()).hexdigest()
    hash_3 = hashlib.sha256(data_2.encode()).hexdigest()

    test("SHA-256 produces consistent hash", hash_1 == hash_2)
    test("SHA-256 hash length is 64 characters", len(hash_1) == 64)
    test("Different data produces different hash", hash_1 != hash_3)

except Exception as error:
    test(f"Hashing test error: {error}", False)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if failed == 0:
    print("All tests passed. System is ready for demo.")
else:
    print(f"{failed} test(s) failed. Please fix them before submission.")
