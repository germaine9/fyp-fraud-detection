# test_system_a_plus.py
# Enhanced system verification tests for the FYP prototype.
#
# Run with:
#     python test_system_a_plus.py
#
# Purpose:
# - Check dataset, preprocessing, trained models, ANN comparison model, output files
# - Check XGBoost end-to-end prediction path
# - Check blockchain integrity, Proof-of-Work, persistence, and tamper detection
# - Check OCR module import
# - Avoid touching the real blockchain_data.json file

import os
import sys
import json
import hashlib
import tempfile
import importlib.util

import joblib
import pandas as pd


print("=" * 70)
print("A+ SYSTEM VERIFICATION TESTS")
print("=" * 70)

passed = 0
failed = 0


def test(name, condition):
    global passed, failed

    if condition:
        print(f"PASS — {name}")
        passed += 1
    else:
        print(f"FAIL — {name}")
        failed += 1


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# TEST 1: Dataset
# ============================================================

section("TEST 1: Dataset")

try:
    df = pd.read_csv("healthcare_fraud_detection.csv")

    test("Dataset loads successfully", True)
    test("Dataset is not empty", df.shape[0] > 0)
    test("Dataset has at least 1,000 rows", df.shape[0] >= 1000)
    test("Dataset has Is_Fraud column", "Is_Fraud" in df.columns)

    if "Is_Fraud" in df.columns:
        fraud_rate = df["Is_Fraud"].mean()
        test("Fraud rate is within valid range", 0 < fraud_rate < 1)
        test("Fraud rate is suitable for imbalanced fraud detection", 0.01 <= fraud_rate <= 0.30)

    required_columns = [
        "Claim_Amount",
        "Approved_Amount",
        "Number_of_Claims_Per_Provider_Monthly",
        "Patient_Age",
        "Patient_Gender",
        "Diagnosis_Code",
        "Procedure_Code",
        "Insurance_Type",
        "Claim_Status",
        "Length_of_Stay",
        "Visit_Type",
        "Chronic_Condition_Flag",
        "Prior_Visits_12m",
    ]

    for col in required_columns:
        test(f"Required column exists: {col}", col in df.columns)

except Exception as error:
    test(f"Dataset test error: {error}", False)
    df = None


# ============================================================
# TEST 2: Preprocessing files
# ============================================================

section("TEST 2: Preprocessing artefacts")

try:
    preprocess_info = joblib.load("preprocess_info.pkl")

    required_keys = [
        "feature_columns",
        "numeric_cols",
        "categorical_cols",
        "numeric_means",
        "categorical_modes",
        "claim_q3",
        "claim_iqr",
        "high_claim_threshold",
    ]

    test("preprocess_info.pkl loads successfully", True)

    for key in required_keys:
        test(f"preprocess_info contains {key}", key in preprocess_info)

    if "feature_columns" in preprocess_info:
        feature_columns = preprocess_info["feature_columns"]
        test("Feature column list is not empty", len(feature_columns) > 0)
        test("Processed feature count is reasonable", len(feature_columns) >= 20)

except Exception as error:
    test(f"preprocess_info load error: {error}", False)
    preprocess_info = None


# ============================================================
# TEST 3: Scaler
# ============================================================

section("TEST 3: Scaler")

try:
    scaler = joblib.load("scaler.pkl")

    test("scaler.pkl loads successfully", True)
    test("Scaler is StandardScaler", "StandardScaler" in str(type(scaler)))
    test("Scaler has fitted mean_", hasattr(scaler, "mean_"))
    test("Scaler has fitted scale_", hasattr(scaler, "scale_"))

except Exception as error:
    test(f"Scaler load error: {error}", False)
    scaler = None


# ============================================================
# TEST 4: Baseline models and final XGBoost model
# ============================================================

section("TEST 4: Baseline models and final XGBoost model")

try:
    baseline_models = joblib.load("baseline_models.pkl")

    expected_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "XGBoost",
    ]

    test("baseline_models.pkl loads successfully", True)
    test("Baseline model file contains dictionary", isinstance(baseline_models, dict))
    test("Contains 4 baseline models", len(baseline_models) == 4)

    for name in expected_models:
        test(f"Baseline model exists: {name}", name in baseline_models)

    if "XGBoost" in baseline_models:
        xgb_model = baseline_models["XGBoost"]
        test("Final selected model is XGBoost", xgb_model.__class__.__name__ == "XGBClassifier")
        test("XGBoost supports predict_proba", hasattr(xgb_model, "predict_proba"))

except Exception as error:
    test(f"Baseline/XGBoost model load error: {error}", False)
    baseline_models = None


# ============================================================
# TEST 5: End-to-end XGBoost prediction path
# ============================================================

section("TEST 5: End-to-end XGBoost prediction path")

try:
    if df is None or preprocess_info is None or scaler is None or baseline_models is None:
        raise RuntimeError("Required dataset/model artefacts are missing from earlier tests.")

    xgb_model = baseline_models["XGBoost"]

    sample_df = df.head(5).copy()

    # Remove training-only/non-feature columns, same as the app.
    app_df = sample_df.drop(
        columns=["Provider_ID", "Claim_ID", "Claim_Submission_Date", "Is_Fraud"],
        errors="ignore",
    )

    numeric_cols = preprocess_info["numeric_cols"]
    categorical_cols = preprocess_info["categorical_cols"]

    for col in numeric_cols:
        if col not in app_df.columns:
            app_df[col] = preprocess_info["numeric_means"][col]
        else:
            app_df[col] = pd.to_numeric(app_df[col], errors="coerce").fillna(
                preprocess_info["numeric_means"][col]
            )

    for col in categorical_cols:
        if col not in app_df.columns:
            app_df[col] = preprocess_info["categorical_modes"][col]
        else:
            app_df[col] = app_df[col].fillna(preprocess_info["categorical_modes"][col]).astype(str)

    app_df["claim_to_cost_ratio"] = app_df["Claim_Amount"] / (app_df["Approved_Amount"] + 1)
    app_df["cost_outlier_flag"] = (
        app_df["Claim_Amount"] > preprocess_info["claim_q3"] + 1.5 * preprocess_info["claim_iqr"]
    ).astype(int)
    app_df["high_claim_frequency"] = (
        app_df["Number_of_Claims_Per_Provider_Monthly"] > preprocess_info["high_claim_threshold"]
    ).astype(int)

    app_df = pd.get_dummies(app_df)
    app_df = app_df.reindex(columns=preprocess_info["feature_columns"], fill_value=0)

    X_scaled = scaler.transform(app_df)
    scores = xgb_model.predict_proba(X_scaled)[:, 1]
    decisions = ["Fraudulent" if float(score) >= 0.5 else "Legitimate" for score in scores]

    test("Sample preprocessing matches trained feature count", X_scaled.shape[1] == len(preprocess_info["feature_columns"]))
    test("XGBoost returns one score per sample row", len(scores) == len(sample_df))
    test("Fraud scores are probabilities between 0 and 1", all(0 <= float(score) <= 1 for score in scores))
    test("Prediction decisions are valid labels", all(label in ["Fraudulent", "Legitimate"] for label in decisions))

except Exception as error:
    test(f"End-to-end prediction test error: {error}", False)


# ============================================================
# TEST 6: ANN comparison model
# ============================================================

section("TEST 6: ANN comparison model")

try:
    from tensorflow.keras.models import load_model

    ann_model = load_model("ann_model.keras")

    test("ANN comparison model loads successfully", True)
    test("ANN output shape is one output neuron", ann_model.output_shape[-1] == 1)

except Exception as error:
    test(f"ANN model load error: {error}", False)


# ============================================================
# TEST 7: Model summary files
# ============================================================

section("TEST 7: Model summary files")

summary_files = [
    "model_summary.csv",
    "ann_model_summary.csv",
    "full_model_summary.csv",
    "ann_classification_report.csv",
]

for file_name in summary_files:
    try:
        summary_df = pd.read_csv(file_name)
        test(f"{file_name} loads successfully", True)
        test(f"{file_name} is not empty", not summary_df.empty)

        if file_name == "full_model_summary.csv" and "Model" in summary_df.columns:
            test("full_model_summary.csv includes XGBoost", "XGBoost" in summary_df["Model"].values)
            test("full_model_summary.csv includes ANN", "ANN" in summary_df["Model"].values)

    except Exception as error:
        test(f"{file_name} load error: {error}", False)


# ============================================================
# TEST 8: Output image files
# ============================================================

section("TEST 8: Output image files")

image_files = [
    "confusion_matrix_XGBoost.png",
    "model_comparison.png",
    "roc_curve_comparison.png",
    "confusion_matrix_ANN.png",
    "ann_training_history.png",
    "roc_curve_ANN.png",
    "cross_validation_recall.png",
    "smote_comparison.png",
]

for file_name in image_files:
    test(f"Image file exists: {file_name}", os.path.exists(file_name))
    if os.path.exists(file_name):
        test(f"Image file is not empty: {file_name}", os.path.getsize(file_name) > 0)


# ============================================================
# TEST 9: Blockchain
# ============================================================

section("TEST 9: Blockchain")

try:
    sys.path.insert(0, ".")
    from blockchain import Blockchain, MODEL_VERSION

    with tempfile.TemporaryDirectory() as tmpdir:
        test_chain_file = os.path.join(tmpdir, "test_system_chain.json")

        bc = Blockchain(chain_file=test_chain_file)

        test("Blockchain initialises with genesis block", len(bc.chain) == 1)
        test("Genesis block decision is GENESIS", bc.chain[0].decision == "GENESIS")
        test("Genesis block has model version", hasattr(bc.chain[0], "model_version"))

        block = bc.add_record(
            {"claim_id": "TEST-001", "amount": 1000, "model_version": MODEL_VERSION},
            0.82,
            source="System Test",
        )

        test("New block added successfully", len(bc.chain) == 2)
        test("Fraud decision is correct", block.decision == "Fraudulent")
        test("Block hash is 64 characters", len(block.hash) == 64)
        test("Block has nonce attribute", hasattr(block, "nonce"))
        test("Block has model_version attribute", hasattr(block, "model_version"))
        test("Block model version is XGBoost", "XGBoost" in block.model_version)

        if hasattr(bc, "difficulty"):
            test("Proof-of-Work hash meets difficulty", block.hash.startswith("0" * bc.difficulty))

        is_valid, message = bc.verify_integrity()
        test("Blockchain integrity check passes", is_valid)

        # Persistence check: reload from the temporary chain file.
        bc_reload = Blockchain(chain_file=test_chain_file)
        test("Blockchain persists and reloads blocks", len(bc_reload.chain) == 2)
        reloaded_valid, _ = bc_reload.verify_integrity()
        test("Reloaded blockchain remains valid", reloaded_valid)

        # Tamper detection check.
        original_previous_hash = bc.chain[1].previous_hash
        bc.chain[1].previous_hash = "tampered_hash"

        is_valid_after_tamper, _ = bc.verify_integrity()
        test("Tamper detection works", not is_valid_after_tamper)

        append_blocked = False
        try:
            bc.add_record({"claim_id": "SHOULD-NOT-APPEND"}, 0.50, source="System Test")
        except RuntimeError:
            append_blocked = True
        test("Invalid ledger refuses new records", append_blocked)

        bc.chain[1].previous_hash = original_previous_hash

except Exception as error:
    test(f"Blockchain test error: {error}", False)


# ============================================================
# TEST 10: SHA-256 hashing
# ============================================================

section("TEST 10: SHA-256 hashing")

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
# TEST 11: OCR module import
# ============================================================

section("TEST 11: OCR module")

try:
    spec = importlib.util.find_spec("document_scanner")
    test("document_scanner.py module is available", spec is not None)

    if spec is not None:
        import document_scanner
        test("OCR render function exists", hasattr(document_scanner, "render_document_scanner"))
        test("OCR parse function exists", hasattr(document_scanner, "parse_claim_fields_from_text"))

except Exception as error:
    test(f"OCR module test error: {error}", False)


# ============================================================
# TEST 12: App file existence
# ============================================================

section("TEST 12: Streamlit app file")

candidate_apps = [
    "app_polished.py",
    "app.py",
    "app_ux_final.py",
    "app_ux_fixed.py",
    "app_corrected.py",
    "app_clean.py",
]

existing_apps = [file_name for file_name in candidate_apps if os.path.exists(file_name)]

test("At least one Streamlit app file exists", len(existing_apps) > 0)

if existing_apps:
    print(f"Found app file(s): {', '.join(existing_apps)}")


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n" + "=" * 70)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 70)

if failed == 0:
    print("All tests passed. System is ready for demo.")
else:
    print(f"{failed} test(s) failed. Please fix them before submission.")
