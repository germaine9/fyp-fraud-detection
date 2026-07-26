# train_ann_fixed.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from imblearn.over_sampling import SMOTE

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


DATASET_FILE = "healthcare_fraud_detection.csv"
DATASET_SOURCE = "https://www.kaggle.com/datasets/esseasd/healthcare-fraud-detection-dataset"
REQUIRED_COLUMNS = {
    "Is_Fraud", "Claim_Amount", "Approved_Amount",
    "Number_of_Claims_Per_Provider_Monthly"
}


def validate_dataset(dataframe: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    labels = set(pd.Series(dataframe["Is_Fraud"]).dropna().unique().tolist())
    if not labels.issubset({0, 1}) or not labels:
        raise ValueError("Is_Fraud must contain binary labels 0 and 1.")


# ============================================================
# STEP 0: Reproducibility
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)


# ============================================================
# STEP 1: Load dataset
# ============================================================

df = pd.read_csv(DATASET_FILE)
validate_dataset(df)
print(f"Dataset source: {DATASET_SOURCE}")

print("Dataset loaded:", df.shape)
print("\nClass distribution:")
print(df["Is_Fraud"].value_counts())


# ============================================================
# STEP 2: Drop unnecessary columns
# ============================================================

drop_cols = ["Provider_ID", "Claim_ID", "Claim_Submission_Date"]
df = df.drop(columns=drop_cols, errors="ignore")

print("\nColumns after dropping unnecessary columns:")
print(df.columns.tolist())


# ============================================================
# STEP 3: Split features and target BEFORE preprocessing
# ============================================================

X = df.drop("Is_Fraud", axis=1)
y = df["Is_Fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)


# ============================================================
# STEP 4: Preprocessing functions
# ============================================================

def fit_preprocess_train(X_train):
    """
    Fit preprocessing only on the training set to avoid data leakage.
    """

    X_train = X_train.copy()

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    numeric_means = X_train[numeric_cols].mean()

    if len(categorical_cols) > 0:
        categorical_modes = X_train[categorical_cols].mode().iloc[0]
    else:
        categorical_modes = pd.Series(dtype="object")

    for col in numeric_cols:
        X_train[col] = X_train[col].fillna(numeric_means[col])

    for col in categorical_cols:
        X_train[col] = X_train[col].fillna(categorical_modes[col])

    claim_q1 = X_train["Claim_Amount"].quantile(0.25)
    claim_q3 = X_train["Claim_Amount"].quantile(0.75)
    claim_iqr = claim_q3 - claim_q1

    high_claim_threshold = X_train[
        "Number_of_Claims_Per_Provider_Monthly"
    ].quantile(0.90)

    X_train["claim_to_cost_ratio"] = (
        X_train["Claim_Amount"] / (X_train["Approved_Amount"] + 1)
    )

    X_train["cost_outlier_flag"] = (
        X_train["Claim_Amount"] > claim_q3 + 1.5 * claim_iqr
    ).astype(int)

    X_train["high_claim_frequency"] = (
        X_train["Number_of_Claims_Per_Provider_Monthly"] > high_claim_threshold
    ).astype(int)

    X_train = pd.get_dummies(X_train)

    preprocess_info = {
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "numeric_means": numeric_means,
        "categorical_modes": categorical_modes,
        "claim_q1": claim_q1,
        "claim_q3": claim_q3,
        "claim_iqr": claim_iqr,
        "high_claim_threshold": high_claim_threshold,
        "feature_columns": X_train.columns.tolist()
    }

    return X_train, preprocess_info


def transform_preprocess(X_data, preprocess_info):
    """
    Apply training-set preprocessing rules to test/new data.
    """

    X_data = X_data.copy()

    numeric_cols = preprocess_info["numeric_cols"]
    categorical_cols = preprocess_info["categorical_cols"]
    numeric_means = preprocess_info["numeric_means"]
    categorical_modes = preprocess_info["categorical_modes"]

    for col in numeric_cols:
        if col in X_data.columns:
            X_data[col] = X_data[col].fillna(numeric_means[col])

    for col in categorical_cols:
        if col in X_data.columns:
            X_data[col] = X_data[col].fillna(categorical_modes[col])

    claim_q3 = preprocess_info["claim_q3"]
    claim_iqr = preprocess_info["claim_iqr"]
    high_claim_threshold = preprocess_info["high_claim_threshold"]

    X_data["claim_to_cost_ratio"] = (
        X_data["Claim_Amount"] / (X_data["Approved_Amount"] + 1)
    )

    X_data["cost_outlier_flag"] = (
        X_data["Claim_Amount"] > claim_q3 + 1.5 * claim_iqr
    ).astype(int)

    X_data["high_claim_frequency"] = (
        X_data["Number_of_Claims_Per_Provider_Monthly"] > high_claim_threshold
    ).astype(int)

    X_data = pd.get_dummies(X_data)

    X_data = X_data.reindex(
        columns=preprocess_info["feature_columns"],
        fill_value=0
    )

    return X_data


# ============================================================
# STEP 5: Apply preprocessing
# ============================================================

X_train_processed, preprocess_info = fit_preprocess_train(X_train)
X_test_processed = transform_preprocess(X_test, preprocess_info)

joblib.dump(preprocess_info, "ann_preprocess_info.pkl")

print("\nPreprocessing completed.")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
print("Saved: ann_preprocess_info.pkl")


# ============================================================
# STEP 6: Apply SMOTE only to training data
# ============================================================

fraud_before = y_train.value_counts()[1]
legit_before = y_train.value_counts()[0]

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

fraud_after = pd.Series(y_train_smote).value_counts()[1]
legit_after = pd.Series(y_train_smote).value_counts()[0]

print("\nBefore SMOTE:")
print(f"Legitimate: {legit_before} | Fraud: {fraud_before}")

print("\nAfter SMOTE:")
print(f"Legitimate: {legit_after} | Fraud: {fraud_after}")


# ============================================================
# STEP 7: Scale features
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test_processed)

joblib.dump(scaler, "ann_scaler.pkl")

print("\nANN scaler saved to ann_scaler.pkl")


# ============================================================
# STEP 8: Build ANN model
# ============================================================

def build_ann(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),

        Dense(64, activation="relu"),
        BatchNormalization(),
        Dropout(0.30),

        Dense(32, activation="relu"),
        BatchNormalization(),
        Dropout(0.20),

        Dense(16, activation="relu"),
        Dropout(0.10),

        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


input_dim = X_train_scaled.shape[1]
model = build_ann(input_dim)

print("\nANN Model Summary:")
model.summary()


# ============================================================
# STEP 9: Train ANN model
# ============================================================

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=6,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining ANN...")
print("Note: validation_split is taken from the SMOTE-balanced training matrix. ")
print("The held-out test set remains untouched and is used only for final evaluation.")

history = model.fit(
    X_train_scaled,
    y_train_smote,
    epochs=60,
    batch_size=32,
    validation_split=0.20,
    callbacks=[early_stop],
    verbose=1
)


# ============================================================
# STEP 10: Evaluate ANN
# ============================================================

print("\n" + "=" * 60)
print("ANN Evaluation Results")
print("=" * 60)

y_pred_prob = model.predict(X_test_scaled).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

print(classification_report(
    y_test,
    y_pred,
    target_names=["Legitimate", "Fraud"],
    zero_division=0
))

acc = accuracy_score(y_test, y_pred)
precision_fraud = report_dict["1"]["precision"]
recall_fraud = report_dict["1"]["recall"]
f1_fraud = report_dict["1"]["f1-score"]
roc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy          : {acc:.4f}")
print(f"Precision (Fraud) : {precision_fraud:.4f}")
print(f"Recall (Fraud)    : {recall_fraud:.4f}")
print(f"F1-score (Fraud)  : {f1_fraud:.4f}")
print(f"ROC-AUC           : {roc:.4f}")


# ============================================================
# STEP 11: Save ANN classification report
# ============================================================

ann_report_df = pd.DataFrame(report_dict).transpose()
ann_report_df.to_csv("ann_classification_report.csv")

print("\nSaved: ann_classification_report.csv")


# ============================================================
# STEP 12: Confusion Matrix
# ============================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — ANN")
plt.tight_layout()
plt.savefig("confusion_matrix_ANN.png", dpi=300)
plt.show()

print("Saved: confusion_matrix_ANN.png")


# ============================================================
# STEP 13: Training history graph
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history["loss"], label="Train Loss")
ax1.plot(history.history["val_loss"], label="Validation Loss")
ax1.set_title("ANN Training and Validation Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

ax2.plot(history.history["accuracy"], label="Train Accuracy")
ax2.plot(history.history["val_accuracy"], label="Validation Accuracy")
ax2.set_title("ANN Training and Validation Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("ann_training_history.png", dpi=300)
plt.show()

print("Saved: ann_training_history.png")


# ============================================================
# STEP 14: ROC Curve for ANN
# ============================================================

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ANN (AUC = {roc:.4f})")
plt.plot([0, 1], [0, 1], "k--", label="Random baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — ANN")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_ANN.png", dpi=300)
plt.show()

print("Saved: roc_curve_ANN.png")


# ============================================================
# STEP 15: ANN summary row
# ============================================================

ann_row = pd.DataFrame([{
    "Model": "ANN",
    "Accuracy": round(acc, 4),
    "Precision (Fraud)": round(precision_fraud, 4),
    "Recall (Fraud)": round(recall_fraud, 4),
    "F1-score (Fraud)": round(f1_fraud, 4),
    "ROC-AUC": round(roc, 4)
}])

ann_row.to_csv("ann_model_summary.csv", index=False)

print("\nSaved: ann_model_summary.csv")


# ============================================================
# STEP 16: Compare ANN with baseline models
# ============================================================

try:
    baseline_summary = pd.read_csv("model_summary.csv")

    # Make sure column names are consistent
    if "F1 (Fraud)" in baseline_summary.columns:
        baseline_summary = baseline_summary.rename(
            columns={"F1 (Fraud)": "F1-score (Fraud)"}
        )

    full_summary = pd.concat(
        [baseline_summary, ann_row],
        ignore_index=True
    )

    print("\n" + "=" * 70)
    print("FULL MODEL COMPARISON: BASELINE MODELS + ANN")
    print("=" * 70)
    print(full_summary.to_string(index=False))

    full_summary.to_csv("full_model_summary.csv", index=False)

    print("\nSaved: full_model_summary.csv")

except FileNotFoundError:
    print("\nmodel_summary.csv not found.")
    print("Run main_fixed.py first if you want baseline + ANN comparison.")


# ============================================================
# STEP 17: Save ANN model
# ============================================================

model.save("ann_model.keras")

print("\nANN model saved to ann_model.keras")
print("Training and evaluation completed successfully.")
print("The deployed Streamlit application continues to use the XGBoost preprocessing artefacts produced by main_fixed.py.")