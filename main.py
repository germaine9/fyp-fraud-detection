# main_fixed.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


DATASET_FILE = "healthcare_fraud_detection.csv"
DATASET_SOURCE = "https://www.kaggle.com/datasets/esseasd/healthcare-fraud-detection-dataset"
REQUIRED_COLUMNS = {
    "Is_Fraud", "Claim_Amount", "Approved_Amount",
    "Number_of_Claims_Per_Provider_Monthly"
}


def validate_dataset(dataframe: pd.DataFrame) -> None:
    """Fail early when the supplied dataset cannot support the documented pipeline."""
    missing = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    labels = set(pd.Series(dataframe["Is_Fraud"]).dropna().unique().tolist())
    if not labels.issubset({0, 1}) or not labels:
        raise ValueError("Is_Fraud must contain binary labels 0 and 1.")


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
    Fit preprocessing only on training data.
    This avoids data leakage from the test set.
    """

    X_train = X_train.copy()

    # Identify numeric and categorical columns
    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include="object").columns.tolist()

    # Store training-set means and modes
    numeric_means = X_train[numeric_cols].mean()

    if len(categorical_cols) > 0:
        categorical_modes = X_train[categorical_cols].mode().iloc[0]
    else:
        categorical_modes = pd.Series(dtype="object")

    # Fill missing numeric values using training mean
    for col in numeric_cols:
        X_train[col] = X_train[col].fillna(numeric_means[col])

    # Fill missing categorical values using training mode
    for col in categorical_cols:
        X_train[col] = X_train[col].fillna(categorical_modes[col])

    # Feature engineering thresholds learned from training data only
    claim_q1 = X_train["Claim_Amount"].quantile(0.25)
    claim_q3 = X_train["Claim_Amount"].quantile(0.75)
    claim_iqr = claim_q3 - claim_q1

    high_claim_threshold = X_train["Number_of_Claims_Per_Provider_Monthly"].quantile(0.90)

    # Feature engineering
    X_train["claim_to_cost_ratio"] = (
        X_train["Claim_Amount"] / (X_train["Approved_Amount"] + 1)
    )

    X_train["cost_outlier_flag"] = (
        X_train["Claim_Amount"] > claim_q3 + 1.5 * claim_iqr
    ).astype(int)

    X_train["high_claim_frequency"] = (
        X_train["Number_of_Claims_Per_Provider_Monthly"] > high_claim_threshold
    ).astype(int)

    # Convert categorical columns to dummy variables
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
    Apply the same preprocessing rules learned from training data
    to validation, test, or new uploaded data.
    """

    X_data = X_data.copy()

    numeric_cols = preprocess_info["numeric_cols"]
    categorical_cols = preprocess_info["categorical_cols"]
    numeric_means = preprocess_info["numeric_means"]
    categorical_modes = preprocess_info["categorical_modes"]

    # Fill missing numeric values
    for col in numeric_cols:
        if col in X_data.columns:
            X_data[col] = X_data[col].fillna(numeric_means[col])

    # Fill missing categorical values
    for col in categorical_cols:
        if col in X_data.columns:
            X_data[col] = X_data[col].fillna(categorical_modes[col])

    claim_q3 = preprocess_info["claim_q3"]
    claim_iqr = preprocess_info["claim_iqr"]
    high_claim_threshold = preprocess_info["high_claim_threshold"]

    # Apply same feature engineering
    X_data["claim_to_cost_ratio"] = (
        X_data["Claim_Amount"] / (X_data["Approved_Amount"] + 1)
    )

    X_data["cost_outlier_flag"] = (
        X_data["Claim_Amount"] > claim_q3 + 1.5 * claim_iqr
    ).astype(int)

    X_data["high_claim_frequency"] = (
        X_data["Number_of_Claims_Per_Provider_Monthly"] > high_claim_threshold
    ).astype(int)

    # Convert categorical columns to dummy variables
    X_data = pd.get_dummies(X_data)

    # Match training feature columns exactly
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

joblib.dump(preprocess_info, "preprocess_info.pkl")

print("\nPreprocessing completed.")
print("Processed training shape:", X_train_processed.shape)
print("Processed testing shape:", X_test_processed.shape)
print("Saved: preprocess_info.pkl")


# ============================================================
# STEP 6: SMOTE — handle class imbalance on training set only
# ============================================================

fraud_before = y_train.value_counts()[1]
legit_before = y_train.value_counts()[0]

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)

fraud_after = pd.Series(y_train_smote).value_counts()[1]
legit_after = pd.Series(y_train_smote).value_counts()[0]

print("\nBefore SMOTE:")
print(f"Legitimate: {legit_before} | Fraud: {fraud_before}")

print("\nAfter SMOTE:")
print(f"Legitimate: {legit_after} | Fraud: {fraud_after}")


# ============================================================
# STEP 7: SMOTE before vs after chart
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.bar(
    ["Legitimate", "Fraud"],
    [legit_before, fraud_before],
    color=["steelblue", "salmon"],
    edgecolor="white"
)
ax1.set_title("Before SMOTE")
ax1.set_ylabel("Number of Records")
ax1.set_ylim(0, legit_after + 500)

for i, v in enumerate([legit_before, fraud_before]):
    ax1.text(i, v + 100, str(v), ha="center", fontweight="bold")


ax2.bar(
    ["Legitimate", "Fraud"],
    [legit_after, fraud_after],
    color=["steelblue", "salmon"],
    edgecolor="white"
)
ax2.set_title("After SMOTE")
ax2.set_ylabel("Number of Records")
ax2.set_ylim(0, legit_after + 500)

for i, v in enumerate([legit_after, fraud_after]):
    ax2.text(i, v + 100, str(v), ha="center", fontweight="bold")

plt.suptitle(
    "Class Distribution Before and After SMOTE",
    fontsize=13,
    fontweight="bold"
)
plt.tight_layout()
plt.savefig("smote_comparison.png", dpi=300)
plt.show()

print("Saved: smote_comparison.png")


# ============================================================
# STEP 8: Feature scaling
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled = scaler.transform(X_test_processed)

joblib.dump(scaler, "scaler.pkl")

print("\nScaler saved to scaler.pkl")


# ============================================================
# STEP 9: Define baseline models
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=42)
}


# ============================================================
# STEP 10: Train, evaluate, and save results
# ============================================================

model_names = []
accuracies = []
precisions = []
recalls = []
f1_scores = []
roc_aucs = []

saved_models = {}

for name, model in models.items():
    print("\n" + "=" * 60)
    print(f"Training: {name}")
    print("=" * 60)

    # Train model
    model.fit(X_train_scaled, y_train_smote)

    # Predictions
    pred = model.predict(X_test_scaled)
    prob = model.predict_proba(X_test_scaled)[:, 1]

    # Evaluation
    report = classification_report(y_test, pred, output_dict=True, zero_division=0)
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, prob)

    precision_fraud = report["1"]["precision"]
    recall_fraud = report["1"]["recall"]
    f1_fraud = report["1"]["f1-score"]

    print(classification_report(y_test, pred, zero_division=0))
    print(f"Accuracy          : {acc:.4f}")
    print(f"Precision (Fraud) : {precision_fraud:.4f}")
    print(f"Recall (Fraud)    : {recall_fraud:.4f}")
    print(f"F1-score (Fraud)  : {f1_fraud:.4f}")
    print(f"ROC-AUC           : {roc:.4f}")

    # Store results
    model_names.append(name)
    accuracies.append(acc)
    precisions.append(precision_fraud)
    recalls.append(recall_fraud)
    f1_scores.append(f1_fraud)
    roc_aucs.append(roc)

    saved_models[name] = model

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)

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
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()

    filename = f"confusion_matrix_{name.replace(' ', '_')}.png"
    plt.savefig(filename, dpi=300)
    plt.show()

    print(f"Saved: {filename}")


joblib.dump(saved_models, "baseline_models.pkl")
print("\nAll baseline models saved to baseline_models.pkl")


# ============================================================
# STEP 11: Model comparison chart
# ============================================================

x = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))

ax.bar(x - width, accuracies, width, label="Accuracy", color="steelblue")
ax.bar(x, recalls, width, label="Recall (Fraud)", color="salmon")
ax.bar(x + width, f1_scores, width, label="F1-score (Fraud)", color="mediumseagreen")

ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Baseline Model Comparison — Accuracy, Recall, and F1-score")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1)
ax.legend()

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=300)
plt.show()

print("Saved: model_comparison.png")


# ============================================================
# STEP 12: ROC curve comparison
# ============================================================

plt.figure(figsize=(8, 6))

for name, model in saved_models.items():
    prob = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)

    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], "k--", label="Random baseline")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison — Baseline Models")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison.png", dpi=300)
plt.show()

print("Saved: roc_curve_comparison.png")


# ============================================================
# STEP 13: Summary table
# ============================================================

summary = pd.DataFrame({
    "Model": model_names,
    "Accuracy": [round(a, 4) for a in accuracies],
    "Precision (Fraud)": [round(p, 4) for p in precisions],
    "Recall (Fraud)": [round(r, 4) for r in recalls],
    "F1-score (Fraud)": [round(f, 4) for f in f1_scores],
    "ROC-AUC": [round(r, 4) for r in roc_aucs]
})

print("\n" + "=" * 70)
print("FINAL BASELINE MODEL SUMMARY TABLE")
print("=" * 70)
print(summary.to_string(index=False))

summary.to_csv("model_summary.csv", index=False)

print("\nSaved: model_summary.csv")


# ============================================================
# STEP 14: Supplementary cross-validation
# SMOTE and scaling are fitted inside each fold. The imputation/encoding
# representation was learned from the overall training partition before this
# supplementary CV stage; the completely untouched test set remains the main
# final evaluation evidence documented in the report.
# ============================================================

print("\n" + "=" * 70)
print("SUPPLEMENTARY CROSS VALIDATION — 5-Fold Recall Scores")
print("=" * 70)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("smote", SMOTE(random_state=42)),
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X_train_processed,
        y_train,
        cv=cv,
        scoring="recall"
    )

    cv_results[name] = scores

    print(f"\n{name}:")
    print(f"Fold scores : {scores.round(4)}")
    print(f"Mean Recall : {scores.mean():.4f}")
    print(f"Std Dev     : {scores.std():.4f}")


# ============================================================
# STEP 15: Cross-validation chart
# ============================================================

cv_means = [cv_results[name].mean() for name in model_names]
cv_stds = [cv_results[name].std() for name in model_names]

plt.figure(figsize=(10, 5))

bars = plt.bar(
    model_names,
    cv_means,
    yerr=cv_stds,
    color="steelblue",
    edgecolor="white",
    capsize=5
)

plt.ylabel("Mean Recall")
plt.title("5-Fold Cross-Validation — Fraud Recall with Standard Deviation")
plt.ylim(0, 1.1)

for bar, mean in zip(bars, cv_means):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        mean + 0.03,
        f"{mean:.2f}",
        ha="center",
        fontweight="bold"
    )

plt.tight_layout()
plt.savefig("cross_validation_recall.png", dpi=300)
plt.show()

print("Saved: cross_validation_recall.png")


# ============================================================
# STEP 16: Save the highest-recall model and deployed XGBoost model
# ============================================================

best_index = np.argmax(recalls)
best_model_name = model_names[best_index]
best_model = saved_models[best_model_name]

joblib.dump(best_model, "highest_recall_baseline_model.pkl")
joblib.dump(saved_models["XGBoost"], "deployed_xgboost_model.pkl")

print("\n" + "=" * 70)
print("HIGHEST-RECALL BASELINE MODEL")
print("=" * 70)
print(f"Best model based on fraud recall: {best_model_name}")
print(f"Fraud recall: {recalls[best_index]:.4f}")
print("Saved: highest_recall_baseline_model.pkl")
print("Saved: deployed_xgboost_model.pkl")


print("\nTraining and evaluation completed successfully.")