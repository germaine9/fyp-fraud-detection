import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

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

# ─────────────────────────────────────────
# STEP 1: Load dataset
# ─────────────────────────────────────────
df = pd.read_csv("healthcare_fraud_detection.csv")
print("Dataset loaded:", df.shape)
print("\nClass distribution:")
print(df['Is_Fraud'].value_counts())

# ─────────────────────────────────────────
# STEP 2: Drop useless columns
# ─────────────────────────────────────────
df = df.drop(['Provider_ID', 'Claim_ID', 'Claim_Submission_Date'], axis=1)

# ─────────────────────────────────────────
# STEP 3: Handle missing values properly
# ─────────────────────────────────────────
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ─────────────────────────────────────────
# STEP 4: Feature Engineering
# ─────────────────────────────────────────
# Claim-to-cost ratio: flags providers billing disproportionately high amounts
df['claim_to_cost_ratio'] = df['Claim_Amount'] / (df['Approved_Amount'] + 1)

# Cost outlier flag: IQR-based binary flag for extreme claim amounts
Q1 = df['Claim_Amount'].quantile(0.25)
Q3 = df['Claim_Amount'].quantile(0.75)
IQR = Q3 - Q1
df['cost_outlier_flag'] = (df['Claim_Amount'] > Q3 + 1.5 * IQR).astype(int)

# Claim frequency flag: flags providers with unusually high monthly claims
df['high_claim_frequency'] = (
    df['Number_of_Claims_Per_Provider_Monthly'] >
    df['Number_of_Claims_Per_Provider_Monthly'].quantile(0.90)
).astype(int)

print("\nFeature engineering done. New shape:", df.shape)

# ─────────────────────────────────────────
# STEP 5: Convert text columns to numbers
# ─────────────────────────────────────────
df = pd.get_dummies(df)

# ─────────────────────────────────────────
# STEP 6: Split features and target
# ─────────────────────────────────────────
X = df.drop('Is_Fraud', axis=1)
y = df['Is_Fraud']

# ─────────────────────────────────────────
# STEP 7: Train-test split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# STEP 8: SMOTE — handle class imbalance
# (must be BEFORE scaling)
# ─────────────────────────────────────────
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# ─────────────────────────────────────────
# STEP 9: Feature scaling
# (must be AFTER SMOTE)
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler for Streamlit app later
joblib.dump(scaler, 'scaler.pkl')
print("\nScaler saved to scaler.pkl")

# ─────────────────────────────────────────
# STEP 10: Define all models
# ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# ─────────────────────────────────────────
# STEP 11: Train, evaluate, and save results
# ─────────────────────────────────────────
model_names = []
accuracies = []
recalls = []
f1_scores = []
roc_aucs = []

saved_models = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print('='*50)

    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, pred, output_dict=True)
    acc = accuracy_score(y_test, pred)
    roc = roc_auc_score(y_test, prob)

    print(classification_report(y_test, pred))
    print(f"Accuracy : {acc:.4f}")
    print(f"ROC-AUC  : {roc:.4f}")

    model_names.append(name)
    accuracies.append(acc)
    recalls.append(report['1']['recall'])
    f1_scores.append(report['1']['f1-score'])
    roc_aucs.append(roc)

    saved_models[name] = model

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Legitimate', 'Fraud'],
                yticklabels=['Legitimate', 'Fraud'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"confusion_matrix_{name.replace(' ', '_')}.png")
    plt.show()
    print(f"Saved: confusion_matrix_{name.replace(' ', '_')}.png")

# Save all models
joblib.dump(saved_models, 'baseline_models.pkl')
print("\nAll baseline models saved to baseline_models.pkl")

# ─────────────────────────────────────────
# STEP 12: Accuracy comparison bar chart
# ─────────────────────────────────────────
x = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, accuracies, width, label='Accuracy')
ax.bar(x,         recalls,    width, label='Recall (Fraud)')
ax.bar(x + width, f1_scores,  width, label='F1-Score (Fraud)')

ax.set_xlabel("Model")
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Accuracy, Recall, F1-Score")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0, 1.1)
ax.legend()
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()
print("Saved: model_comparison.png")

# ─────────────────────────────────────────
# STEP 13: ROC Curve comparison
# ─────────────────────────────────────────
plt.figure(figsize=(8, 6))

for name, model in saved_models.items():
    prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc = roc_auc_score(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.2f})")

plt.plot([0, 1], [0, 1], 'k--', label='Random baseline')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison — All Baseline Models")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_comparison.png")
plt.show()
print("Saved: roc_curve_comparison.png")

# ─────────────────────────────────────────
# STEP 14: Summary table
# ─────────────────────────────────────────
summary = pd.DataFrame({
    'Model': model_names,
    'Accuracy': [round(a, 4) for a in accuracies],
    'Recall (Fraud)': [round(r, 4) for r in recalls],
    'F1 (Fraud)': [round(f, 4) for f in f1_scores],
    'ROC-AUC': [round(r, 4) for r in roc_aucs]
})

print("\n" + "="*60)
print("FINAL SUMMARY TABLE")
print("="*60)
print(summary.to_string(index=False))
summary.to_csv("model_summary.csv", index=False)
print("\nSaved: model_summary.csv")