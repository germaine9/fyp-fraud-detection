import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import train_test_split, cross_val_score
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
df['claim_to_cost_ratio'] = df['Claim_Amount'] / (df['Approved_Amount'] + 1)

Q1  = df['Claim_Amount'].quantile(0.25)
Q3  = df['Claim_Amount'].quantile(0.75)
IQR = Q3 - Q1
df['cost_outlier_flag'] = (df['Claim_Amount'] > Q3 + 1.5 * IQR).astype(int)

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

# Save original counts BEFORE SMOTE for chart
fraud_before = y_train.value_counts()[1]
legit_before  = y_train.value_counts()[0]

# ─────────────────────────────────────────
# STEP 8: SMOTE — handle class imbalance
# (must be BEFORE scaling)
# ─────────────────────────────────────────
sm = SMOTE(random_state=42)
X_train_smote, y_train_smote = sm.fit_resample(X_train, y_train)

fraud_after = pd.Series(y_train_smote).value_counts()[1]
legit_after  = pd.Series(y_train_smote).value_counts()[0]

print("\nBefore SMOTE:")
print(f"  Legitimate: {legit_before} | Fraud: {fraud_before}")
print("\nAfter SMOTE:")
print(f"  Legitimate: {legit_after} | Fraud: {fraud_after}")

# ─────────────────────────────────────────
# NEW: SMOTE Before vs After Chart
# ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.bar(['Legitimate', 'Fraud'],
        [legit_before, fraud_before],
        color=['steelblue', 'salmon'],
        edgecolor='white')
ax1.set_title("Before SMOTE")
ax1.set_ylabel("Number of Records")
ax1.set_ylim(0, legit_after + 500)
for i, v in enumerate([legit_before, fraud_before]):
    ax1.text(i, v + 100, str(v), ha='center', fontweight='bold')

ax2.bar(['Legitimate', 'Fraud'],
        [legit_after, fraud_after],
        color=['steelblue', 'salmon'],
        edgecolor='white')
ax2.set_title("After SMOTE")
ax2.set_ylabel("Number of Records")
ax2.set_ylim(0, legit_after + 500)
for i, v in enumerate([legit_after, fraud_after]):
    ax2.text(i, v + 100, str(v), ha='center', fontweight='bold')

plt.suptitle("Class Distribution Before and After SMOTE",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("smote_comparison.png")
plt.show()
print("Saved: smote_comparison.png")

# ─────────────────────────────────────────
# STEP 9: Feature scaling (after SMOTE)
# ─────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_smote)
X_test_scaled  = scaler.transform(X_test)

joblib.dump(scaler, 'scaler.pkl')
print("\nScaler saved to scaler.pkl")

# ─────────────────────────────────────────
# STEP 10: Define all models
# ─────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000, random_state=42),
    "Decision Tree":       DecisionTreeClassifier(random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(eval_metric='logloss', random_state=42)
}

# ─────────────────────────────────────────
# STEP 11: Train, evaluate, save results
# ─────────────────────────────────────────
model_names = []
accuracies  = []
recalls     = []
f1_scores   = []
roc_aucs    = []
saved_models = {}

for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print('='*50)

    model.fit(X_train_scaled, y_train_smote)
    pred = model.predict(X_test_scaled)
    prob = model.predict_proba(X_test_scaled)[:, 1]

    report = classification_report(y_test, pred, output_dict=True)
    acc    = accuracy_score(y_test, pred)
    roc    = roc_auc_score(y_test, prob)

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

joblib.dump(saved_models, 'baseline_models.pkl')
print("\nAll baseline models saved to baseline_models.pkl")

# ─────────────────────────────────────────
# STEP 12: Model comparison bar chart
# ─────────────────────────────────────────
x     = np.arange(len(model_names))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - width, accuracies, width, label='Accuracy',       color='steelblue')
ax.bar(x,         recalls,    width, label='Recall (Fraud)', color='salmon')
ax.bar(x + width, f1_scores,  width, label='F1 (Fraud)',     color='mediumseagreen')

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
    prob       = model.predict_proba(X_test_scaled)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, prob)
    auc        = roc_auc_score(y_test, prob)
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
    'Model':          model_names,
    'Accuracy':       [round(a, 4) for a in accuracies],
    'Recall (Fraud)': [round(r, 4) for r in recalls],
    'F1 (Fraud)':     [round(f, 4) for f in f1_scores],
    'ROC-AUC':        [round(r, 4) for r in roc_aucs]
})

print("\n" + "="*60)
print("FINAL SUMMARY TABLE")
print("="*60)
print(summary.to_string(index=False))
summary.to_csv("model_summary.csv", index=False)
print("\nSaved: model_summary.csv")

# ─────────────────────────────────────────
# NEW STEP 15: Cross validation (5-fold)
# ─────────────────────────────────────────
print("\n" + "="*60)
print("CROSS VALIDATION — 5-Fold Recall Scores")
print("="*60)

cv_results = {}

for name, model in models.items():
    scores = cross_val_score(
        model,
        X_train_scaled,
        y_train_smote,
        cv=5,
        scoring='recall'
    )
    cv_results[name] = scores
    print(f"{name}:")
    print(f"  Fold scores : {scores.round(4)}")
    print(f"  Mean Recall : {scores.mean():.4f}")
    print(f"  Std Dev     : {scores.std():.4f}")

# Cross validation bar chart
cv_means = [cv_results[n].mean() for n in model_names]
cv_stds  = [cv_results[n].std()  for n in model_names]

plt.figure(figsize=(10, 5))
bars = plt.bar(model_names, cv_means,
               yerr=cv_stds,
               color='steelblue',
               edgecolor='white',
               capsize=5)
plt.ylabel("Mean Recall (5-Fold CV)")
plt.title("Cross Validation — Mean Recall with Standard Deviation")
plt.ylim(0, 1.1)
for bar, mean in zip(bars, cv_means):
    plt.text(bar.get_x() + bar.get_width() / 2,
             bar.get_height() + 0.02,
             f"{mean:.4f}",
             ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig("cross_validation_recall.png")
plt.show()
print("\nSaved: cross_validation_recall.png")

# ─────────────────────────────────────────
# NEW STEP 16: Feature importance chart
# ─────────────────────────────────────────
print("\n" + "="*60)
print("FEATURE IMPORTANCE — Random Forest")
print("="*60)

rf_model      = saved_models["Random Forest"]
feature_names = list(X.columns)
importances   = rf_model.feature_importances_
indices       = np.argsort(importances)[::-1][:15]

print("Top 15 most important features:")
for i, idx in enumerate(indices):
    print(f"  {i+1:2}. {feature_names[idx]:<45} {importances[idx]:.4f}")

plt.figure(figsize=(10, 6))
plt.barh(
    range(15),
    importances[indices][::-1],
    color='steelblue',
    edgecolor='white'
)
plt.yticks(
    range(15),
    [feature_names[i] for i in indices][::-1]
)
plt.xlabel("Importance Score")
plt.title("Top 15 Most Important Features — Random Forest")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.show()
print("Saved: feature_importance.png")

# ─────────────────────────────────────────
# FINAL: List all saved files
# ─────────────────────────────────────────
print("\n" + "="*60)
print("ALL FILES SAVED")
print("="*60)
files = [
    "scaler.pkl",
    "baseline_models.pkl",
    "model_summary.csv",
    "smote_comparison.png",
    "model_comparison.png",
    "roc_curve_comparison.png",
    "cross_validation_recall.png",
    "feature_importance.png",
    "confusion_matrix_Logistic_Regression.png",
    "confusion_matrix_Decision_Tree.png",
    "confusion_matrix_Random_Forest.png",
    "confusion_matrix_XGBoost.png",
]
for f in files:
    print(f"  ✓ {f}")

print("\nDone! Run ann_model.py next.")