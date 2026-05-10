import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

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

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# ─────────────────────────────────────────
# STEP 1: Load dataset
# ─────────────────────────────────────────
df = pd.read_csv("healthcare_fraud_detection.csv")
print("Dataset loaded:", df.shape)

# ─────────────────────────────────────────
# STEP 2: Drop useless columns
# ─────────────────────────────────────────
df = df.drop(['Provider_ID', 'Claim_ID', 'Claim_Submission_Date'], axis=1)

# ─────────────────────────────────────────
# STEP 3: Handle missing values
# ─────────────────────────────────────────
for col in df.select_dtypes(include='number').columns:
    df[col] = df[col].fillna(df[col].mean())

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].fillna(df[col].mode()[0])

# ─────────────────────────────────────────
# STEP 4: Feature Engineering
# ─────────────────────────────────────────
df['claim_to_cost_ratio'] = df['Claim_Amount'] / (df['Approved_Amount'] + 1)

Q1 = df['Claim_Amount'].quantile(0.25)
Q3 = df['Claim_Amount'].quantile(0.75)
IQR = Q3 - Q1
df['cost_outlier_flag'] = (df['Claim_Amount'] > Q3 + 1.5 * IQR).astype(int)

df['high_claim_frequency'] = (
    df['Number_of_Claims_Per_Provider_Monthly'] >
    df['Number_of_Claims_Per_Provider_Monthly'].quantile(0.90)
).astype(int)

print("Feature engineering done. New shape:", df.shape)

# ─────────────────────────────────────────
# STEP 5: Convert text to numbers
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
# STEP 8: SMOTE (before scaling)
# ─────────────────────────────────────────
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)
print("\nAfter SMOTE:")
print(pd.Series(y_train).value_counts())

# ─────────────────────────────────────────
# STEP 9: Feature scaling (after SMOTE)
# ─────────────────────────────────────────
scaler = joblib.load('scaler.pkl')
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)

# ─────────────────────────────────────────
# STEP 10: Build ANN model
# ─────────────────────────────────────────
def build_ann(input_dim):
    model = Sequential([

        # Hidden layer 1
        Dense(64, activation='relu', input_dim=input_dim),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden layer 2
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),

        # Output layer
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model

input_dim = X_train.shape[1]
model = build_ann(input_dim)
model.summary()

# ─────────────────────────────────────────
# STEP 11: Train ANN
# ─────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

print("\nTraining ANN...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop],
    verbose=1
)

# ─────────────────────────────────────────
# STEP 12: Evaluate ANN
# ─────────────────────────────────────────
print("\n" + "="*50)
print("ANN Evaluation Results")
print("="*50)

y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype(int)

print(classification_report(y_test, y_pred,
      target_names=['Legitimate', 'Fraud']))

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_pred_prob)

print(f"Accuracy : {acc:.4f}")
print(f"ROC-AUC  : {roc:.4f}")

# ─────────────────────────────────────────
# STEP 13: Confusion Matrix
# ─────────────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Legitimate', 'Fraud'],
            yticklabels=['Legitimate', 'Fraud'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — ANN")
plt.tight_layout()
plt.savefig("confusion_matrix_ANN.png")
plt.show()
print("Saved: confusion_matrix_ANN.png")

# ─────────────────────────────────────────
# STEP 14: Training history graph
# ─────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss curve
ax1.plot(history.history['loss'],     label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_title("ANN Training Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()

# Accuracy curve
ax2.plot(history.history['accuracy'],     label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy')
ax2.set_title("ANN Training Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.legend()

plt.tight_layout()
plt.savefig("ann_training_history.png")
plt.show()
print("Saved: ann_training_history.png")

# ─────────────────────────────────────────
# STEP 15: ROC Curve for ANN
# ─────────────────────────────────────────
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, label=f"ANN (AUC = {roc:.4f})", color='purple')
plt.plot([0, 1], [0, 1], 'k--', label='Random baseline')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — ANN")
plt.legend()
plt.tight_layout()
plt.savefig("roc_curve_ANN.png")
plt.show()
print("Saved: roc_curve_ANN.png")

# ─────────────────────────────────────────
# STEP 16: Compare ANN vs baseline models
# ─────────────────────────────────────────
baseline_summary = pd.read_csv("model_summary.csv")

ann_row = pd.DataFrame([{
    'Model': 'ANN',
    'Accuracy': round(acc, 4),
    'Recall (Fraud)': round(
        classification_report(y_test, y_pred,
        output_dict=True)['1']['recall'], 4),
    'F1 (Fraud)': round(
        classification_report(y_test, y_pred,
        output_dict=True)['1']['f1-score'], 4),
    'ROC-AUC': round(roc, 4)
}])

full_summary = pd.concat([baseline_summary, ann_row], ignore_index=True)

print("\n" + "="*65)
print("FULL MODEL COMPARISON (Baseline + ANN)")
print("="*65)
print(full_summary.to_string(index=False))
full_summary.to_csv("full_model_summary.csv", index=False)
print("\nSaved: full_model_summary.csv")

# ─────────────────────────────────────────
# STEP 17: Save ANN model
# ─────────────────────────────────────────
model.save("ann_model.h5")
print("\nANN model saved to ann_model.h5")
print("Done! Run blockchain.py next.")