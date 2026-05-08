import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from imblearn.over_sampling import SMOTE

# 1. Load dataset
df = pd.read_csv("healthcare_fraud_detection.csv")

# 2. Drop useless columns
df = df.drop(['Provider_ID', 'Claim_ID'], axis=1)

# 3. Handle missing values
df = df.ffill()

# 4. Convert text to numbers
df = pd.get_dummies(df)

# 5. Split features and target
X = df.drop('Is_Fraud', axis=1)
y = df['Is_Fraud']

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 7. Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. Handle imbalance using SMOTE
sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# 9. Models to compare
models = {
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Decision Tree": DecisionTreeClassifier()
}

# Store results for graph
model_names = []
accuracies = []

# 10. Train and evaluate models
for name, model in models.items():

    model.fit(X_train, y_train)

    pred = model.predict(X_test)

    print(f"\n{name}")
    print(classification_report(y_test, pred))

    acc = accuracy_score(y_test, pred)
    print("Accuracy:", acc)

    # Save results
    model_names.append(name)
    accuracies.append(acc)

    # Confusion Matrix
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")

    plt.show()

# 11. Accuracy Comparison Graph
plt.figure(figsize=(8,5))

plt.bar(model_names, accuracies)

plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison")

plt.ylim(0.8, 1.0)

plt.show()