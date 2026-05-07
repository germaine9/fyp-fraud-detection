import pandas as pd

# 1. Load dataset
df = pd.read_csv("healthcare_fraud_detection.csv")

# 2. Drop useless columns
df = df.drop(['Provider_ID', 'Claim_ID'], axis=1)

# 3. Handle missing values
df = df.fillna(method='ffill')

# 4. Convert text to numbers
df = pd.get_dummies(df)

# 5. Split features and target
X = df.drop('Is_Fraud', axis=1)
y = df['Is_Fraud']

# 6. Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from imblearn.over_sampling import SMOTE

sm = SMOTE(random_state=42)
X_train, y_train = sm.fit_resample(X_train, y_train)

# 7. Train model
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 8. Predict
pred = model.predict(X_test)

# 9. Evaluate
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))