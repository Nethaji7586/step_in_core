import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# 1️⃣ Load CSV
df = pd.read_csv("loan.csv")  # Unga file path

# 2️⃣ Features and target
X = df.drop(columns=['Approved'])
y = df['Approved']

# 3️⃣ Convert text to numbers
X = pd.get_dummies(X)  # Gender -> Male/Female encoded as 0/1

# 4️⃣ Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Train RandomForest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 6️⃣ Test model
y_pred = model.predict(X_test)
print("✅ Classification Report:\n")
print(classification_report(y_test, y_pred))

# 7️⃣ Save model
joblib.dump(model, "loan_model.joblib")
print("\n✅ Model saved as loan_model.joblib")
