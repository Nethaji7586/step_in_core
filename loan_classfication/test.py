import pandas as pd
import joblib

model = joblib.load("loan_model.joblib")

new_data = {
    'Age': [14],
    'Gender': ['Female'],
    'Salary': [600],
    'CreditScore': [670]
}

# 3️⃣ Convert to DataFrame
df_new = pd.DataFrame(new_data)

# 4️⃣ Convert categorical text to numbers (same as training)
df_new = pd.get_dummies(df_new)

# 5️⃣ Align columns with training data (important!)
# Missing columns = 0
for col in model.feature_names_in_:
    if col not in df_new.columns:
        df_new[col] = 0

df_new = df_new[model.feature_names_in_]

# 6️⃣ Predict
prediction = model.predict(df_new)[0]
probabilities = model.predict_proba(df_new)[0]

print("Loan Approved:", prediction)
