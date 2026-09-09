import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv("lung_cancer.csv")

# 🔥 CLEAN COLUMN NAMES PROPERLY
data.columns = data.columns.str.strip()  # remove leading/trailing spaces
data.columns = data.columns.str.replace(" ", "_")  # replace spaces with underscore

print("Columns used for training:")
print(data.columns)

# Convert categorical columns
data['LUNG_CANCER'] = data['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
data['GENDER'] = data['GENDER'].map({'M': 1, 'F': 0})

# Features and target
X = data.drop('LUNG_CANCER', axis=1)
y = data['LUNG_CANCER']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("lung_model.pkl", "wb"))

print("✅ Model retrained successfully!")