import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Excel Dataset
df = pd.read_excel("C:/Users/acer/Downloads/Student_Performance_Dataset_50000.xlsx")

print(df.head())

# Remove Student ID
df = df.drop("StudentID", axis=1)

# Encode categorical columns
encoder_gender = LabelEncoder()
encoder_department = LabelEncoder()
encoder_target = LabelEncoder()

df["Gender"] = encoder_gender.fit_transform(df["Gender"])
df["Department"] = encoder_department.fit_transform(df["Department"])
df["Performance"] = encoder_target.fit_transform(df["Performance"])

# Features and Target
X = df.drop("Performance", axis=1)
y = df["Performance"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Train Model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Save Model
joblib.dump(model, "student_model.pkl")
joblib.dump(encoder_gender, "gender_encoder.pkl")
joblib.dump(encoder_department, "department_encoder.pkl")
joblib.dump(encoder_target, "target_encoder.pkl")

print("\nModel Saved Successfully")