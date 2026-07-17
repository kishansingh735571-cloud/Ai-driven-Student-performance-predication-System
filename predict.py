import joblib
import pandas as pd

# Load Model
model = joblib.load("student_model.pkl")
gender_encoder = joblib.load("gender_encoder.pkl")
department_encoder = joblib.load("department_encoder.pkl")
target_encoder = joblib.load("target_encoder.pkl")

print("========== Student Performance Prediction ==========")

# Input
age = int(input("Age: "))

# Accept Male/male/MALE
gender = input("Gender (Male/Female): ").strip().lower()

if gender == "male":
    gender = "Male"
elif gender == "female":
    gender = "Female"
else:
    print("Invalid Gender!")
    exit()

# Accept CSE/cse/Cse etc.
department = input("Department (CSE/IT/ECE/EEE/ME/CE): ").strip().upper()

valid_departments = ["CSE", "IT", "ECE", "EEE", "ME", "CE"]

if department not in valid_departments:
    print("Invalid Department!")
    exit()

semester = int(input("Semester: "))
study_hours = float(input("Study Hours: "))
attendance = int(input("Attendance: "))
assignments = int(input("Assignments: "))
quiz = int(input("Quiz Score: "))
internal = int(input("Internal Marks: "))
previous = int(input("Previous Marks: "))
sleep = float(input("Sleep Hours: "))

# Encode
gender = gender_encoder.transform([gender])[0]
department = department_encoder.transform([department])[0]

# Create DataFrame
data = pd.DataFrame({
    "Age": [age],
    "Gender": [gender],
    "Department": [department],
    "Semester": [semester],
    "StudyHours": [study_hours],
    "Attendance": [attendance],
    "Assignments": [assignments],
    "QuizScore": [quiz],
    "InternalMarks": [internal],
    "PreviousMarks": [previous],
    "SleepHours": [sleep]
})

# Prediction
prediction = model.predict(data)

# Convert prediction back to label
result = target_encoder.inverse_transform(prediction)

print("\n==============================")
print("Predicted Performance :", result[0])
print("==============================")