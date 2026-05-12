# Heart Disease Prediction using machine learning
# ------------------------------------------------
# Install required libraries:
# Pip install pandas numpy sckit - learn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score



# ------------------------------------------------
# Load Dataset
# -----------------------------------------------
# Download dataset from kaggle 
# Example file name: heart_disease_data(1).csv
heart_dataset = pd.read_csv('heart_data.csv')


# Display First 5 rows
print("\nHeart Dataset Preview: \n")
print(heart_dataset.head())



# Split Features and Testing data
X = heart_dataset.drop(columns='target')
Y = heart_dataset['target']


# Split Training and Testing Data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state = 2)


# Train Model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, Y_train)



# Model Accuracy

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)



print("\nAccuracy on Training Data :", training_data_accuracy)


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)



print("Accuracy on Test Data", test_data_accuracy)



# Predict Heart Disease from User input:
print("\nEnter Patient Details:")


age = float(input("Age: "))
sex = float(input("Sex (1 = Male, 0 = Female): "))
cp = float(input("Chest Pain Type (0-3): "))
trestbps = float(input("Resting Blood Pressure: "))
chol = float(input("Cholesterol: "))
fbs = float(input("Fasting Blood Sugar > 120 (1 = Yes, 0 = No): "))
restecg = float(input("Rest ECG Results (0-2): "))
thalach = float(input("Maximum Heart Rate Achieved: "))
exang = float(input("Exercise Induced Angina (1 = Yes, 0 = No): "))
oldpeak = float(input("ST Depression: "))
slope = float(input("Slope of Peak Exercise ST Segment: "))
ca = float(input("Major Vessels Colored (0-3): "))
thal = float(input("Thal (0-3): "))




# Create Input Data
input_data = [[
    age, sex, cp, trestbps, chol,
    fbs, restecg, thalach,
    exang, oldpeak, slope, ca, thal
]]

# Prediction
prediction = model.predict(input_data)

print("\nPrediction Result:")

if prediction[0] == 1:
    print("Person has Heart Disease")
else:
    print("Person does NOT have Heart Disease")