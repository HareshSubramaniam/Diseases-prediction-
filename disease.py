
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

heart_df=pd.read_csv('/content/heart.csv')

diab_df=pd.read_csv("/content/diabetes.csv")

cancer_df=pd.read_csv("/content/cancer_classification.csv")

heart_df.head()

diab_df.head()

cancer_df.head()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

feature_cols = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal"
]

X = heart_df[feature_cols].values
y = heart_df["target"].values

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


model = LogisticRegression(max_iter=500)
model.fit(x_train_scaled, y_train)

print("Model Accuracy:", model.score(x_test_scaled, y_test))


patients = heart_df.loc[0:5, feature_cols].values


patients_scaled = scaler.transform(patients)


probs = model.predict_proba(patients_scaled)[:, 1]


for idx, prob in zip(range(0, 2), probs):
    print(f"Patient {idx+1}: Predicted heart disease possibility = {prob*100:.2f}%")

import matplotlib.pyplot as plt
import numpy as np
probs = model.predict_proba(patients_scaled)[:, 1]
percentages = np.round(probs * 100)
patient_labels = [f"P{i+1}" for i in range(len(percentages))]
plt.figure(figsize=(8,5))
bars = plt.bar(patient_labels, percentages, color='pink')
plt.ylim(0, 100)
plt.ylabel("Probability out of 100 (%)")
plt.title("Heart  Disease Prediction Probabilities")
for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{int(pct)}%", ha='center', va='bottom')

plt.show()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
feature_cols = [
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "worst texture",
    "worst perimeter",
    "worst smoothness",
    "worst compactness"
]
X = cancer_df[feature_cols].values
y = cancer_df["benign_0__mal_1"].values

x_can_train, x_can_test, y_can_train, y_can_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
x_can_train_scaled = scaler.fit_transform(x_can_train)
x_can_test_scaled = scaler.transform(x_can_test)

model = LogisticRegression(max_iter=500)
model.fit(x_can_train_scaled, y_can_train)

print("Model Accuracy:", model.score(x_can_test_scaled, y_can_test))

patients = cancer_df.loc[0:5, feature_cols].values

patients_scaled= scaler.transform(patients)

probs = model.predict_proba(patients_scaled)[:, 1]

for idx, prob in zip(range(0, 2), probs):
    print(f"Patient {idx+1}: Predicted cancer possibility = {prob*100:.2f}%")

import matplotlib.pyplot as plt
import numpy as np
probs = model.predict_proba(patients_scaled)[:, 1]
percentages = np.round(probs * 100)
patient_labels = [f"P{i+1}" for i in range(len(percentages))]
plt.figure(figsize=(8,5))
bars = plt.bar(patient_labels, percentages, color='red')
plt.ylim(0, 100)
plt.ylabel("Probability out of 100 (%)")
plt.title("cancer Disease Prediction Probabilities")
for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{int(pct)}%", ha='center', va='bottom')

plt.show()

from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
y_can_pred=model.predict(x_can_test_scaled)
acc=accuracy_score(y_can_test,y_can_pred)
print("Accuracy:",acc)
print("Classification Report:\n",classification_report(y_can_test,y_can_pred))
y_can_pred = model.predict(x_can_test_scaled)
print("Confusion Matrix:\n", confusion_matrix(y_can_test, y_can_pred))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

feature_cols = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

X = diab_df[feature_cols].values
y = diab_df["Outcome"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=500)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)



print("Accuracy:", acc)
for idx, prob in zip(range(0, 10), probs):
    print(f"Patient {idx}: Predicted diabetes possibility = {prob*100:.2f}%")


print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
patients = diab_df.loc[0:4, feature_cols].values
patients_scaled= scaler.transform(patients)
probs = model.predict_proba(patients_scaled)[:, 1]

import matplotlib.pyplot as plt
import numpy as np
probs = model.predict_proba(patients_scaled)[:, 1]
percentages = np.round(probs * 100)
patient_labels = [f"P{i+1}" for i in range(len(percentages))]
plt.figure(figsize=(8,5))
bars = plt.bar(patient_labels, percentages, color='blue')
plt.ylim(0, 100)
plt.ylabel("Probability out of 100 (%)")
plt.title("Diabetes Disease Prediction Probabilities")
for bar, pct in zip(bars, percentages):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{int(pct)}%", ha='center', va='bottom')

plt.show()

