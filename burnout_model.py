import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Load Dataset
data = pd.read_csv("burnout_dataset.csv")

# Separate Features and Target
X = data.drop("burnout", axis=1)
y = data["burnout"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Logistic Regression Model
log_model = LogisticRegression(max_iter=200)
log_model.fit(X_train, y_train)

# Train Random Forest Model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
log_pred = log_model.predict(X_test)
rf_pred = rf_model.predict(X_test)

# Evaluation
print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

print("\nClassification Report (Random Forest):")
print(classification_report(y_test, rf_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Manual Prediction System
print("\nBurnout Risk Prediction System")

study = float(input("Study hours per day: "))
sleep = float(input("Sleep hours: "))
screen = float(input("Screen time hours: "))
exercise = float(input("Exercise minutes: "))
assignment = float(input("Assignment load (1-10): "))
anxiety = float(input("Anxiety level (1-10): "))

user_data = np.array([[study, sleep, screen, exercise, assignment, anxiety]])

prediction = rf_model.predict(user_data)

if prediction[0] == 0:
    print("Burnout Risk: LOW")
elif prediction[0] == 1:
    print("Burnout Risk: MEDIUM")
else:
    print("Burnout Risk: HIGH")