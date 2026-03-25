import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier

data = pd.read_csv("burnout_dataset.csv")

data["study_sleep_ratio"] = data["study_hours"] / (data["sleep_hours"] + 1)
data["stress_index"] = data["assignment_load"] * data["anxiety_level"]
data["health_score"] = data["sleep_hours"] + data["exercise_minutes"] / 10

X = data.drop("burnout", axis=1)
y = data["burnout"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_model = LogisticRegression(max_iter=500)
log_model.fit(X_train_scaled, y_train)

log_pred = log_model.predict(X_test_scaled)

xgb = XGBClassifier(eval_metric='mlogloss')

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

grid = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    verbose=1,
    n_jobs=-1
)

grid.fit(X_train, y_train)

best_xgb = grid.best_estimator_

xgb_pred = best_xgb.predict(X_test)

ensemble = VotingClassifier(
    estimators=[
        ('lr', log_model),
        ('xgb', best_xgb)
    ],
    voting='soft'
)

ensemble.fit(X_train, y_train)

ensemble_pred = ensemble.predict(X_test)

print("Logistic Regression Accuracy:", accuracy_score(y_test, log_pred))
print("XGBoost Accuracy:", accuracy_score(y_test, xgb_pred))
print("Ensemble Accuracy:", accuracy_score(y_test, ensemble_pred))

print("\nBest Parameters:", grid.best_params_)

print("\nClassification Report (XGBoost):")
print(classification_report(y_test, xgb_pred))

cm = confusion_matrix(y_test, xgb_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
plt.title("Confusion Matrix - XGBoost")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

importance = best_xgb.feature_importances_
features = X.columns

plt.figure(figsize=(8, 5))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()

print("\nBurnout Risk Prediction System")

study = float(input("Study hours per day: "))
sleep = float(input("Sleep hours: "))
screen = float(input("Screen time hours: "))
exercise = float(input("Exercise minutes: "))
assignment = float(input("Assignment load (1-10): "))
anxiety = float(input("Anxiety level (1-10): "))

study_sleep_ratio = study / (sleep + 1)
stress_index = assignment * anxiety
health_score = sleep + exercise / 10

user_data = np.array([[
    study, sleep, screen, exercise, assignment, anxiety,
    study_sleep_ratio, stress_index, health_score
]])

prediction = best_xgb.predict(user_data)

if prediction[0] == 0:
    print("Burnout Risk: LOW")
elif prediction[0] == 1:
    print("Burnout Risk: MEDIUM")
else:
    print("Burnout Risk: HIGH")
