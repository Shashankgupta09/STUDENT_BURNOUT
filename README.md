.

📘 Student Burnout Prediction System
📌 Project Description

The Student Burnout Prediction System is a Machine Learning-based project designed to predict burnout risk levels (Low, Medium, High) among students based on academic and lifestyle factors.

This system uses:

📊 Logistic Regression

🌲 Random Forest Classifier

The model analyzes student-related parameters such as:

Study hours

Sleep duration

Screen time

Exercise time

Assignment load

Anxiety level

Based on these inputs, the system predicts the student’s burnout risk category.

The goal of this project is to demonstrate how machine learning can be applied to real-life student wellness problems.

🧠 Problem Statement

Student burnout is a growing issue due to:

High academic pressure

Poor sleep habits

Excessive screen time

Low physical activity

Increased anxiety

This project aims to build a classification model that predicts burnout risk and can help in early intervention.

🛠️ Technologies Used

Python

Pandas

NumPy

Matplotlib

Seaborn

Scikit-learn

📂 Dataset Information

The dataset (burnout_dataset.csv) contains the following features:

Feature	Description
study_hours	Number of hours studied per day
sleep_hours	Average sleep hours
screen_time	Daily screen time (hours)
exercise_minutes	Daily exercise time (minutes)
assignment_load	Academic load (scale 1–10)
anxiety_level	Anxiety level (scale 1–10)
burnout	Target variable (0 = Low, 1 = Medium, 2 = High)
⚙️ Project Workflow

1️⃣ Load dataset
2️⃣ Separate features and target
3️⃣ Train-test split (80% training, 20% testing)
4️⃣ Train Logistic Regression model
5️⃣ Train Random Forest model
6️⃣ Evaluate models using:

Accuracy Score

Confusion Matrix

Classification Report
7️⃣ Build manual input prediction system

📊 Model Evaluation

The models are evaluated using:

Accuracy Score

Confusion Matrix (visualized using heatmap)

Precision, Recall, F1-Score

Random Forest generally performs better due to handling non-linearity and feature importance.
