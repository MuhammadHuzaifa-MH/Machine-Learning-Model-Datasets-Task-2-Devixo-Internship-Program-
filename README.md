# Task 02: Build and Compare Machine Learning Models (Customer Churn Prediction)

## 📌 Introduction
This repository contains the complete implementation of **Task 2** for real-world predictive modeling. The project focuses on building, evaluating, and comparing multiple machine learning algorithms to predict customer churn using a large-scale dataset (440,000+ records)[cite: 1, 2].

## 🎯 Objectives
* Perform comprehensive **Data Preprocessing** (missing values cleaning, duplicate removal, categorical encoding, feature scaling, and train-test splitting)[cite: 2].
* Train at least **three different machine learning classification algorithms** (Logistic Regression, Decision Tree, Random Forest)[cite: 2, 3].
* Evaluate models using robust metrics (**Accuracy, Precision, Recall, F1 Score**)[cite: 3].
* Create a **Model Comparison Matrix** highlighting training times, performance scores, advantages, and limitations[cite: 3].
* Apply **Hyperparameter Tuning** using `RandomizedSearchCV` as a bonus feature[cite: 3].

## 🛠️ Technologies Used
* **Python 3.x**[cite: 1]
* **Pandas & NumPy** (Data manipulation)[cite: 1]
* **Scikit-learn** (Machine Learning models & preprocessing)[cite: 1]
* **Matplotlib & Seaborn** (Data visualization)[cite: 1]

---

## 📂 Project Structure
```text
ML_Task_2/
│
├── churn.csv                     # Customer Churn Dataset
├── churn_task.py                 # Main Python ML Pipeline Script
└── README.md                     # Project Documentation
