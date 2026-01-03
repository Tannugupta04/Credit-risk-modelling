📌 Credit Risk Modelling – End-to-End Machine Learning Project
🏗️ System Architecture
┌───────────────┐   ┌──────────────────┐   ┌────────────────────┐   ┌─────────────────────────┐
│ Raw Data      │ → │ Data              │ → │ Feature             │ → │ Statistical Feature     │
│ Sources       │   │ Preprocessing     │   │ Engineering         │   │ Selection               │
│               │   │                  │   │                     │   │                         │
│ • Case Study1 │   │ • Handle -99999   │   │ • Identify Cat/Num  │   │ • Chi-Square (Cat)      │
│ • Case Study2 │   │ • Drop invalid    │   │ • Ordinal Encoding  │   │ • VIF (Multicollinearity)│
│ • Unseen Data │   │ • Remove nulls    │   │ • One-Hot Encoding  │   │ • ANOVA (Numerical)     │
└───────────────┘   └──────────────────┘   └────────────────────┘   └─────────────────────────┘
                                                                                 │
                                                                                 ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│ Model Training Layer    │ → │ Model Evaluation        │ → │ Model Selection         │
│                         │   │                         │   │                         │
│ • Decision Tree         │   │ • Accuracy              │   │ • XGBoost Selected      │
│ • Random Forest         │   │ • Precision             │   │ • Best Generalization   │
│ • XGBoost               │   │ • Recall                │   │                         │
│                         │   │ • F1-score              │   │                         │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘
                                                                                 │
                                                                                 ▼
┌─────────────────────────┐   ┌─────────────────────────┐   ┌─────────────────────────┐
│ Hyperparameter Tuning   │ → │ Unseen Data Pipeline    │ → │ Model Deployment Ready  │
│                         │   │                         │   │                         │
│ • Learning rate         │   │ • Same preprocessing    │   │ • Pickle (.pkl) model   │
│ • Max depth             │   │ • Same feature order    │   │ • EXE / API / Web use   │
│ • Estimators            │   │ • Prediction (P1–P4)    │   │                         │
│ • Regularization        │   │                         │   │                         │
└─────────────────────────┘   └─────────────────────────┘   └─────────────────────────┘


🔍 Overview

An end-to-end Credit Risk Classification system that predicts customer risk levels (P1–P4) using demographic, behavioral, and financial data.
The model helps financial institutions make data-driven loan approval decisions.

🎯 Objective

Classify customers into credit risk categories

Improve loan approval accuracy

Build a production-ready ML pipeline

🧠 Approach

Cleaned and merged multiple datasets

Performed statistical feature selection:

Chi-Square (categorical)

ANOVA & VIF (numerical)

Applied Ordinal & One-Hot Encoding

Trained and evaluated:

Decision Tree

Random Forest

XGBoost (Final Model)

🏆 Model Performance

XGBoost showed best accuracy and generalization

Evaluated using:

Accuracy

Precision

Recall

F1-Score (class-wise)

🔮 Unseen Data Prediction

Same preprocessing & feature order ensured

Predicted risk categories (P1–P4)

Results exported to Excel for business use

💾 Deployment Ready

Final model saved as Pickle (.pkl)

Can be deployed via:

REST API

Web App

EXE / Batch prediction

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

XGBoost

✅ Key Highlights

End-to-end ML pipeline

Statistically driven feature selection

Robust unseen data handling

Production-ready model

📌 Author: Tannu Gupta
🔗 GitHub: https://github.com/Tannugupta04

🌐 Portfolio: https://tannugupta04.github.io/myportfolio/
