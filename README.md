# 📌 Credit Risk Modelling – Machine Learning Project

## 🔍 Overview
An end-to-end **Credit Risk Classification system** that predicts customer risk levels (**P1–P4**) using demographic, behavioral, and financial data.  
The model helps financial institutions make **data-driven loan approval decisions**.

---

## 🎯 Objective
- Classify customers into credit risk categories  
- Improve loan approval accuracy  
- Build a **production-ready ML pipeline**

---

## 🧠 Approach
- Cleaned and merged multiple datasets  
- Performed **statistical feature selection**:
  - Chi-Square Test (Categorical features)
  - ANOVA & VIF (Numerical features)
- Applied **Ordinal Encoding** and **One-Hot Encoding**
- Trained and evaluated:
  - Decision Tree
  - Random Forest
  - **XGBoost (Final Model)**

---

## 🏆 Model Performance
- **XGBoost** achieved the best accuracy and generalization
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-Score (class-wise)

---

## 🔮 Unseen Data Prediction
- Ensured same preprocessing and feature order
- Predicted customer risk categories (**P1–P4**)
- Exported predictions to **Excel** for business use

---

## 💾 Deployment Ready
- Final trained model saved as **Pickle (.pkl)**
- Can be deployed using:
  - REST API
  - Web Application
  - EXE / Batch Prediction
## complete process

┌────────────┐ → ┌──────────────┐ → ┌──────────────────┐ → ┌─────────────────────┐
│ Raw Data   │   │ Preprocess   │   │ Feature Engg.    │   │ Feature Selection   │
│ CS1, CS2,  │   │ -99999/nulls │   │ Encoding         │   │ Chi², ANOVA, VIF   │
│ Unseen     │   │ Merge data   │   │ OHE, Ordinal     │   │                     │
└────────────┘   └──────────────┘   └──────────────────┘   └─────────────────────┘
                                                              ↓
┌──────────────────┐ → ┌──────────────────┐ → ┌──────────────────┐
│ Model Training   │   │ Model Evaluation │   │ Model Selection  │
│ DT, RF, XGB     │   │ Acc, Prec, Rec   │   │ XGBoost Final   │
└──────────────────┘   └──────────────────┘   └──────────────────┘
                                                              ↓
┌──────────────────┐ → ┌──────────────────┐ → ┌──────────────────┐
│ Hyperparameter   │   │ Unseen Prediction│   │ Deployment Ready │
│ Tuning (XGB)     │   │ Risk (P1–P4)     │   │ Pickle / API /  │
│                  │   │                  │   │ Web / EXE       │
└──────────────────┘   └──────────────────┘   └──────────────────┘



---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn  
- XGBoost  

---

## ✅ Key Highlights
- End-to-end machine learning pipeline
- Statistically driven feature selection
- Robust unseen data handling
- Production-ready deployment

---

## 👤 Author
**Tannu Gupta**  
🔗 GitHub: https://github.com/Tannugupta04  
🌐 Portfolio: https://tannugupta04.github.io/myportfolio/
