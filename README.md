
# 📌 Credit Risk Modelling – Machine Learning Project


## 🔥 Business-Driven Risk Segmentation (Executive Summary)

Built an **ML-based Credit Risk Segmentation System** that classifies customers into **P1–P4 risk categories**, enabling **data-driven and flexible loan approval decisions** aligned with business goals.

### 🧭 Business Risk Appetite–Based Targeting
The solution is designed around **business risk appetite**, allowing stakeholders to dynamically choose target customer segments based on growth objectives and risk tolerance:

- **Low Risk Appetite** → Focus only on **P1**  
  *(Safest customers, minimal default risk)*

- **Moderate Risk Appetite** → Approve **P1–P3**  
  *(Balanced approach between growth and risk)*

- **High Risk Appetite** → Expand to **P1–P4**  
  *(Aggressive growth strategy with higher revenue potential)*

### 📊 Business Impact
- Translated complex **machine learning outputs into actionable business strategies**
- Enabled teams to **balance risk vs revenue** based on organizational and market goals
- Provided flexibility to adapt loan approval policies without retraining the model

### 🏗️ End-to-End Delivery
Delivered the system as a **complete production-ready application**, covering:

- Data preprocessing & statistical feature selection  
- **XGBoost-based multi-class risk classification model**  
- Robust unseen data prediction pipeline  
- **Flask-based deployment** for real-world usability

# Complete flow

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
<img width="1214" height="1107" alt="Screenshot 2026-01-03 220118" src="https://github.com/user-attachments/assets/16aaef1e-0db6-4d46-955f-7d4242f94d7c" />




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
