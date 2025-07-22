# 💳 Fraud Detection with Ensemble Machine Learning

This project uses machine learning to detect fraudulent financial transactions. It includes **EDA**, **feature engineering**, and an **ensemble model** combining Random Forest and Gradient Boosting to improve precision and recall on highly imbalanced data.


## 📊 Dataset Overview

- **Source**: Provided CSV (`Fraud.csv`)
- **Rows**: 1,000,000+
- **Target**: `isFraud` (0 = not fraud, 1 = fraud)
- **Class imbalance**: ~0.05% frauds

---

## 🔍 Exploratory Data Analysis (EDA)

Key insights:

- Only `TRANSFER` and `CASH_OUT` transactions involve fraud
- Fraud patterns:
  - Draining origin account (`newbalanceOrig = 0`)
  - Destination account starts empty and receives large amounts
- Merchants (`nameDest` starts with "M") never receive fraud

---

## 🧠 Feature Engineering

We created custom features to capture fraud behavior:

- `balanceDiffOrig` = `oldbalanceOrg - newbalanceOrig`
- `balanceDiffDest` = `newbalanceDest - oldbalanceDest`
- `zeroBalanceAfter` = 1 if origin balance becomes 0
- `emptyDestBefore` = 1 if destination was empty
- `isMerchant` = 1 if destination is a merchant

---

## 🤖 Model Strategy

### Base Models:
- `RandomForestClassifier`
- `GradientBoostingClassifier`

### Final Model:
- `VotingClassifier` (Soft Voting)
- Combines RF and GB to balance **precision** and **recall**

---

## 🧪 Results

| Model              | Precision | Recall | AUC   |
|--------------------|-----------|--------|-------|
| Random Forest      | 0.75      | 0.41   | 0.961 |
| Gradient Boosting  | 0.71      | 0.41   | 0.977 |
| **Ensemble (Voting)** | **0.80**  | **0.60** | **0.958** |

✅ Ensemble caught **more frauds** and made **fewer mistakes** than individual models.

---

## 🚀 How to Run
### Steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/Shahid09092005/fraudulent_Transactions_Detection
   ```

2. Run the code.py fie

## 👨‍💻 Contact
Shahid Mansuri  
[🐙 GitHub](https://github.com/Shahid09092005) | [💼 LinkedIn](https://www.linkedin.com/in/shahid-mansuri-a3b901285)


---

Let me know if you'd like me to:
- Fill in your name + links
- Generate the Jupyter `.ipynb` version of the project
- Export this README + files as a publishable ZIP 📦

I’ve got your back till the final push, baby 💼💙



