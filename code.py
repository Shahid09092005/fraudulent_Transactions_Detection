# -*- coding: utf-8 -*

!pip install pandas scikit-learn numpy

# import the requirement libraries

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils import shuffle

# load the data
df =pd.read_csv('/content/Fraud.csv')
print("Successfully loaded the data")

"""**Analyse the data**"""

df.info()

df.sample(8) # sample of the data

# checking the null values in the data
df.isnull().sum()

"""not much values are null, so no need to impute it so easily  we can remove the null values in the data."""

# checking how's the look mathematically
df.describe()

#  value count of the output column
df['isFraud'].value_counts(normalize=True)

"""so , we should focus that sample of the data is to much imbalance

below is the visualization of the target column given in the dataset
"""

import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x='isFraud', data=df)
plt.title('Fraud vs Non-Fraud Count')

print("Transaction Types Distribution:")
print(df['type'].value_counts())

sns.countplot(data=df, x='type', order=df['type'].value_counts().index, palette='Set2')
plt.title('Transaction Type Frequencies')
plt.xticks(rotation=45)
plt.show()

sns.countplot(x='type', hue='isFraud', data=df)
plt.title('Transaction Type vs Fraud')

"""from above , we can concude that only "transfer" and "Cash_out" have the fraud value true.
So , in trasaction only there two plays critical role and rest of them we will futher remove it

Numerical values of the fraud Distribution
"""

# types of Fraud Distribution
fraud_by_type = df.groupby('type')['isFraud'].value_counts().unstack().fillna(0)
print("Fraud by Transaction Type:")
print(fraud_by_type)

sns.boxplot(x='isFraud', y='amount', data=df)

"""Note : here no need to remove the outliers because they have fraud value and there is only only outlier that is not fraud

Checking the behaviour of the account's amount
"""

# Balance Checks
df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']

print("\n🔹 Mean balanceDiffOrig (Fraud vs Non-Fraud):")
print(df.groupby('isFraud')['balanceDiffOrig'].mean())

print("\n🔹 Mean balanceDiffDest (Fraud vs Non-Fraud):")
print(df.groupby('isFraud')['balanceDiffDest'].mean())

"""It conclude that : how the money flow differs in fraudulent vs. normal transactions"""

# Merchants Check
df['isMerchant'] = df['nameDest'].str.startswith("M").astype(int)
merchant_stats = df.groupby(['isFraud', 'isMerchant']).size().unstack().fillna(0)

print("\n🔹 Merchant Fraud Stats:")
print(merchant_stats)

merchant_stats.plot(kind='bar', stacked=True, color=['skyblue', 'orange'])
plt.title("Merchant Involvement in Fraud")
plt.ylabel("Transaction Count")
plt.xticks(ticks=[0,1], labels=["Not Fraud", "Fraud"])
plt.show()

"""Conclude that Merchant not involve in the fraud transaction

# After permoming EDA , starting building the model
"""

# Feature engineering
df['balanceDiffOrig'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['balanceDiffDest'] = df['newbalanceDest'] - df['oldbalanceDest']
df['zeroBalanceAfter'] = (df['newbalanceOrig'] == 0).astype(int)
df['emptyDestBefore'] = (df['oldbalanceDest'] == 0).astype(int)
df['isMerchant'] = df['nameDest'].fillna('').astype(str).str.startswith('M').astype(int)

# Filter fraud-related transaction types only
df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()

# 🔧 Convert 'type' to numeric
df['type'] = df['type'].map({'TRANSFER': 0, 'CASH_OUT': 1})

#  Drop rows with missing target
df = df[df['isFraud'].notna()]

# Drop irrelevant columns
df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1, inplace=True)

# Drop rows where 'isFraud' is NaN because only 1 value is null
df = df[df['isFraud'].notna()]

# Define features and target
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Shuffle dataset to avoid order bias
X, y = shuffle(X, y, random_state=42)

#  Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

"""# ML algo"""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_score, recall_score
import time

"""1. Checking result on the Random Forest"""

#-------------------- RANDOM FOREST --------------------
print("Training on Random Forest...")
start_rf = time.time()
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
rf_time = time.time() - start_rf

# Predict and evaluate
rf_pred = rf.predict(X_test)
rf_proba = rf.predict_proba(X_test)[:, 1]

print(f"Random Forest Results (trained in {int(rf_time)}s):")
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))
print("ROC AUC Score:", roc_auc_score(y_test, rf_proba))

"""Note: Random Forest had higher precision and trained faster.

2. GRADIENT BOOSTING
"""

# -------------------- GRADIENT BOOSTING --------------------
print("Training Gradient Boosting...")
start_gb = time.time()
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)
gb.fit(X_train, y_train)
gb_time = time.time() - start_gb

# Predict and evaluate
gb_pred = gb.predict(X_test)
gb_proba = gb.predict_proba(X_test)[:, 1]

print(f"Gradient Boosting Results (trained in {int(gb_time)}s):")
print("Classification Report:\n", classification_report(y_test, gb_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, gb_pred))
print("ROC AUC Score:", roc_auc_score(y_test, gb_proba))

"""Note : Gradient Boosting had higher AUC and better at catching edge cases.

# which model

As the result of both different model both had lower recall individually, meaning they were missing frauds.

So I used a soft voting ensemble to:

1.   Leverage Random Forest’s robustness
2.   Capture Gradient Boosting’s sensitivity
3.   Increase recall and precision together without sacrificing much

As a result, the ensemble caught more frauds and made fewer false alerts, which is critical in real fraud systems.
"""

# Define Base model
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, random_state=42)

# Combine with Soft Voting
ensemble = VotingClassifier(estimators=[
    ('rf', rf),
    ('gb', gb)
], voting='soft')

# Training Soft Voting Ensemble...
start = time.time()
ensemble.fit(X_train, y_train)
elapsed = time.time() - start
print(f"✅ Training completed in {int(elapsed)} seconds.\n")

# Evaluate Performance of the model
y_pred = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)[:, 1]

print("Ensemble Model Performance (Soft Voting):")

print("Classification Report:\n", classification_report(y_test, y_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("Precision (Fraud):", round(precision_score(y_test, y_pred), 2))

print("Recall (Fraud):", round(recall_score(y_test, y_pred), 2))

print("ROC AUC Score:", round(roc_auc_score(y_test, y_proba), 3))
