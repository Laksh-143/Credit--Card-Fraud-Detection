# 🛡️ Credit Card Fraud Detection System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Library](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)

## 📌 Project Overview
This project implements a comprehensive Machine Learning pipeline to detect fraudulent credit card transactions. The dataset features a **severe class imbalance** (0.6% fraud vs. 99.4% legit), mimicking real-world financial scenarios. 

The system moves beyond basic classification by engineering advanced **Geospatial** and **Behavioral** features (velocity, distance) and uses **BorderlineSMOTE** to handle imbalance, achieving a balanced 76% Precision and 73% Recall.

## 📂 Dataset
The dataset used for this project contains simulated credit card transaction data, including legit and fraudulent transactions.

- **Source:** [https://www.kaggle.com/datasets/kartik2112/fraud-detection]
- **Train Size:** ~1.3 Million rows
- **Test Size:** ~500k rows
- **Columns:**
   trans_date_trans_time – The exact timestamp of the purchase.
   cc_num – The unique Credit Card Number.
   merchant – The name of the store.
   category – The type of business.
   amt – The amount spent in dollar.
   first / last – The customer's First and Last Name.
   gender – The customer's gender.
   street / city / zip – The customer's home address text.
   state – The US State of the customer.
   lat / long – The exact GPS coordinates of the Customer's Home.
   city_pop – The population of the customer's city.
   job – The customer's profession.
   dob – The customer's Date of Birth.
   trans_num – A unique random ID for the transaction.
   unix_time – The time represented in seconds.
   merch_lat / merch_long – The exact GPS coordinates of the Merchant/Store.
   is_fraud – The Target Variable (0 = Legit, 1 = Fraud); the answer key your model uses to learn and be evaluated.

## ⚙️ Tech Stack
- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Seaborn, Matplotlib
- **Machine Learning:** XGBoost (Extreme Gradient Boosting)
- **Imbalance Handling:** Imbalanced-learn (BorderlineSMOTE)
- **Metrics:** Scikit-Learn (Precision-Recall Curve, Confusion Matrix)

## 🧠 Key Features Engineered
Instead of relying on raw data, I created stronger signals for the model:
1.  **Haversine Distance:** Calculated the physical distance between the cardholder's home and the merchant. (Sudden long-distance transactions are suspicious).
2.  **Transaction Velocity:** Calculated `daily_trans_count` to detect "spending sprees" (rapid-fire transactions in a short window).
3.  **Log-Transformed Amount:** Applied `Log(1+x)` to the transaction amount to normalize the skewed distribution of money.
4.  **Demographics:** Calculated User Age from DOB to capture generational spending patterns.
5.  **Frequency Encoding:** Mapped high-cardinality categorical variables (Merchant, Job, State) to their frequency distributions to prevent dimensionality explosion.

## 🚀 Methodology

### 1. Preprocessing & Leakage Prevention
Strict separation of Train/Test sets **before** encoding or scaling to ensure no data leakage. 
- **Frequency Encoding** was learned on Train and mapped to Test.
- **StandardScaling** was applied to normalize Distance, Age, and Amount.

### 2. Handling Imbalance
Used **BorderlineSMOTE** (Synthetic Minority Over-sampling Technique) to generate synthetic fraud examples. 
* *Why Borderline?* Unlike standard SMOTE, it focuses only on the "hard-to-classify" examples near the decision boundary, reducing noise.

### 3. Model Training
Trained an **XGBoost Classifier** with `tree_method='hist'` for efficiency on large tabular data.

### 4. Threshold Optimization
Instead of the default 0.5 threshold, I utilized the **Precision-Recall Curve** to mathematically find the optimal decision threshold (e.g., 0.9745) that maximizes the F1-Score.

## 📊 Results
The final model achieves a strong balance suitable for a banking environment:

| Metric | Score | Interpretation |
| :--- | :--- | :--- |
| **Precision** | **76%** | Low False Positive rate (Customers aren't annoyed by false alarms). |
| **Recall** | **73%** | High Detection rate (Captures the majority of fraud cases). |
| **AUC-ROC** | **0.98** | Excellent separation between Fraud and Legit classes. |

## 🛠️ How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/fraud-detection.git](https://github.com/yourusername/fraud-detection.git)

