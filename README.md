# 🧠 Customer Purchase Prediction using Decision Tree Classification

## 📌 Project Overview

This project demonstrates a **complete end-to-end machine learning workflow** for a **binary classification problem** using a **Decision Tree Classifier**.

The goal of the model is to **predict whether a customer will make a purchase** based on:

* Demographic information
* Online behavior
* Past purchasing history
* Geographical region

The project uses **synthetic data** to simulate a real-world customer dataset and applies proper **data preprocessing, feature engineering, dimensionality reduction, model training, evaluation, and visualization** techniques.

---

## 🎯 Problem Statement

> **Can we predict whether a customer will purchase a product based on their age, income, website activity, previous purchases, and region?**

This is formulated as a **supervised binary classification problem**:

* `0` → Not Purchased
* `1` → Purchased

---

## 🗂️ Dataset Description

The dataset contains **1,000 synthetic customer records** with the following features:

| Feature                      | Description                  |
| ---------------------------- | ---------------------------- |
| Age                          | Customer age                 |
| Income                       | Annual income                |
| Time_spent_on_website        | Minutes spent on the website |
| Number_of_previous_purchases | Past purchases               |
| Region                       | Customer geographical region |
| Purchased                    | Target variable (0 or 1)     |

✔ Missing values were intentionally introduced to simulate real-world data.

---

## 🔧 Data Preprocessing & Feature Engineering

The following preprocessing steps were applied:

### 1️⃣ Data Cleaning

* Removed duplicate records
* Handled missing values using:

  * **Mean imputation** for numerical features
  * **Most frequent value** for categorical features

### 2️⃣ Feature Transformation

* **Log transformation** applied to Income
* **Standardization** using `StandardScaler`

### 3️⃣ Categorical Encoding

* Tokenized multi-word region names (e.g., *"North Central"*)
* Encoded regions using **MultiLabelBinarizer**

### 4️⃣ Dimensionality Reduction

* Applied **PCA** on encoded region features
* Reduced region features to a **single PCA component**

---

## 🤖 Model Used

* **Decision Tree Classifier**
* Key properties:

  * Non-linear
  * Interpretable
  * Handles mixed data types well
  * No need for feature scaling (done here for completeness)

---

## 🔁 Sampling Strategy

Instead of a fixed train-test split, the project uses:

* **Random sampling**
* **Multiple training rounds**
* Evaluates robustness across different samples

This simulates real-world scenarios where data subsets change.

---

## 📊 Model Evaluation

### Metrics Used:

* **Accuracy Score**
* **Confusion Matrix**

### Confusion Matrix Interpretation:

* True Positives (TP)
* True Negatives (TN)
* False Positives (FP)
* False Negatives (FN)

The confusion matrix provides deeper insight beyond accuracy.

---

## 🌳 Model Visualization

* The trained decision tree is visualized using `plot_tree`
* Helps understand:

  * Feature importance
  * Decision rules
  * How predictions are made

---

## 🧠 Skills Demonstrated

### 🔹 Machine Learning

* Supervised classification
* Decision Trees
* Model evaluation
* Confusion matrix analysis

### 🔹 Data Processing

* Missing value handling
* Feature scaling
* Encoding categorical variables
* Dimensionality reduction (PCA)

### 🔹 Python & Libraries

* Pandas & NumPy
* Scikit-learn
* Matplotlib
* NLTK (tokenization)

### 🔹 Software Engineering

* Clean, modular code
* Reproducibility using random seeds
* Well-documented pipeline

