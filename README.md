# 🛒 Customer Purchase Prediction – Data Analytics Project  
Predicting whether a customer is likely to make a purchase based on their demographic, behavioral, and transactional characteristics using a Decision Tree Classifier.

This project was built as part of a Data Analytics learning workflow and is aligned with real-world analytics tasks similar to what a Data Analyst Intern would perform (SQL/EDA/Feature Engineering/Modeling/Insights).

---

## 📌 Project Objective  
The goal of this project is to predict whether a customer is likely to make a purchase.  
Since the dataset does not contain a direct binary `"Purchased"` column, we derive the target variable:

### **Target Column: `Will_Purchase`**
Defined as:
Will_Purchase = 1 if Discount_Used == True
Will_Purchase = 0 otherwise




This approach is practical because customers who use a discount tend to exhibit stronger purchase intent.

---

## 📂 Dataset Information  
**File:** `Ecommerce_Consumer_Behavior_Analysis_Data.csv`  
**Size:** ~1,000 rows × 28 columns  
**Features include:**

- Customer demographics  
- Income, marital status, education  
- Purchase category and amount  
- Discount usage  
- Loyalty program membership  
- Payment method  
- Shipping preference  
- Social media influence  
- Time spent researching products  
- Return rate  
- Buying intent labels  
- Time_to_Decision (days before purchase)  

The dataset contains a mix of numerical, boolean, categorical, and date fields.

---

## 🧠 Machine Learning Approach  
**Model Used:** Decision Tree Classifier  
**Why?**  
- Easy to interpret  
- Works well with categorical + numerical variables  
- Suitable for marketing/purchase prediction problems  

---

## 🧹 Data Preprocessing Steps  

1. **Dropped ID columns** (`Customer_ID`)  
2. **Derived binary target** (`Will_Purchase`)  
3. **Handled missing values**  
4. **Converted boolean-like text ("True"/"False") to integers**  
5. **Converted date columns to month/day features**  
6. **One-Hot Encoded categorical variables**  
7. **Performed train-test split (75%/25%)**

---

## 🛠️ Tech Stack  

- **Python 3.x**  
- **Pandas**  
- **NumPy**  
- **Matplotlib / Seaborn**  
- **Scikit-learn**  
- **VS Code** 
