# ===========================================================
# CUSTOMER PURCHASE PREDICTION (Decision Tree)
# Dataset: Ecommerce_Consumer_Behavior_Analysis_Data.csv
# Target: Derived Will_Purchase (0/1)
# ===========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------------------------------------
# 1. LOAD DATA
# -----------------------------------------------------------

df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")

print("\nDataset Loaded Successfully!\n")
print(df.head())
print("\nColumns:", list(df.columns))

# -----------------------------------------------------------
# 2. DROP ID COLUMN
# -----------------------------------------------------------

if "Customer_ID" in df.columns:
    df = df.drop("Customer_ID", axis=1)

# -----------------------------------------------------------
# 3. CREATE BINARY TARGET (0/1)
# -----------------------------------------------------------

df["Will_Purchase"] = df["Discount_Used"].replace({"True": 1, "False": 0})
df["Will_Purchase"] = df["Will_Purchase"].astype(int)

print("\nTarget column 'Will_Purchase' created.")

# -----------------------------------------------------------
# 4. CLEANING
# -----------------------------------------------------------

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object", "bool"]).columns.tolist()

df[num_cols] = df[num_cols].fillna(df[num_cols].median())

for c in cat_cols:
    if df[c].mode().empty:
        df[c] = df[c].fillna("Unknown")
    else:
        df[c] = df[c].fillna(df[c].mode()[0])

# -----------------------------------------------------------
# 5. FEATURE ENGINEERING
# -----------------------------------------------------------

if "Time_of_Purchase" in df.columns:
    df["Time_of_Purchase"] = pd.to_datetime(df["Time_of_Purchase"], errors="ignore")
    df["Purchase_Month"] = pd.to_datetime(df["Time_of_Purchase"], errors="coerce").dt.month.fillna(0).astype(int)
    df["Purchase_Day"] = pd.to_datetime(df["Time_of_Purchase"], errors="coerce").dt.day.fillna(0).astype(int)
    df = df.drop("Time_of_Purchase", axis=1)

df["Discount_Used"] = df["Discount_Used"].replace({"True":1, "False":0}).astype(int)
df["Customer_Loyalty_Program_Member"] = df["Customer_Loyalty_Program_Member"].replace({"True":1, "False":0}).astype(int)

# -----------------------------------------------------------
# 6. SPLIT FEATURES & TARGET
# -----------------------------------------------------------

X = df.drop("Will_Purchase", axis=1)
y = df["Will_Purchase"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

if len(cat_cols) > 0:
    ohe = OneHotEncoder(sparse_output=False, drop="first")
    ohe_arr = ohe.fit_transform(X[cat_cols])
    ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(cat_cols))

    X = X.drop(columns=cat_cols)
    X = pd.concat([X.reset_index(drop=True), ohe_df.reset_index(drop=True)], axis=1)

# -----------------------------------------------------------
# 7. TRAIN/TEST SPLIT
# -----------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# -----------------------------------------------------------
# 8. TRAIN DECISION TREE
# -----------------------------------------------------------

model = DecisionTreeClassifier(
    max_depth=6,
    min_samples_split=20,
    random_state=42
)

model.fit(X_train, y_train)
print("\nModel trained successfully.")

# -----------------------------------------------------------
# 9. EVALUATION
# -----------------------------------------------------------

y_pred = model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------------------------------------
# 10. FEATURE IMPORTANCE
# -----------------------------------------------------------

feat_imp = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop Features:\n", feat_imp.head(10))

sns.barplot(x="Importance", y="Feature", data=feat_imp.head(10))
plt.title("Top 10 Feature Importances")
plt.show()
