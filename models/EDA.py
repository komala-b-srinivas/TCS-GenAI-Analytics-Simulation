import pandas as pd
from sklearn.impute import KNNImputer

# Load the dataset
df = pd.read_csv("../data/Delinquency_prediction_dataset.csv")

# Step 1: Fill median for Credit_Score and Loan_Balance
df["Credit_Score"] = df["Credit_Score"].fillna(df["Credit_Score"].median())
df["Loan_Balance"] = df["Loan_Balance"].fillna(df["Loan_Balance"].median())

# Step 2: KNN Imputation for Income
# We’ll only include numeric columns relevant for imputing Income
knn_data = df[["Age", "Income", "Credit_Score", "Loan_Balance", "Debt_to_Income_Ratio"]]

# Initialize the imputer
imputer = KNNImputer(n_neighbors=5)

# Fit and transform
imputed_data = imputer.fit_transform(knn_data)

# Put imputed Income back into df
df["Income"] = imputed_data[:, knn_data.columns.get_loc("Income")]

# Step 3: Check if all missing values are handled
print("\n✅ Missing values after all imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# (Optional) Save the cleaned dataset
df.to_csv("Cleaned_Delinquency_Dataset.csv", index=False)
