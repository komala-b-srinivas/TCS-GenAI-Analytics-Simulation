import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("../output/Cleaned_Delinquency_Dataset.csv")

# Features and target
X = df[["Credit_Utilization", "Missed_Payments", "Credit_Score", "Debt_to_Income_Ratio", "Income"]]
y = df["Delinquent_Account"]

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# Train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_resampled, y_resampled)

# Predict on test set
y_probs = model.predict_proba(X_test)[:, 1]
y_pred = (y_probs > 0.5).astype(int)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_probs))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Class balance
print("\nClass distribution:\n", y.value_counts())
