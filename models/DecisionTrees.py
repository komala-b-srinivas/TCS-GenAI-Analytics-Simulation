import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt

# === Paths ===
base_dir = "/Users/komalabelursrinivas/Documents/Forage/TCS"
data_path = os.path.join(base_dir, "output", "Cleaned_Delinquency_Dataset.csv")
output_dir = os.path.join(base_dir, "output")
plot_path = os.path.join(output_dir, "decision_tree_plot.png")

# === Ensure output directory exists ===
os.makedirs(output_dir, exist_ok=True)

# === Load data ===
df = pd.read_csv(data_path)

# === Feature selection ===
X = df[["Credit_Utilization", "Missed_Payments", "Credit_Score", "Debt_to_Income_Ratio", "Income"]]
y = df["Delinquent_Account"]

# === Split data ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# === Train model ===
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# === Predict & evaluate ===
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("AUC Score:", roc_auc_score(y_test, y_probs))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# === Save decision tree plot ===
plt.figure(figsize=(16, 10))
plot_tree(model, feature_names=X.columns, class_names=["Not Delinquent", "Delinquent"], filled=True)
plt.title("Decision Tree for Delinquency Prediction")
plt.tight_layout()
plt.savefig(plot_path)
plt.close()

print("✅ Plot saved at:", plot_path)

