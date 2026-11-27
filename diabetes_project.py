import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

# ---------------------------------------------------------------
# 1. Problem statement
# Predict whether a patient has diabetes for women of Pima heritage using
# health measurements (glucose, BMI, age, etc.), so that a doctor can decide
# who should be sent for a diabetes test first.
# ---------------------------------------------------------------

# ---------------------------------------------------------------
# 2. Data: load the file and check its shape, columns and missing values.
# ---------------------------------------------------------------
df = pd.read_csv("diabetes.csv")

print("Shape:", df.shape)
print("\nColumns:", list(df.columns))
print("\nLabel counts:")
print(df["Outcome"].value_counts())

# In these columns a 0 is impossible (nobody has 0 blood pressure), so a 0
# really means "missing". We turn those zeros into NaN so they show up as missing.
zero_cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[zero_cols] = df[zero_cols].replace(0, np.nan)

print("\nMissing values per column:")
print(df.isnull().sum())

# ---------------------------------------------------------------
# 3. EDA: charts to understand the label and the features.
# ---------------------------------------------------------------
# Chart 1: how many patients have diabetes and how many do not.
sns.countplot(x="Outcome", data=df)
plt.title("Patients with and without diabetes")
plt.show()

# Chart 2: how many values are missing in each column.
df.isnull().sum().plot(kind="bar")
plt.title("Missing values per column")
plt.show()

# Chart 3: glucose level for each group.
sns.boxplot(x="Outcome", y="Glucose", data=df)
plt.title("Glucose by Outcome")
plt.show()

# Chart 4: which columns are related to each other and to Outcome.
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.title("Correlation heatmap")
plt.show()

# ---------------------------------------------------------------
# 4. Split and baseline: 80/20 split, then a baseline that always predicts
# the most common class. Every real model must beat this score.
# ---------------------------------------------------------------
y = df["Outcome"]                  # the label
X = df.drop(columns="Outcome")     # the features

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

baseline = DummyClassifier(strategy="most_frequent")
baseline_acc = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="accuracy").mean()
print("\nBaseline accuracy (cross-validation):", round(baseline_acc, 3))

# ---------------------------------------------------------------
# 5. Pipeline: fill missing values and scale inside a Pipeline, so these steps
# are learned from the training part only and nothing leaks from the test set.
# ---------------------------------------------------------------
def make_pipeline(model):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])

models = {
    "Logistic Regression": make_pipeline(LogisticRegression(max_iter=1000)),
    "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=4, random_state=42)),
    "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=200, random_state=42)),
}

# ---------------------------------------------------------------
# 6. Model comparison: 5-fold cross-validation on the training set for each
# model, compared with the baseline. The best F1 decides the chosen model.
# ---------------------------------------------------------------
rows = [{"Model": "Baseline", "CV Accuracy": round(baseline_acc, 3), "CV F1": 0.0}]
for name, pipe in models.items():
    acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy").mean()
    f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1").mean()
    rows.append({"Model": name, "CV Accuracy": round(acc, 3), "CV F1": round(f1, 3)})

results = pd.DataFrame(rows)
print("\nModel comparison (cross-validation on train set):")
print(results)

best_name = results.iloc[1:].sort_values("CV F1", ascending=False).iloc[0]["Model"]
print("\nChosen model:", best_name)

# ---------------------------------------------------------------
# 7. Evaluation: fit the chosen model on the train set and score it once on
# the test set, using accuracy, confusion matrix, precision and recall.
# ---------------------------------------------------------------
best_model = models[best_name]
best_model.fit(X_train, y_train)
pred = best_model.predict(X_test)

print("\nTest accuracy :", round(accuracy_score(y_test, pred), 3))
print("Test precision:", round(precision_score(y_test, pred), 3))
print("Test recall   :", round(recall_score(y_test, pred), 3))
print("Test F1       :", round(f1_score(y_test, pred), 3))

cm = confusion_matrix(y_test, pred)
print("\nConfusion matrix (rows = real, columns = predicted):")
print(cm)

ConfusionMatrixDisplay(cm, display_labels=["No diabetes", "Diabetes"]).plot()
plt.title("Confusion matrix on the test set")
plt.show()

# ---------------------------------------------------------------
# 7b. What does the model get wrong? Compare the patients it missed
# (false negatives) and wrongly flagged (false positives) with the rest.
# ---------------------------------------------------------------
test = X_test.copy()
test["Real"] = y_test
test["Predicted"] = pred

missed = test[(test["Real"] == 1) & (test["Predicted"] == 0)]       # false negatives
caught = test[(test["Real"] == 1) & (test["Predicted"] == 1)]       # true positives
false_alarm = test[(test["Real"] == 0) & (test["Predicted"] == 1)]  # false positives

print("\nMissed diabetics:", len(missed), "| Caught diabetics:", len(caught),
      "| False alarms:", len(false_alarm))

compare = pd.DataFrame({
    "Missed diabetics": missed[["Glucose", "BMI", "Age", "Insulin"]].mean(),
    "Caught diabetics": caught[["Glucose", "BMI", "Age", "Insulin"]].mean(),
    "False alarms": false_alarm[["Glucose", "BMI", "Age", "Insulin"]].mean(),
}).round(1)
print("\nAverage values of each group:")
print(compare)