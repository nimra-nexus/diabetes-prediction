# Diabetes Prediction Using Machine Learning

Module 3 ML project by Nimra.

## Problem

Predict whether a patient has diabetes, for women of Pima heritage, using health measurements (glucose, BMI, blood pressure, age and more), so that a doctor can decide who should be sent for a diabetes test first.

This is a supervised classification problem with two classes: diabetes (1) or no diabetes (0).

## Dataset

- **Source:** Pima Indians Diabetes Database (UCI Machine Learning), downloaded from Kaggle
- **Size:** 768 rows, 8 features and 1 label
- **Label:** `Outcome` (1 = diabetes, 0 = no diabetes). 500 patients are 0 and 268 are 1, so the classes are imbalanced.
- **Features:** Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
- **Cleaning:** In Glucose, BloodPressure, SkinThickness, Insulin and BMI a value of 0 is impossible, so zeros were treated as missing values.

## What the project does

1. Loads the data and checks its shape, columns and missing values
2. Makes four EDA charts (label count, missing values, glucose by outcome, correlation heatmap)
3. Splits the data 80/20 (`random_state=42`, stratified) and calculates a baseline that always predicts the most common class
4. Builds a Pipeline (median imputation, standard scaling, model) so nothing leaks from the test set
5. Compares Logistic Regression, Decision Tree and Random Forest with 5-fold cross-validation against the baseline
6. Scores the chosen model once on the test set (accuracy, precision, recall, F1, confusion matrix)
7. Looks at which patients the model gets wrong

## Results

5-fold cross-validation on the training set:

| Model | CV Accuracy | CV F1 |
|---|---|---|
| Baseline (always predicts no diabetes) | 0.651 | 0.000 |
| Logistic Regression | 0.788 | 0.655 |
| Decision Tree (max depth 4) | 0.735 | 0.589 |
| Random Forest (200 trees) | 0.774 | 0.653 |

**Chosen model: Logistic Regression**, because it has the best cross-validation F1 and accuracy. F1 was the deciding measure because the classes are imbalanced.

Test set (154 rows):

| Metric | Score |
|---|---|
| Accuracy | 0.708 |
| Precision | 0.600 |
| Recall | 0.500 |
| F1 | 0.545 |

Confusion matrix (rows = real, columns = predicted):

|  | Predicted: no diabetes | Predicted: diabetes |
|---|---|---|
| **Real: no diabetes** | 82 | 18 |
| **Real: diabetes** | 27 | 27 |

## What the model gets wrong

The model misses 27 of the 54 diabetic patients in the test set. The missed patients have an average glucose of 118, while the diabetic patients it catches average 162, so the missed patients look like non-diabetic patients on glucose, which is the model's strongest clue. Insulin is also missing for 374 of 768 patients, so the model has little real information on it.

## Limitations

- The test set has only 154 rows, so the scores could change with a different split.
- Recall is only 0.50, so the model should not replace a medical test.
- Many Insulin and SkinThickness values were missing and filled with the median.
- Next steps: hyperparameter tuning, other models and more data.

## How to run

1. Install the libraries:

   ```
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```

2. Keep `diabetes.csv` in the same folder as the code.
3. Run:

   ```
   python diabetes_project.py
   ```

## Files

- `diabetes_project.py`: the full project code
- `diabetes.csv`: the dataset
- `README.md`: this file