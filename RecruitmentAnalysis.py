# MIS581 Capstone Project Python Code
# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Display all rows when needed
pd.set_option('display.max_rows', None)

# Import AACT Studies Table
studies = pd.read_csv(
    "studies.txt",
    sep="|",
    low_memory=False
)

print("\nDataset Shape")
print(studies.shape)

print("\nColumn Names")
print(studies.columns.tolist())

print("\nDataset Information")
print(studies.info())

# Data Preparation
# Explore Missing Values
missing_count = studies.isnull().sum()

missing_percent = (studies.isnull().sum() / len(studies)) * 100

missing_df = pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_percent})
print(missing_df.sort_values(by="Missing %", ascending=False))

# Export for appendix
missing_df.to_csv("missing_values_summary.csv")

# Descriptive Statistics

print("\nEnrollment Statistics")
print(studies["enrollment"].describe())

print("\nNumber of Arms Statistics")
print(studies["number_of_arms"].describe())

# Study Status Distribution
print("\nOverall Status Distribution")
status_dist = (
    studies["overall_status"].value_counts(normalize=True)*100
)

print(round(status_dist, 2))

#Study Type Distribution
print("\nStudy Type Distribution")
study_type_dist = (
    studies["study_type"].value_counts(normalize=True)*100
)

print(round(study_type_dist, 2))

# Trial Phase Distribution
print("\nPhase Distribution")
phase_dist = (
    studies["phase"].value_counts(normalize=True)*100
)

print(round(phase_dist, 2))
# Visualizations

# Enrollment Histogram
plt.figure(figsize=(8,5))

studies["enrollment"].dropna().hist(
    bins=30
)

plt.title("Distribution of Clinical Trial Enrollment")
plt.xlabel("Enrollment")
plt.ylabel("Frequency")

plt.show()

# Trial Status Plot

plt.figure(figsize=(8,5))

studies["overall_status"] \
    .value_counts() \
    .head(10) \
    .plot(kind="bar")

plt.title(
    "Clinical Trial Status Distribution"
)

plt.xlabel("Status")
plt.ylabel("Count")

plt.show()

# Trial Phase Plot
plt.figure(figsize=(8,5))

studies["phase"] \
    .value_counts() \
    .plot(kind="bar")

plt.title(
    "Clinical Trial Phase Distribution"
)

plt.xlabel("Phase")
plt.ylabel("Count")

plt.show()

# Create Success Variable
# completed studies = successful
studies["success"] = np.where(
    studies["overall_status"] == "COMPLETED",
    1,
    0
)

print("\nSuccess Variable Summary")
print(
    studies["success"]
    .value_counts()
)

# Hypothesis Test #1
# Phase vs success
# chi-square test
phase_table = pd.crosstab(
    studies["phase"],
    studies["success"]
)

chi2, p, dof, expected = chi2_contingency(
    phase_table
)

print("\nChi-Square Test Results")

print("Chi-Square Statistic:", chi2)
print("P-Value:", p)
print("Degrees of Freedom:", dof)

if p < 0.05:
    print(
        "Result: Reject Null Hypothesis"
    )
else:
    print(
        "Result: Fail to Reject Null Hypothesis"
    )

# Hypothesis Test #2
# Enrollment vs Success
# T - Test

completed = studies[
    studies["success"] == 1
]["enrollment"]

not_completed = studies[
    studies["success"] == 0
]["enrollment"]

t_stat, p_value = ttest_ind(
    completed.dropna(),
    not_completed.dropna(),
    equal_var=False
)

print("\nEnrollment T-Test")

print("T Statistic:", t_stat)
print("P-Value:", p_value)

if p_value < 0.05:
    print(
        "Result: Reject Null Hypothesis"
    )
else:
    print(
        "Result: Fail to Reject Null Hypothesis"
    )

# Logistic Regression
# Predict Study Success

analysis_df = studies[
    [
        "success",
        "enrollment",
        "number_of_arms"
    ]
].dropna()

X = analysis_df[
    [
        "enrollment",
        "number_of_arms"
    ]
]

y = analysis_df["success"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42
)

model = LogisticRegression(
    max_iter=1000
)

model.fit(
    X_train,
    y_train
)

predictions = model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nLogistic Regression Results")

print(
    "Model Accuracy:",
    round(accuracy,3)
)

coefficients = pd.DataFrame({
    "Variable": X.columns,
    "Coefficient": model.coef_[0]
})

print("\nRegression Coefficients")
print(coefficients)

coefficients.to_csv(
    "logistic_regression_coefficients.csv",
    index=False
)

# Export Summary Statistics
summary_stats = studies[
    [
        "enrollment",
        "number_of_arms"
    ]
].describe()

summary_stats.to_csv(
    "summary_statistics.csv"
)

print(
    "\nAnalysis Complete"
)
