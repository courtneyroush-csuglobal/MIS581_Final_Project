# MIS581 CAPSTONE PROJECT
# Participant Retention Analysis


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import f_oneway

# LOAD DATA

print("Loading datasets...")

drop_withdrawals = pd.read_csv(
    "drop_withdrawals.txt",
    sep="|",
    low_memory=False
)

studies = pd.read_csv(
    "studies.txt",
    sep="|",
    low_memory=False
)

print("\nDrop Withdrawals Dataset Shape:")
print(drop_withdrawals.shape)

print("\nStudies Dataset Shape:")
print(studies.shape)

# REVIEW COLUMN NAMES

print("\nDrop Withdrawals Columns:")
print(drop_withdrawals.columns.tolist())

# BASIC MISSING VALUE REVIEW

print("\nMissing Values Summary")

missing = drop_withdrawals.isnull().sum()

missing_percent = (
                          drop_withdrawals.isnull().sum()
                          / len(drop_withdrawals)
                  ) * 100

missing_df = pd.DataFrame({
    "Missing Count": missing,
    "Percent Missing": missing_percent
})

print(
    missing_df
    .sort_values(
        "Percent Missing",
        ascending=False
    )
)

# DESCRIPTIVE STATISTICS
print("\nDescriptive Statistics")

print(
    drop_withdrawals.describe(
        include="all"
    )
)

# WITHDRAWAL REASONS
try:

    print("\nTop Withdrawal Reasons")

    withdrawal_reasons = (
        drop_withdrawals["reason"]
        .value_counts()
    )

    print(
        withdrawal_reasons.head(20)
    )

except:

    print(
        "\nReason column not found."
    )

# VISUALIZATION

try:

    plt.figure(
        figsize=(10, 6)
    )

    withdrawal_reasons.head(10).plot(
        kind="bar"
    )

    plt.title(
        "Top Participant Withdrawal Reasons"
    )

    plt.xlabel(
        "Withdrawal Reason"
    )

    plt.ylabel(
        "Frequency"
    )

    plt.tight_layout()

    plt.savefig(
        "TopWithdrawalReasons.png"
    )

    plt.show()

except:

    print(
        "Could not create withdrawal reason plot."
    )

# CREATE STUDY-LEVEL WITHDRAWAL TOTALS

try:

    withdrawals_per_study = (

        drop_withdrawals

        .groupby("nct_id")["count"]

        .sum()

        .reset_index()

    )

except:

    print(
        "\nUnable to find 'count' column."
    )

    print(
        "Review column names above."
    )

# RENAME VARIABLE

withdrawals_per_study.rename(

    columns={
        "count":
            "total_withdrawals"
    },

    inplace=True

)

print(
    "\nTotal Withdrawals Per Study"
)

print(
    withdrawals_per_study.head()
)

# MERGE WITH STUDIES TABLE

retention_df = studies.merge(

    withdrawals_per_study,

    on="nct_id",

    how="left"

)

# Missing withdrawals = 0

retention_df[
    "total_withdrawals"
] = retention_df[
    "total_withdrawals"
].fillna(0)

# REMOVE STUDIES WITH NO ENROLLMENT

retention_df = retention_df[
    retention_df["enrollment"] > 0
    ]

# CALCULATE RETENTION RATE

retention_df[
    "retention_rate"
] = (

            retention_df["enrollment"]

            -

            retention_df["total_withdrawals"]

    ) / retention_df["enrollment"]

# Ensure values stay valid

retention_df[
    "retention_rate"
] = retention_df[
    "retention_rate"
].clip(
    lower=0,
    upper=1
)

# RETENTION STATISTICS

print(
    "\nRetention Rate Statistics"
)

print(
    retention_df[
        "retention_rate"
    ].describe()
)

# RETENTION BY PHASE

phase_retention = (

    retention_df

    .groupby("phase")

    ["retention_rate"]

    .mean()

    .sort_values(
        ascending=False
    )

)

print(
    "\nAverage Retention By Phase"
)

print(
    phase_retention
)

# BAR CHART

plt.figure(
    figsize=(10, 6)
)

phase_retention.plot(
    kind="bar"
)

plt.title(
    "Average Participant Retention by Trial Phase"
)

plt.xlabel(
    "Trial Phase"
)

plt.ylabel(
    "Average Retention Rate"
)

plt.tight_layout()

plt.savefig(
    "RetentionByPhase.png"
)

plt.show()

# ANOVA TEST

groups = []

for phase in retention_df[
    "phase"
].dropna().unique():
    phase_data = retention_df[
        retention_df["phase"] == phase
        ][
        "retention_rate"
    ]

    groups.append(
        phase_data
    )

f_stat, p_value = f_oneway(
    *groups
)

print(
    "\nANOVA Results"
)

print(
    "F Statistic:",
    f_stat
)

print(
    "P Value:",
    p_value
)

if p_value < 0.05:

    print(
        "Reject Null Hypothesis"
    )

else:

    print(
        "Fail to Reject Null Hypothesis"
    )

# EXPORT RESULTS

phase_retention.to_csv(
    "PhaseRetentionSummary.csv"
)

retention_df[
    [
        "nct_id",
        "enrollment",
        "total_withdrawals",
        "retention_rate"
    ]
].head(
    1000
).to_csv(
    "RetentionSample.csv",
    index=False
)

print(
    "\nRetention Analysis Complete"
)