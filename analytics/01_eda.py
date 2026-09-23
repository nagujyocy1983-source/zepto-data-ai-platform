import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# --------------------------------------------------
# 1. Load Titanic dataset ONCE
# --------------------------------------------------

print("=" * 60)
print("LOADING TITANIC DATASET")
print("=" * 60)

df = sns.load_dataset("titanic")


# --------------------------------------------------
# 2. Save offline fallback immediately
# --------------------------------------------------

output_file = "analytics/titanic.csv"
df.to_csv(output_file, index=False)

print(f"\nOffline fallback saved to: {output_file}")


# --------------------------------------------------
# 3. Dataset shape
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET SHAPE")
print("=" * 60)

print(df.shape)


# --------------------------------------------------
# 4. Dataset information
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET INFO")
print("=" * 60)

df.info()


# --------------------------------------------------
# 5. Descriptive statistics
# --------------------------------------------------

print("\n" + "=" * 60)
print("DESCRIPTIVE STATISTICS")
print("=" * 60)

print(df.describe(include="all"))


# --------------------------------------------------
# 6. Missing-value analysis
# --------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUE ANALYSIS")
print("=" * 60)

missing_counts = df.isnull().sum()

missing_percentages = (
    df.isnull().mean() * 100
).round(2)

missing_report = pd.DataFrame(
    {
        "missing_count": missing_counts,
        "missing_percentage": missing_percentages,
    }
)

missing_report = missing_report[
    missing_report["missing_count"] > 0
]

print(missing_report)


# --------------------------------------------------
# 7. Basic dataset summary
# --------------------------------------------------

print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(f"Rows: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


print("\n" + "=" * 60)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ------------------------------------------------------------
# 8. Missing-value handling
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING-VALUE HANDLING")
print("=" * 60)

# Calculate missing percentages before applying any strategy
missing_counts = df.isnull().sum()
missing_percentages = (df.isnull().mean() * 100).round(2)

print("\nMissing values before cleaning:")
for column in df.columns:
    if missing_counts[column] > 0:
        print(
            f"{column}: "
            f"{missing_counts[column]} missing "
            f"({missing_percentages[column]}%)"
        )


# ------------------------------------------------------------
# Strategy 1: age
# 19.87% missing -> 5% to 30% -> median imputation
# ------------------------------------------------------------

age_median = df["age"].median()
df["age"] = df["age"].fillna(age_median)

print("\nAGE:")
print(f"Missing percentage: {missing_percentages['age']}%")
print(f"Strategy: Median imputation")
print(f"Median used: {age_median:.2f}")


# ------------------------------------------------------------
# Strategy 2: embarked
# 0.22% missing -> below 5% -> drop rows
# ------------------------------------------------------------

print("\nEMBARKED:")
print(f"Missing percentage: {missing_percentages['embarked']}%")
print("Strategy: Drop rows with missing embarked values")

df = df.dropna(subset=["embarked"])


# ------------------------------------------------------------
# Strategy 3: deck
# 77.22% missing -> very high missingness -> drop column
# ------------------------------------------------------------

print("\nDECK:")
print(f"Missing percentage: {missing_percentages['deck']}%")
print("Strategy: Drop column")
print("Justification: 77.22% missing values make imputation unreliable.")

df = df.drop(columns=["deck"])


# ------------------------------------------------------------
# Strategy 4: embark_town
# 0.22% missing -> below 5% -> drop rows
# ------------------------------------------------------------

print("\nEMBARK_TOWN:")
print(f"Missing percentage: {missing_percentages['embark_town']}%")
print("Strategy: Drop rows with missing embark_town values")

df = df.dropna(subset=["embark_town"])


# ------------------------------------------------------------
# Final missing-value verification
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MISSING VALUES AFTER CLEANING")
print("=" * 60)

remaining_missing = df.isnull().sum()
remaining_missing = remaining_missing[remaining_missing > 0]

if remaining_missing.empty:
    print("No missing values remain.")
else:
    print(remaining_missing)


# ------------------------------------------------------------
# Cleaned dataset summary
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLEANED DATASET SUMMARY")
print("=" * 60)

print(f"Rows after cleaning: {df.shape[0]}")
print(f"Columns after cleaning: {df.shape[1]}")

print("\nRemaining columns:")
print(df.columns.tolist())

print("\n" + "=" * 60)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# STEP 3 - UNIVARIATE ANALYSIS
# ============================================================

import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Histograms - Age and Fare
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3 - UNIVARIATE ANALYSIS")
print("=" * 60)

print("\nCreating histograms for AGE and FARE...")


# Age histogram
plt.figure(figsize=(8, 5))
plt.hist(df["age"], bins=20, edgecolor="black")
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# Fare histogram
plt.figure(figsize=(8, 5))
plt.hist(df["fare"], bins=20, edgecolor="black")
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()


# ------------------------------------------------------------
# 2. Box plots - Age and Fare
# ------------------------------------------------------------

print("\nCreating box plots for AGE and FARE...")


# Age box plot
plt.figure(figsize=(8, 4))
plt.boxplot(df["age"])
plt.title("Age Box Plot")
plt.ylabel("Age")
plt.tight_layout()
plt.show()


# Fare box plot
plt.figure(figsize=(8, 4))
plt.boxplot(df["fare"])
plt.title("Fare Box Plot")
plt.ylabel("Fare")
plt.tight_layout()
plt.show()


print("\nHistograms and box plots created successfully.")

# ================================================================
# STEP 4 - OUTLIER ANALYSIS AND FARE DISTRIBUTION
# ================================================================

print("\n" + "=" * 60)
print("STEP 4 - OUTLIER ANALYSIS AND FARE DISTRIBUTION")
print("=" * 60)


# ------------------------------------------------
# 1. IQR OUTLIER ANALYSIS - AGE
# ------------------------------------------------

age_q1 = df["age"].quantile(0.25)
age_q3 = df["age"].quantile(0.75)
age_iqr = age_q3 - age_q1

age_lower_bound = age_q1 - 1.5 * age_iqr
age_upper_bound = age_q3 + 1.5 * age_iqr

age_outliers = df[
    (df["age"] < age_lower_bound) |
    (df["age"] > age_upper_bound)
]

print("\nAGE OUTLIER ANALYSIS")
print("-" * 60)
print(f"Q1: {age_q1:.2f}")
print(f"Q3: {age_q3:.2f}")
print(f"IQR: {age_iqr:.2f}")
print(f"Lower bound: {age_lower_bound:.2f}")
print(f"Upper bound: {age_upper_bound:.2f}")
print(f"Number of age outliers: {len(age_outliers)}")


# ------------------------------------------------
# 2. IQR OUTLIER ANALYSIS - FARE
# ------------------------------------------------

fare_q1 = df["fare"].quantile(0.25)
fare_q3 = df["fare"].quantile(0.75)
fare_iqr = fare_q3 - fare_q1

fare_lower_bound = fare_q1 - 1.5 * fare_iqr
fare_upper_bound = fare_q3 + 1.5 * fare_iqr

fare_outliers = df[
    (df["fare"] < fare_lower_bound) |
    (df["fare"] > fare_upper_bound)
]

print("\nFARE OUTLIER ANALYSIS")
print("-" * 60)
print(f"Q1: {fare_q1:.2f}")
print(f"Q3: {fare_q3:.2f}")
print(f"IQR: {fare_iqr:.2f}")
print(f"Lower bound: {fare_lower_bound:.2f}")
print(f"Upper bound: {fare_upper_bound:.2f}")
print(f"Number of fare outliers: {len(fare_outliers)}")


# ------------------------------------------------
# 3. FARE MEAN, MEDIAN AND MODE
# ------------------------------------------------

fare_mean = df["fare"].mean()
fare_median = df["fare"].median()
fare_mode = df["fare"].mode().iloc[0]

print("\nFARE STATISTICS")
print("-" * 60)
print(f"Mean: {fare_mean:.2f}")
print(f"Median: {fare_median:.2f}")
print(f"Mode: {fare_mode:.2f}")


# ------------------------------------------------
# 4. SKEWNESS INTERPRETATION
# ------------------------------------------------

if fare_mean > fare_median > fare_mode:
    fare_skewness = "right-skewed"
elif fare_mean < fare_median < fare_mode:
    fare_skewness = "left-skewed"
else:
    fare_skewness = "approximately symmetric"

print("\nFARE DISTRIBUTION INTERPRETATION")
print("-" * 60)
print(
    f"Fare distribution is {fare_skewness} "
    f"because Mean ({fare_mean:.2f}), "
    f"Median ({fare_median:.2f}), and "
    f"Mode ({fare_mode:.2f}) follow the observed ordering."
)


# ------------------------------------------------
# STEP 4 COMPLETION
# ------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# STEP 5 - BIVARIATE ANALYSIS
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 - BIVARIATE ANALYSIS")
print("=" * 60)


# ------------------------------------------------------------
# 1. SURVIVAL RATE BY SEX
# ------------------------------------------------------------

print("\nSURVIVAL RATE BY SEX")
print("-" * 60)

female_mask = df["sex"] == "female"
male_mask = df["sex"] == "male"

female_survival_rate = df.loc[female_mask, "survived"].mean() * 100
male_survival_rate = df.loc[male_mask, "survived"].mean() * 100

print(f"Female survival rate: {female_survival_rate:.2f}%")
print(f"Male survival rate: {male_survival_rate:.2f}%")


# ------------------------------------------------------------
# 2. SURVIVAL RATE BY PCLASS
# ------------------------------------------------------------

print("\nSURVIVAL RATE BY PCLASS")
print("-" * 60)

pclass_1_mask = df["pclass"] == 1
pclass_2_mask = df["pclass"] == 2
pclass_3_mask = df["pclass"] == 3

pclass_1_survival_rate = df.loc[pclass_1_mask, "survived"].mean() * 100
pclass_2_survival_rate = df.loc[pclass_2_mask, "survived"].mean() * 100
pclass_3_survival_rate = df.loc[pclass_3_mask, "survived"].mean() * 100

print(f"1st class survival rate: {pclass_1_survival_rate:.2f}%")
print(f"2nd class survival rate: {pclass_2_survival_rate:.2f}%")
print(f"3rd class survival rate: {pclass_3_survival_rate:.2f}%")


# ------------------------------------------------------------
# 3. SURVIVAL RATE BY SEX AND PCLASS
# ------------------------------------------------------------

print("\nSURVIVAL RATE BY SEX AND PCLASS")
print("-" * 60)

female_class1_mask = (df["sex"] == "female") & (df["pclass"] == 1)
female_class2_mask = (df["sex"] == "female") & (df["pclass"] == 2)
female_class3_mask = (df["sex"] == "female") & (df["pclass"] == 3)

male_class1_mask = (df["sex"] == "male") & (df["pclass"] == 1)
male_class2_mask = (df["sex"] == "male") & (df["pclass"] == 2)
male_class3_mask = (df["sex"] == "male") & (df["pclass"] == 3)

female_class1_rate = df.loc[female_class1_mask, "survived"].mean() * 100
female_class2_rate = df.loc[female_class2_mask, "survived"].mean() * 100
female_class3_rate = df.loc[female_class3_mask, "survived"].mean() * 100

male_class1_rate = df.loc[male_class1_mask, "survived"].mean() * 100
male_class2_rate = df.loc[male_class2_mask, "survived"].mean() * 100
male_class3_rate = df.loc[male_class3_mask, "survived"].mean() * 100

print(f"Female + 1st class: {female_class1_rate:.2f}%")
print(f"Female + 2nd class: {female_class2_rate:.2f}%")
print(f"Female + 3rd class: {female_class3_rate:.2f}%")

print(f"Male + 1st class: {male_class1_rate:.2f}%")
print(f"Male + 2nd class: {male_class2_rate:.2f}%")
print(f"Male + 3rd class: {male_class3_rate:.2f}%")


# ------------------------------------------------------------
# 4. CORRELATION MATRIX
# ------------------------------------------------------------

print("\nCORRELATION MATRIX")
print("-" * 60)

correlation_columns = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]

corr_matrix = df[correlation_columns].corr()

print(corr_matrix.round(3))


# ------------------------------------------------------------
# 5. CORRELATION HEATMAP
# ------------------------------------------------------------

print("\nCreating correlation heatmap...")

plt.figure(figsize=(9, 7))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True
)

plt.title("Titanic Correlation Heatmap")
plt.tight_layout()

plt.savefig(
    "analytics/correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Correlation heatmap created successfully.")


# ------------------------------------------------------------
# 6. FIND TWO STRONGEST CORRELATIONS
# ------------------------------------------------------------

print("\nTWO STRONGEST CORRELATIONS")
print("-" * 60)

# Use absolute correlation values
abs_corr = corr_matrix.abs().copy()

# Remove self-correlations from the diagonal
for i in range(len(abs_corr)):
    abs_corr.iloc[i, i] = np.nan

# Convert correlation matrix into pairs
correlation_pairs = abs_corr.stack().dropna()

# Remove duplicate pairs
correlation_pairs = correlation_pairs[
    correlation_pairs.index.map(
        lambda x: x[0] < x[1]
    )
]

# Sort by absolute correlation
top_two = correlation_pairs.sort_values(ascending=False).head(2)

for (feature1, feature2), value in top_two.items():

    actual_value = corr_matrix.loc[feature1, feature2]

    direction = "positive" if actual_value > 0 else "negative"

    print(
        f"{feature1} ↔ {feature2}: "
        f"{actual_value:.3f} ({direction})"
    )


# ------------------------------------------------------------
# STEP 5 COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# STEP 6 - STANDARDIZATION CHECK
# ============================================================

print("\n" + "=" * 60)
print("STEP 6 - STANDARDIZATION CHECK")
print("=" * 60)

from sklearn.preprocessing import StandardScaler


# ------------------------------------------------------------
# Select numerical columns for exploratory standardization
# ------------------------------------------------------------

standardization_columns = ["age", "fare"]

standardization_df = df[standardization_columns].copy()


# ------------------------------------------------------------
# BEFORE STANDARDIZATION
# ------------------------------------------------------------

print("\nBEFORE STANDARDIZATION")
print("-" * 60)

before_stats = pd.DataFrame({
    "Mean": standardization_df.mean(),
    "Std": standardization_df.std()
})

print(before_stats.round(4))


# ------------------------------------------------------------
# APPLY STANDARDIZATION
# ------------------------------------------------------------

scaler = StandardScaler()

standardized_values = scaler.fit_transform(
    standardization_df
)

standardized_df = pd.DataFrame(
    standardized_values,
    columns=standardization_columns,
    index=standardization_df.index
)


# ------------------------------------------------------------
# AFTER STANDARDIZATION
# ------------------------------------------------------------

print("\nAFTER STANDARDIZATION")
print("-" * 60)

after_stats = pd.DataFrame({
    "Mean": standardized_df.mean(),
    "Std": standardized_df.std()
})

print(after_stats.round(4))


# ------------------------------------------------------------
# DISPLAY FIRST 5 VALUES
# ------------------------------------------------------------

print("\nFIRST 5 VALUES - BEFORE vs AFTER")
print("-" * 60)

comparison_df = pd.DataFrame({
    "age_before": standardization_df["age"].head(),
    "age_after": standardized_df["age"].head(),
    "fare_before": standardization_df["fare"].head(),
    "fare_after": standardized_df["fare"].head()
})

print(comparison_df.round(4))


# ------------------------------------------------------------
# INTERPRETATION
# ------------------------------------------------------------

print("\nSTANDARDIZATION INTERPRETATION")
print("-" * 60)

print(
    "Standardization transforms age and fare into z-scores "
    "with approximately mean 0 and standard deviation 1."
)

print(
    "This standardization is performed only for exploratory "
    "analysis and will NOT be reused as the final modeling "
    "preprocessing."
)


# ------------------------------------------------------------
# STEP 6 COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 60)