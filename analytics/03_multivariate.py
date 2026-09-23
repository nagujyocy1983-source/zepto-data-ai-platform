from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "titanic.csv"
README_FILE = BASE_DIR / "README.md"


# ============================================================
# LOAD AND CLEAN DATA
# ============================================================

print("\n" + "=" * 60)
print("MODULE 2 - MULTIVARIATE VISUALIZATIONS")
print("=" * 60)

df = pd.read_csv(DATA_FILE)

print(f"Original dataset shape: {df.shape}")

# Required EDA cleaning
df["age"] = df["age"].fillna(df["age"].median())

df = df.dropna(
    subset=["embarked", "embark_town"]
).copy()

if "deck" in df.columns:
    df = df.drop(columns=["deck"])

print(f"Cleaned dataset shape: {df.shape}")
print(f"Remaining missing values: {df.isna().sum().sum()}")


# ============================================================
# COMMON SETTINGS
# ============================================================

sns.set_theme(style="whitegrid")


# ============================================================
# 1. SURVIVAL BY SEX AND PCLASS
# ============================================================

print("\nCreating Chart 1...")

survival_by_sex_class = (
    df.groupby(["sex", "pclass"], as_index=False)["survived"]
    .mean()
)

plt.figure(figsize=(9, 6))

sns.barplot(
    data=survival_by_sex_class,
    x="pclass",
    y="survived",
    hue="sex",
    errorbar=None,
)

plt.title(
    "Survival Rate by Sex and Passenger Class"
)
plt.xlabel("Passenger Class")
plt.ylabel("Survival Rate")

plt.ylim(0, 1)

plt.tight_layout()

plt.savefig(
    BASE_DIR /
    "multivariate_01_survival_sex_pclass.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved: analytics/"
    "multivariate_01_survival_sex_pclass.png"
)


# ============================================================
# 2. AGE BY SEX AND SURVIVAL
# ============================================================

print("\nCreating Chart 2...")

plt.figure(figsize=(9, 6))

sns.violinplot(
    data=df,
    x="sex",
    y="age",
    hue="survived",
    split=True,
    inner="quart",
    common_norm=False,
)

plt.title(
    "Age Distribution by Sex and Survival Status"
)
plt.xlabel("Sex")
plt.ylabel("Age")

plt.tight_layout()

plt.savefig(
    BASE_DIR /
    "multivariate_02_age_sex_survival.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved: analytics/"
    "multivariate_02_age_sex_survival.png"
)


# ============================================================
# 3. FARE BY PCLASS AND SURVIVAL
# ============================================================

print("\nCreating Chart 3...")

plt.figure(figsize=(10, 6))

sns.boxplot(
    data=df,
    x="pclass",
    y="fare",
    hue="survived",
)

plt.title(
    "Fare Distribution by Passenger Class and Survival"
)
plt.xlabel("Passenger Class")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    BASE_DIR /
    "multivariate_03_fare_pclass_survival.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved: analytics/"
    "multivariate_03_fare_pclass_survival.png"
)


# ============================================================
# 4. AGE VS FARE BY SURVIVAL AND SEX
# ============================================================

print("\nCreating Chart 4...")

plt.figure(figsize=(10, 7))

sns.scatterplot(
    data=df,
    x="age",
    y="fare",
    hue="survived",
    style="sex",
    alpha=0.7,
    s=70,
)

plt.title(
    "Age vs Fare by Survival Status and Sex"
)
plt.xlabel("Age")
plt.ylabel("Fare")

plt.tight_layout()

plt.savefig(
    BASE_DIR /
    "multivariate_04_age_fare_survival_sex.png",
    dpi=300,
    bbox_inches="tight",
)

plt.close()

print(
    "Saved: analytics/"
    "multivariate_04_age_fare_survival_sex.png"
)


# ============================================================
# APPEND WRITTEN INTERPRETATIONS TO README
# ============================================================

interpretation_marker = (
    "## Multivariate Visualizations"
)

interpretation_section = r"""

---

## Multivariate Visualizations

The following four multivariate visualizations were created to examine interactions between multiple passenger attributes and survival outcomes.

### 1. Survival Rate by Sex and Passenger Class

`multivariate_01_survival_sex_pclass.png`

The chart shows survival differences across both sex and passenger class. Survival rates vary substantially by passenger class, with higher-class passengers generally showing higher survival rates. The combined view also shows a clear difference between male and female survival outcomes within the same passenger class.

### 2. Age Distribution by Sex and Survival Status

`multivariate_02_age_sex_survival.png`

This visualization examines age together with sex and survival status. It shows how the age distributions differ between survivors and non-survivors for male and female passengers. The distribution also highlights the presence of younger passengers among both survival groups.

### 3. Fare Distribution by Passenger Class and Survival

`multivariate_03_fare_pclass_survival.png`

This chart combines fare, passenger class, and survival. Fare distributions differ strongly across passenger classes, and survivors generally appear across a different fare distribution than non-survivors within several classes. This supports the relationship between passenger class, economic status represented by fare, and survival.

### 4. Age vs Fare by Survival Status and Sex

`multivariate_04_age_fare_survival_sex.png`

The scatter plot combines age, fare, survival status, and sex. It shows how survival observations are distributed across different age and fare ranges while also distinguishing passenger sex. Higher-fare observations are concentrated more heavily among first-class passengers, providing an additional view of the interaction between socioeconomic position and survival.

These visualizations provide multivariate evidence that survival was associated with combinations of passenger class, sex, age, and fare rather than a single variable in isolation.
"""

readme_text = README_FILE.read_text(
    encoding="utf-8"
)

if interpretation_marker not in readme_text:
    with README_FILE.open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(
            interpretation_section
        )

    print(
        "\nMultivariate interpretations "
        "appended to analytics/README.md"
    )
else:
    print(
        "\nMultivariate section already exists "
        "in analytics/README.md"
    )


# ============================================================
# FINAL VERIFICATION
# ============================================================

required_outputs = [
    "multivariate_01_survival_sex_pclass.png",
    "multivariate_02_age_sex_survival.png",
    "multivariate_03_fare_pclass_survival.png",
    "multivariate_04_age_fare_survival_sex.png",
]

missing_outputs = [
    name
    for name in required_outputs
    if not (BASE_DIR / name).exists()
]

if missing_outputs:
    raise RuntimeError(
        "Missing visualization files: "
        + ", ".join(missing_outputs)
    )

print("\n" + "=" * 60)
print("4 MULTIVARIATE VISUALIZATIONS COMPLETED")
print("=" * 60)

for name in required_outputs:
    print(f"- analytics/{name}")

print("\nREADME interpretations completed.")
print("=" * 60)