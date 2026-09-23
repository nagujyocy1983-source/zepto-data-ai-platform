import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# STEP 1 - LOAD CLEANED DATA
# ============================================================

print("\n" + "=" * 60)
print("STEP 1 - LOAD CLEANED DATA")
print("=" * 60)


# Load the cleaned Titanic dataset created by 01_eda.py
df = pd.read_csv("analytics/titanic.csv")


print("\nDataset shape:")
print(df.shape)


print("\nDataset columns:")
print(df.columns.tolist())


print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# STEP 1A - DEFINE FEATURES AND TARGET
# ============================================================

print("\n" + "=" * 60)
print("STEP 1A - DEFINE FEATURES AND TARGET")
print("=" * 60)


# survived is the classification target
y = df["survived"]


# Select features for classification.
# We intentionally exclude:
# - alive: directly represents survival and would cause target leakage
# - adult_male: redundant derived flag
# - alone: redundant derived flag
# - who: derived from sex/age
# - class: duplicate representation of pclass
# - embark_town: duplicate representation of embarked
#
# Fare is retained for classification because it is a valid
# predictor of survival.

feature_columns = [
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "fare",
    "embarked"
]


X = df[feature_columns]


print("\nSelected features:")
print(feature_columns)


print("\nTarget:")
print("survived")


print("\nFeature shape:")
print(X.shape)


print("\nTarget shape:")
print(y.shape)


# ============================================================
# STEP 1B - CHECK CLASS BALANCE
# ============================================================

print("\n" + "=" * 60)
print("STEP 1B - CLASS BALANCE")
print("=" * 60)


class_counts = y.value_counts().sort_index()

print("\nClass counts:")
print(class_counts)


class_percentages = y.value_counts(
    normalize=True
).sort_index() * 100


print("\nClass percentages:")
print(class_percentages.round(2))


# ============================================================
# STEP 1C - STRATIFIED TRAIN/TEST SPLIT
# ============================================================

print("\n" + "=" * 60)
print("STEP 1C - STRATIFIED TRAIN/TEST SPLIT")
print("=" * 60)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining feature shape:")
print(X_train.shape)


print("\nTesting feature shape:")
print(X_test.shape)


print("\nTraining target distribution:")
print(y_train.value_counts(normalize=True).sort_index().round(3))


print("\nTesting target distribution:")
print(y_test.value_counts(normalize=True).sort_index().round(3))


print("\nWhy stratification?")
print(
    "Stratification preserves approximately the same survived/not-survived "
    "class proportion in both the training and testing datasets."
)


# ============================================================
# STEP 1 COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("STEP 1 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# STEP 2 - PREPROCESSING PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 - PREPROCESSING PIPELINE")
print("=" * 60)


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# ============================================================
# STEP 2A - IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

print("\nSTEP 2A - FEATURE TYPES")
print("-" * 60)


numeric_features = [
    "pclass",
    "age",
    "sibsp",
    "parch",
    "fare"
]


categorical_features = [
    "sex",
    "embarked"
]


print("\nNumerical features:")
print(numeric_features)


print("\nCategorical features:")
print(categorical_features)


# ============================================================
# STEP 2B - NUMERICAL PREPROCESSING
# ============================================================

print("\nSTEP 2B - NUMERICAL PREPROCESSING")
print("-" * 60)


numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


print(
    "Numerical preprocessing: "
    "Median Imputation + StandardScaler"
)


# ============================================================
# STEP 2C - CATEGORICAL PREPROCESSING
# ============================================================

print("\nSTEP 2C - CATEGORICAL PREPROCESSING")
print("-" * 60)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


print(
    "Categorical preprocessing: "
    "Most-Frequent Imputation + One-Hot Encoding"
)


# ============================================================
# STEP 2D - COMBINE PREPROCESSING
# ============================================================

print("\nSTEP 2D - COLUMN TRANSFORMER")
print("-" * 60)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


print("ColumnTransformer created successfully.")


# ============================================================
# STEP 2E - FIT ONLY ON TRAINING DATA
# ============================================================

print("\nSTEP 2E - FIT PREPROCESSOR")
print("-" * 60)


# IMPORTANT:
# The preprocessor is fitted ONLY on X_train.
# X_test is transformed later using the fitted preprocessor.

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


print("\nTraining data after preprocessing:")
print(X_train_processed.shape)


print("\nTesting data after preprocessing:")
print(X_test_processed.shape)


# ============================================================
# STEP 2F - VERIFY NO MISSING VALUES
# ============================================================

print("\nSTEP 2F - PREPROCESSING VALIDATION")
print("-" * 60)


print(
    "\nMissing values in processed training data:"
)
print(pd.DataFrame(X_train_processed).isna().sum().sum())


print(
    "\nMissing values in processed testing data:"
)
print(pd.DataFrame(X_test_processed).isna().sum().sum())


# ============================================================
# STEP 2G - GET TRANSFORMED FEATURE NAMES
# ============================================================

print("\nSTEP 2G - TRANSFORMED FEATURES")
print("-" * 60)


transformed_feature_names = (
    preprocessor.get_feature_names_out()
)


print("\nNumber of transformed features:")
print(len(transformed_feature_names))


print("\nTransformed feature names:")
print(transformed_feature_names)


# ============================================================
# STEP 2 COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("STEP 2 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# STEP 3 - TRAIN CLASSIFICATION MODELS
# ============================================================

print("\n" + "=" * 60)
print("STEP 3 - TRAIN CLASSIFICATION MODELS")
print("=" * 60)


# ------------------------------------------------------------
# STEP 3A - IMPORT CLASSIFIERS
# ------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# ------------------------------------------------------------
# STEP 3B - LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\nSTEP 3A - LOGISTIC REGRESSION")
print("-" * 60)

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_processed,
    y_train
)

logistic_predictions = logistic_model.predict(
    X_test_processed
)

print("Logistic Regression trained successfully.")


# ------------------------------------------------------------
# STEP 3C - DECISION TREE
# ------------------------------------------------------------

print("\nSTEP 3B - DECISION TREE")
print("-" * 60)

decision_tree_model = DecisionTreeClassifier(
    random_state=42,
    max_depth=5
)

decision_tree_model.fit(
    X_train_processed,
    y_train
)

decision_tree_predictions = decision_tree_model.predict(
    X_test_processed
)

print("Decision Tree trained successfully.")


# ------------------------------------------------------------
# STEP 3D - RANDOM FOREST
# ------------------------------------------------------------

print("\nSTEP 3C - RANDOM FOREST")
print("-" * 60)

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_model.fit(
    X_train_processed,
    y_train
)

random_forest_predictions = random_forest_model.predict(
    X_test_processed
)

print("Random Forest trained successfully.")


# ------------------------------------------------------------
# STEP 3E - VERIFY PREDICTIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("MODEL PREDICTION VERIFICATION")
print("=" * 60)

print("\nLogistic Regression predictions:")
print(logistic_predictions[:10])

print("\nDecision Tree predictions:")
print(decision_tree_predictions[:10])

print("\nRandom Forest predictions:")
print(random_forest_predictions[:10])


# ------------------------------------------------------------
# STEP 3 COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 3 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# STEP 4 - MODEL EVALUATION
# ============================================================

print("\n" + "=" * 60)
print("STEP 4 - MODEL EVALUATION")
print("=" * 60)


# ------------------------------------------------------------
# STEP 4A - IMPORT EVALUATION METRICS
# ------------------------------------------------------------

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns


# ------------------------------------------------------------
# STEP 4B - CREATE MODEL DICTIONARY
# ------------------------------------------------------------

models = {
    "Logistic Regression": (
        logistic_model,
        logistic_predictions
    ),
    "Decision Tree": (
        decision_tree_model,
        decision_tree_predictions
    ),
    "Random Forest": (
        random_forest_model,
        random_forest_predictions
    )
}


# ------------------------------------------------------------
# STEP 4C - CALCULATE EVALUATION METRICS
# ------------------------------------------------------------

results = []


for model_name, (model, predictions) in models.items():

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    results.append(
        {
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    )


# ------------------------------------------------------------
# STEP 4D - DISPLAY METRICS
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\nMODEL PERFORMANCE COMPARISON")
print("-" * 60)

print(
    results_df.round(4).to_string(index=False)
)


# ------------------------------------------------------------
# STEP 4E - CONFUSION MATRICES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CONFUSION MATRICES")
print("=" * 60)


for model_name, (model, predictions) in models.items():

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print(f"\n{model_name}")
    print("-" * 60)
    print(cm)


# ------------------------------------------------------------
# STEP 4F - CLASSIFICATION REPORTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASSIFICATION REPORTS")
print("=" * 60)


for model_name, (model, predictions) in models.items():

    print(f"\n{model_name}")
    print("-" * 60)

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


# ------------------------------------------------------------
# STEP 4G - ROC-AUC
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("ROC-AUC SCORES")
print("=" * 60)


roc_results = {}


for model_name, (model, predictions) in models.items():

    probabilities = model.predict_proba(
        X_test_processed
    )[:, 1]

    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    roc_results[model_name] = {
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_score
    }

    print(
        f"{model_name}: ROC-AUC = {auc_score:.4f}"
    )


# ------------------------------------------------------------
# STEP 4H - ROC CURVE
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))

for model_name, roc_data in roc_results.items():

    plt.plot(
        roc_data["fpr"],
        roc_data["tpr"],
        label=f"{model_name} (AUC = {roc_data['auc']:.3f})"
    )


plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")

plt.legend()
plt.tight_layout()

plt.savefig(
    "analytics/roc_curve_comparison.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# STEP 4I - CONFUSION MATRIX HEATMAPS
# ------------------------------------------------------------

for model_name, (model, predictions) in models.items():

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Not Survived", "Survived"],
        yticklabels=["Not Survived", "Survived"]
    )

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"{model_name} - Confusion Matrix")

    plt.tight_layout()

    filename = (
        model_name.lower()
        .replace(" ", "_")
        + "_confusion_matrix.png"
    )

    plt.savefig(
        f"analytics/{filename}",
        dpi=300
    )

    plt.show()


# ------------------------------------------------------------
# STEP 4 COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 4 COMPLETED SUCCESSFULLY")
print("=" * 60)

# ============================================================
# STEP 5 - CLASS IMBALANCE HANDLING
# ============================================================

print("\n" + "=" * 60)
print("STEP 5 - CLASS IMBALANCE HANDLING")
print("=" * 60)


# ------------------------------------------------------------
# STEP 5A - IMPORT IMBALANCE TOOLS
# ------------------------------------------------------------

from imblearn.over_sampling import SMOTE


# ------------------------------------------------------------
# STEP 5B - BASELINE LOGISTIC REGRESSION
# ------------------------------------------------------------

print("\nSTEP 5A - BASELINE LOGISTIC REGRESSION")
print("-" * 60)

baseline_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

baseline_model.fit(
    X_train_processed,
    y_train
)

baseline_predictions = baseline_model.predict(
    X_test_processed
)

baseline_precision = precision_score(
    y_test,
    baseline_predictions
)

baseline_recall = recall_score(
    y_test,
    baseline_predictions
)

baseline_f1 = f1_score(
    y_test,
    baseline_predictions
)

print("Baseline Logistic Regression:")
print(f"Precision: {baseline_precision:.4f}")
print(f"Recall:    {baseline_recall:.4f}")
print(f"F1 Score:  {baseline_f1:.4f}")


# ------------------------------------------------------------
# STEP 5C - CLASS WEIGHT BALANCED
# ------------------------------------------------------------

print("\nSTEP 5B - CLASS WEIGHT='BALANCED'")
print("-" * 60)

balanced_model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight="balanced"
)

balanced_model.fit(
    X_train_processed,
    y_train
)

balanced_predictions = balanced_model.predict(
    X_test_processed
)

balanced_precision = precision_score(
    y_test,
    balanced_predictions
)

balanced_recall = recall_score(
    y_test,
    balanced_predictions
)

balanced_f1 = f1_score(
    y_test,
    balanced_predictions
)

print("Balanced Logistic Regression:")
print(f"Precision: {balanced_precision:.4f}")
print(f"Recall:    {balanced_recall:.4f}")
print(f"F1 Score:  {balanced_f1:.4f}")


# ------------------------------------------------------------
# STEP 5D - SMOTE
# ------------------------------------------------------------

print("\nSTEP 5C - SMOTE")
print("-" * 60)

smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_processed,
    y_train
)

print("Training class distribution before SMOTE:")
print(y_train.value_counts().sort_index())

print("\nTraining class distribution after SMOTE:")
print(y_train_smote.value_counts().sort_index())


# ------------------------------------------------------------
# STEP 5E - TRAIN LOGISTIC REGRESSION WITH SMOTE DATA
# ------------------------------------------------------------

smote_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

smote_model.fit(
    X_train_smote,
    y_train_smote
)

smote_predictions = smote_model.predict(
    X_test_processed
)

smote_precision = precision_score(
    y_test,
    smote_predictions
)

smote_recall = recall_score(
    y_test,
    smote_predictions
)

smote_f1 = f1_score(
    y_test,
    smote_predictions
)

print("\nSMOTE Logistic Regression:")
print(f"Precision: {smote_precision:.4f}")
print(f"Recall:    {smote_recall:.4f}")
print(f"F1 Score:  {smote_f1:.4f}")


# ------------------------------------------------------------
# STEP 5F - COMPARE ALL THREE APPROACHES
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLASS IMBALANCE COMPARISON")
print("=" * 60)

imbalance_results = pd.DataFrame(
    {
        "Method": [
            "Baseline",
            "Class Weight Balanced",
            "SMOTE"
        ],
        "Precision": [
            baseline_precision,
            balanced_precision,
            smote_precision
        ],
        "Recall": [
            baseline_recall,
            balanced_recall,
            smote_recall
        ],
        "F1 Score": [
            baseline_f1,
            balanced_f1,
            smote_f1
        ]
    }
)

print(
    imbalance_results.round(4).to_string(index=False)
)


# ------------------------------------------------------------
# STEP 5G - SAVE COMPARISON TABLE
# ------------------------------------------------------------

imbalance_results.to_csv(
    "analytics/class_imbalance_comparison.csv",
    index=False
)

print(
    "\nSaved comparison table to: "
    "analytics/class_imbalance_comparison.csv"
)


# ------------------------------------------------------------
# STEP 5 COMPLETION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 60)
# ============================================================
# STEP 6 - RANDOM FOREST HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("STEP 6 - RANDOM FOREST HYPERPARAMETER TUNING")
print("=" * 60)

from sklearn.model_selection import GridSearchCV


# ------------------------------------------------------------
# STEP 6A - RANDOM FOREST WITH OOB ENABLED
# ------------------------------------------------------------

print("\nSTEP 6A - GRIDSEARCHCV")
print("-" * 60)

grid_rf = RandomForestClassifier(
    random_state=42,
    oob_score=True,
    n_jobs=-1
)


# ------------------------------------------------------------
# STEP 6B - DEFINE HYPERPARAMETER GRID
# ------------------------------------------------------------

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2"]
}

print("\nHyperparameter grid:")
print(param_grid)


# ------------------------------------------------------------
# STEP 6C - RUN GRIDSEARCHCV
# ------------------------------------------------------------

grid_search = GridSearchCV(
    estimator=grid_rf,
    param_grid=param_grid,
    scoring="f1",
    cv=5,
    n_jobs=-1,
    refit=True
)

grid_search.fit(
    X_train_processed,
    y_train
)


# ------------------------------------------------------------
# STEP 6D - BEST PARAMETERS
# ------------------------------------------------------------

best_rf_model = grid_search.best_estimator_

print("\nBest Random Forest parameters:")
print(grid_search.best_params_)

print("\nBest cross-validation F1 score:")
print(f"{grid_search.best_score_:.4f}")

print("\nBest Random Forest OOB score:")
print(f"{best_rf_model.oob_score_:.4f}")


# ------------------------------------------------------------
# STEP 6E - EVALUATE TUNED RANDOM FOREST
# ------------------------------------------------------------

best_rf_predictions = best_rf_model.predict(
    X_test_processed
)

best_rf_probabilities = best_rf_model.predict_proba(
    X_test_processed
)[:, 1]


best_rf_accuracy = accuracy_score(
    y_test,
    best_rf_predictions
)

best_rf_precision = precision_score(
    y_test,
    best_rf_predictions,
    zero_division=0
)

best_rf_recall = recall_score(
    y_test,
    best_rf_predictions,
    zero_division=0
)

best_rf_f1 = f1_score(
    y_test,
    best_rf_predictions,
    zero_division=0
)

best_rf_auc = roc_auc_score(
    y_test,
    best_rf_probabilities
)


print("\nTUNED RANDOM FOREST TEST PERFORMANCE")
print("-" * 60)

print(f"Accuracy:  {best_rf_accuracy:.4f}")
print(f"Precision: {best_rf_precision:.4f}")
print(f"Recall:    {best_rf_recall:.4f}")
print(f"F1 Score:  {best_rf_f1:.4f}")
print(f"ROC-AUC:   {best_rf_auc:.4f}")


print("\n" + "=" * 60)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# STEP 7 - FARE REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("STEP 7 - FARE REGRESSION")
print("=" * 60)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np


# ------------------------------------------------------------
# STEP 7A - DEFINE REGRESSION FEATURES AND TARGET
# ------------------------------------------------------------

print("\nSTEP 7A - REGRESSION DATA")
print("-" * 60)

regression_features = [
    "survived",
    "pclass",
    "sex",
    "age",
    "sibsp",
    "parch",
    "embarked"
]

regression_target = "fare"


X_reg = df[regression_features]
y_reg = df[regression_target]


print("\nRegression features:")
print(regression_features)

print("\nRegression target:")
print(regression_target)

print("\nRegression feature shape:")
print(X_reg.shape)

print("\nRegression target shape:")
print(y_reg.shape)


# ------------------------------------------------------------
# STEP 7B - TRAIN/TEST SPLIT FOR REGRESSION
# ------------------------------------------------------------

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg,
    y_reg,
    test_size=0.20,
    random_state=42
)


# ------------------------------------------------------------
# STEP 7C - REGRESSION PREPROCESSING
# ------------------------------------------------------------

regression_numeric_features = [
    "survived",
    "pclass",
    "age",
    "sibsp",
    "parch"
]

regression_categorical_features = [
    "sex",
    "embarked"
]


regression_numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


regression_categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ]
)


regression_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            regression_numeric_pipeline,
            regression_numeric_features
        ),
        (
            "categorical",
            regression_categorical_pipeline,
            regression_categorical_features
        )
    ]
)


# ------------------------------------------------------------
# STEP 7D - LINEAR REGRESSION PIPELINE
# ------------------------------------------------------------

regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            regression_preprocessor
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


regression_pipeline.fit(
    X_reg_train,
    y_reg_train
)


# ------------------------------------------------------------
# STEP 7E - REGRESSION PREDICTIONS
# ------------------------------------------------------------

y_reg_predictions = regression_pipeline.predict(
    X_reg_test
)


# ------------------------------------------------------------
# STEP 7F - REGRESSION METRICS
# ------------------------------------------------------------

regression_mae = mean_absolute_error(
    y_reg_test,
    y_reg_predictions
)

regression_rmse = np.sqrt(
    mean_squared_error(
        y_reg_test,
        y_reg_predictions
    )
)

regression_r2 = r2_score(
    y_reg_test,
    y_reg_predictions
)


n_regression = len(y_reg_test)

p_regression = (
    regression_pipeline
    .named_steps["preprocessor"]
    .transform(X_reg_test)
    .shape[1]
)

if n_regression - p_regression - 1 > 0:
    regression_adjusted_r2 = (
        1
        - (
            (1 - regression_r2)
            * (n_regression - 1)
            / (n_regression - p_regression - 1)
        )
    )
else:
    regression_adjusted_r2 = np.nan


print("\nREGRESSION PERFORMANCE")
print("-" * 60)

print(f"MAE:          {regression_mae:.4f}")
print(f"RMSE:         {regression_rmse:.4f}")
print(f"R²:           {regression_r2:.4f}")
print(f"Adjusted R²:  {regression_adjusted_r2:.4f}")


# ------------------------------------------------------------
# STEP 7G - RESIDUAL ANALYSIS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("RESIDUAL ANALYSIS")
print("=" * 60)


residuals = y_reg_test - y_reg_predictions


plt.figure(figsize=(8, 6))

plt.scatter(
    y_reg_predictions,
    residuals,
    alpha=0.7
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Fare Regression - Residual Plot")

plt.tight_layout()

plt.savefig(
    "analytics/fare_regression_residual_plot.png",
    dpi=300
)

plt.show()


# ------------------------------------------------------------
# STEP 7H - HETEROSCEDASTICITY INTERPRETATION
# ------------------------------------------------------------

residual_correlation = np.corrcoef(
    y_reg_predictions,
    np.abs(residuals)
)[0, 1]

print("\nHeteroscedasticity assessment:")

if abs(residual_correlation) >= 0.30:
    print(
        "The residual spread changes noticeably as predicted fare "
        "increases, suggesting possible heteroscedasticity."
    )
else:
    print(
        "The residual spread does not show a strong systematic change "
        "with predicted fare, so there is no strong evidence of "
        "heteroscedasticity from this residual check."
    )

print(
    "The residual plot should be reviewed together with this numerical "
    "check because heteroscedasticity is primarily a visual pattern "
    "in the residual spread."
)


print("\n" + "=" * 60)
print("STEP 7 COMPLETED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# STEP 8 - FINAL MODEL COMPARISON
# ============================================================

print("\n" + "=" * 60)
print("STEP 8 - FINAL MODEL COMPARISON")
print("=" * 60)


# ------------------------------------------------------------
# STEP 8A - ADD AUC TO CLASSIFICATION RESULTS
# ------------------------------------------------------------

final_classification_results = results_df.copy()

final_classification_results["AUC"] = (
    final_classification_results["Model"]
    .map(
        {
            model_name: roc_data["auc"]
            for model_name, roc_data in roc_results.items()
        }
    )
)


print("\nCLASSIFIER COMPARISON")
print("-" * 60)

print(
    final_classification_results[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "AUC"
        ]
    ].round(4).to_string(index=False)
)


# ------------------------------------------------------------
# STEP 8B - TUNED RANDOM FOREST ROW
# ------------------------------------------------------------

tuned_rf_row = pd.DataFrame(
    [
        {
            "Model": "Tuned Random Forest",
            "Accuracy": best_rf_accuracy,
            "Precision": best_rf_precision,
            "Recall": best_rf_recall,
            "F1 Score": best_rf_f1,
            "AUC": best_rf_auc
        }
    ]
)


final_classification_results = pd.concat(
    [
        final_classification_results,
        tuned_rf_row
    ],
    ignore_index=True
)


# ------------------------------------------------------------
# STEP 8C - FINAL CLASSIFICATION + REGRESSION TABLE
# ------------------------------------------------------------

final_comparison_table = pd.DataFrame(
    [
        {
            "Model Type": "Classification",
            "Model": "Logistic Regression",
            "Accuracy": results_df.loc[
                results_df["Model"] == "Logistic Regression",
                "Accuracy"
            ].iloc[0],
            "Precision": results_df.loc[
                results_df["Model"] == "Logistic Regression",
                "Precision"
            ].iloc[0],
            "Recall": results_df.loc[
                results_df["Model"] == "Logistic Regression",
                "Recall"
            ].iloc[0],
            "F1": results_df.loc[
                results_df["Model"] == "Logistic Regression",
                "F1 Score"
            ].iloc[0],
            "AUC": roc_results[
                "Logistic Regression"
            ]["auc"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Adjusted R2": np.nan
        },
        {
            "Model Type": "Classification",
            "Model": "Decision Tree",
            "Accuracy": results_df.loc[
                results_df["Model"] == "Decision Tree",
                "Accuracy"
            ].iloc[0],
            "Precision": results_df.loc[
                results_df["Model"] == "Decision Tree",
                "Precision"
            ].iloc[0],
            "Recall": results_df.loc[
                results_df["Model"] == "Decision Tree",
                "Recall"
            ].iloc[0],
            "F1": results_df.loc[
                results_df["Model"] == "Decision Tree",
                "F1 Score"
            ].iloc[0],
            "AUC": roc_results[
                "Decision Tree"
            ]["auc"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Adjusted R2": np.nan
        },
        {
            "Model Type": "Classification",
            "Model": "Random Forest",
            "Accuracy": results_df.loc[
                results_df["Model"] == "Random Forest",
                "Accuracy"
            ].iloc[0],
            "Precision": results_df.loc[
                results_df["Model"] == "Random Forest",
                "Precision"
            ].iloc[0],
            "Recall": results_df.loc[
                results_df["Model"] == "Random Forest",
                "Recall"
            ].iloc[0],
            "F1": results_df.loc[
                results_df["Model"] == "Random Forest",
                "F1 Score"
            ].iloc[0],
            "AUC": roc_results[
                "Random Forest"
            ]["auc"],
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Adjusted R2": np.nan
        },
        {
            "Model Type": "Classification",
            "Model": "Tuned Random Forest",
            "Accuracy": best_rf_accuracy,
            "Precision": best_rf_precision,
            "Recall": best_rf_recall,
            "F1": best_rf_f1,
            "AUC": best_rf_auc,
            "MAE": np.nan,
            "RMSE": np.nan,
            "R2": np.nan,
            "Adjusted R2": np.nan
        },
        {
            "Model Type": "Regression",
            "Model": "Linear Regression - Fare",
            "Accuracy": np.nan,
            "Precision": np.nan,
            "Recall": np.nan,
            "F1": np.nan,
            "AUC": np.nan,
            "MAE": regression_mae,
            "RMSE": regression_rmse,
            "R2": regression_r2,
            "Adjusted R2": regression_adjusted_r2
        }
    ]
)


print("\nFINAL MODEL COMPARISON TABLE")
print("-" * 60)

print(
    final_comparison_table.round(4).to_string(index=False)
)


# ------------------------------------------------------------
# STEP 8D - SAVE FINAL COMPARISON TABLE
# ------------------------------------------------------------

final_comparison_table.to_csv(
    "analytics/final_model_comparison.csv",
    index=False
)

print(
    "\nSaved final comparison table to: "
    "analytics/final_model_comparison.csv"
)


# ------------------------------------------------------------
# STEP 8E - FINAL CLASSIFIER RECOMMENDATION
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FINAL CLASSIFIER RECOMMENDATION")
print("=" * 60)


classifier_candidates = {
    "Logistic Regression": (
        results_df.loc[
            results_df["Model"] == "Logistic Regression",
            "F1 Score"
        ].iloc[0],
        roc_results["Logistic Regression"]["auc"]
    ),
    "Decision Tree": (
        results_df.loc[
            results_df["Model"] == "Decision Tree",
            "F1 Score"
        ].iloc[0],
        roc_results["Decision Tree"]["auc"]
    ),
    "Random Forest": (
        results_df.loc[
            results_df["Model"] == "Random Forest",
            "F1 Score"
        ].iloc[0],
        roc_results["Random Forest"]["auc"]
    ),
    "Tuned Random Forest": (
        best_rf_f1,
        best_rf_auc
    )
}


recommended_model_name = max(
    classifier_candidates,
    key=lambda model_name: (
        classifier_candidates[model_name][0],
        classifier_candidates[model_name][1]
    )
)


recommended_f1 = classifier_candidates[
    recommended_model_name
][0]

recommended_auc = classifier_candidates[
    recommended_model_name
][1]


print(
    f"\nBased on the test-set F1 score, the model selected for deployment "
    f"is {recommended_model_name}."
)

print(
    f"It achieved an F1 score of {recommended_f1:.4f} and a ROC-AUC "
    f"of {recommended_auc:.4f}."
)

print(
    "F1 is useful here because it balances precision and recall when "
    "evaluating survival classification."
)

print(
    "The final selection should also consider the observed recall, "
    "precision, and ROC-AUC values rather than accuracy alone."
)


print("\n" + "=" * 60)
print("STEP 8 COMPLETED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# STEP 9 - SAVE COMPLETE BEST PIPELINE
# ============================================================

print("\n" + "=" * 60)
print("STEP 9 - SAVE COMPLETE BEST PIPELINE")
print("=" * 60)


import joblib


# ------------------------------------------------------------
# STEP 9A - BUILD COMPLETE RAW-DATA PIPELINE
# ------------------------------------------------------------

final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            best_rf_model
        )
    ]
)


# ------------------------------------------------------------
# STEP 9B - FIT COMPLETE PIPELINE ON TRAINING DATA
# ------------------------------------------------------------

final_pipeline.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# STEP 9C - SAVE PIPELINE
# ------------------------------------------------------------

pipeline_file = (
    "analytics/best_classification_pipeline.joblib"
)

joblib.dump(
    final_pipeline,
    pipeline_file
)

print(
    f"\nComplete pipeline saved to: {pipeline_file}"
)


# ------------------------------------------------------------
# STEP 9D - RELOAD PIPELINE
# ------------------------------------------------------------

loaded_pipeline = joblib.load(
    pipeline_file
)

print("\nPipeline reloaded successfully.")


# ------------------------------------------------------------
# STEP 9E - TEST RELOADED PIPELINE WITH RAW INPUT
# ------------------------------------------------------------

raw_test_sample = X_test.iloc[[0]]

reloaded_prediction = loaded_pipeline.predict(
    raw_test_sample
)

reloaded_probability = loaded_pipeline.predict_proba(
    raw_test_sample
)[:, 1]


print("\nRaw input sample:")
print(raw_test_sample)

print("\nReloaded pipeline prediction:")
print(reloaded_prediction)

print("\nReloaded pipeline survival probability:")
print(
    np.round(
        reloaded_probability,
        4
    )
)


print("\n" + "=" * 60)
print("STEP 9 COMPLETED SUCCESSFULLY")
print("=" * 60)


# ============================================================
# FINAL MODULE 2 COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("MODULE 2 ANALYTICS PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nGenerated artifacts:")

print("- analytics/titanic.csv")
print("- analytics/class_imbalance_comparison.csv")
print("- analytics/roc_curve_comparison.png")
print("- analytics/logistic_regression_confusion_matrix.png")
print("- analytics/decision_tree_confusion_matrix.png")
print("- analytics/random_forest_confusion_matrix.png")
print("- analytics/fare_regression_residual_plot.png")
print("- analytics/final_model_comparison.csv")
print("- analytics/best_classification_pipeline.joblib")

print("\nAll required modeling stages have been executed.")

# ============================================================
# DECISION TREE VISUALIZATION
# ============================================================

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

print("\n" + "=" * 60)
print("DECISION TREE VISUALIZATION")
print("=" * 60)

plt.figure(figsize=(24, 14))

plot_tree(
    decision_tree_model,
    feature_names=transformed_feature_names,
    class_names=["Not Survived", "Survived"],
    filled=True,
    rounded=True,
    fontsize=8
)

plt.title("Decision Tree - Titanic Survival Prediction")
plt.tight_layout()

plt.savefig(
    "analytics/decision_tree_visualization.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("Decision tree visualization saved to:")
print("analytics/decision_tree_visualization.png")