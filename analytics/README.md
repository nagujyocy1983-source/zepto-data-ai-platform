# Analytics Pipeline

This module implements a complete exploratory data analysis, preprocessing, machine learning, class-imbalance analysis, hyperparameter tuning, regression, and model-persistence pipeline using the Titanic dataset.

---

## 1. Module Objective

The objective of this module is to analyze the Titanic dataset and build a reproducible machine learning pipeline for predicting passenger survival.

The module covers:

- Dataset profiling
- Missing-value analysis and cleaning
- Univariate analysis
- Bivariate analysis
- Multivariate data-story visualizations
- Correlation analysis
- Standardization
- Stratified train/test splitting
- Numerical and categorical preprocessing
- Logistic Regression
- Decision Tree
- Random Forest
- Model evaluation
- ROC-AUC analysis
- Class-imbalance handling
- SMOTE oversampling
- Random Forest hyperparameter tuning
- Fare regression
- Residual analysis
- Final model comparison
- Complete pipeline persistence using Joblib
- Reloading and testing the saved pipeline on raw input

---

## 2. Dataset

The Titanic dataset contains:

- Records: **891**
- Columns: **15**
- Target variable: `survived`

The raw dataset is saved as:

```text
analytics/titanic.csv

```
## 3. Data Profiling and Missing-Value Analysis

The original Titanic dataset contains missing values in the following columns:

| Column | Missing Count | Missing Percentage | Action |
|---|---:|---:|---|
| `age` | 177 | 19.87% | Median imputation |
| `embarked` | 2 | 0.22% | Drop rows |
| `deck` | 688 | 77.22% | Drop column |
| `embark_town` | 2 | 0.22% | Drop rows |

### Missing-Value Strategy

The following percentage-based rule was applied:

- Missing rate below 5%: drop the affected rows.
- Missing rate between 5% and 30%: impute.
- Very high missingness: drop the column when reliable imputation would not be appropriate.

Therefore:

- `age` has 19.87% missing values, so the missing values were replaced using the median age.
- `embarked` has 0.22% missing values, so those rows were removed.
- `embark_town` has 0.22% missing values, so those rows were removed.
- `deck` has 77.22% missing values, so the column was removed because reliable imputation would not be appropriate.

After EDA cleaning:

```text
Rows: 889
Columns: 14
Missing values: 0

```
## 4. Univariate Analysis

### 4.1 Age

Age was analyzed using a histogram and box plot.

The median age used for imputation was:

```text
28.0

Using the IQR rule:

```text
Lower bound = Q1 - 1.5 × IQR
Upper bound = Q3 + 1.5 × IQR

```

The number of Age outliers was:

```text
65
```

The Age distribution contains passengers across a broad range, with most observations concentrated among younger and middle-aged passengers.

### 4.2 Fare

Fare was analyzed using a histogram and box plot.

The calculated Fare statistics were:

- Mean: **32.10**
- Median: **14.45**
- Mode: **8.05**
- IQR outliers: **114**

The ordering:

```text
Mean > Median > Mode
```

indicates that Fare is right-skewed. A relatively small number of passengers paid much higher fares, which increases the mean substantially above the median and mode.

---

## 5. Bivariate Analysis

### 5.1 Survival Rate by Sex

| Sex | Survival Rate |
|---|---:|
| Female | 74.20% |
| Male | 18.89% |

Female passengers had a substantially higher observed survival rate than male passengers in this dataset.

### 5.2 Survival Rate by Passenger Class

| Passenger Class | Survival Rate |
|---|---:|
| 1 | 62.96% |
| 2 | 47.28% |
| 3 | 24.24% |

The observed survival rate decreases from first class to third class.

### 5.3 Survival Rate by Sex and Passenger Class

| Sex | Pclass | Survival Rate |
|---|---:|---:|
| Female | 1 | 96.81% |
| Female | 2 | 92.11% |
| Female | 3 | 50.00% |
| Male | 1 | 36.89% |
| Male | 2 | 15.74% |
| Male | 3 | 13.54% |

Combining sex and passenger class reveals stronger group-level differences than either variable alone. Female first- and second-class passengers had the highest observed survival rates, while male second- and third-class passengers had substantially lower observed survival rates.

---

## 6. Correlation Analysis

The correlation matrix was calculated using exactly these six columns:

```text
survived
pclass
age
sibsp
parch
fare
```

The redundant variables `adult_male` and `alone` were excluded.

### Strongest Correlations

The two strongest absolute off-diagonal correlations were:

1. **Pclass and Fare:** approximately **-0.550**
2. **SibSp and Parch:** approximately **0.415**

The negative Pclass-Fare relationship reflects the relationship between passenger-class coding and fare levels. The positive SibSp-Parch relationship indicates that passengers traveling with spouses or siblings were also more likely to travel with parents or children.

The correlation heatmap is saved as:

```text
analytics/correlation_heatmap.png
```

---

## 7. Multivariate Data Story

### Chart 1 - Survival by Sex

The observed female survival rate was approximately 74.20%, compared with approximately 18.89% for males. This shows a substantial difference in observed survival across sex groups.

### Chart 2 - Survival by Passenger Class

First-class passengers had an observed survival rate of approximately 62.96%, compared with 47.28% for second class and 24.24% for third class. Passenger class therefore provides an important socioeconomic dimension to the survival analysis.

### Chart 3 - Sex and Passenger Class Combined

Combining sex and passenger class gives a more detailed survival pattern. Female first- and second-class passengers had survival rates above 90%, whereas male second- and third-class passengers had much lower observed survival rates.

### Chart 4 - Age, Fare and Survival Relationships

Age and Fare were examined together with survival-related variables. Fare is strongly right-skewed and negatively correlated with Pclass, while Age has a weaker direct relationship with survival. The combined analysis demonstrates the value of considering demographic and socioeconomic variables together.

---

## 8. Standardization Check

Age and Fare were standardized using the z-score formula:

```text
z = (x - mean) / standard deviation
```

After transformation, the variables had approximately:

```text
Mean = 0
Standard Deviation = 1
```

This exploratory standardization check is separate from the modeling pipeline preprocessing.

The modeling pipeline performs its own `StandardScaler` transformation, fitted only on the training data.

---

## 9. Class Balance

The target variable `survived` has the following distribution:

| Class | Count |
|---|---:|
| 0 - Not Survived | 549 |
| 1 - Survived | 342 |

The target classes are not perfectly balanced, so stratified splitting and class-imbalance experiments were performed.

---

## 10. Train/Test Split

An 80/20 stratified train/test split was used:

| Dataset | Records |
|---|---:|
| Training | 712 |
| Testing | 179 |

The split used:

```python
train_test_split(
    test_size=0.20,
    random_state=42,
    stratify=y
)
```

Stratification preserves approximately the same survived/not-survived class proportions in the training and testing datasets.

---

## 11. Modeling Preprocessing

### Numerical Features

```text
pclass
age
sibsp
parch
fare
```

Numerical preprocessing:

- Median imputation
- StandardScaler

### Categorical Features

```text
sex
embarked
```

Categorical preprocessing:

- Most-frequent imputation
- One-Hot Encoding
- `handle_unknown="ignore"`

A `ColumnTransformer` was used to apply the appropriate preprocessing to each feature type.

After preprocessing:

```text
Training shape: (712, 10)
Testing shape: (179, 10)
Missing values after preprocessing: 0
```

All preprocessing components used by the modeling pipeline were fitted only on the training data.

---

## 12. Classification Models

Three classifiers were trained using the same train/test split:

1. Logistic Regression
2. Decision Tree
3. Random Forest

---

## 13. Classification Model Evaluation

The following metrics were calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
- ROC-AUC

### Model Performance

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree | 0.7654 | 0.7547 | 0.5797 | 0.6557 | 0.7971 |
| Random Forest | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8287 |

Random Forest achieved the highest accuracy and F1 score among the baseline classifiers. Logistic Regression produced the highest ROC-AUC in this experiment.

### Confusion Matrices

#### Logistic Regression

```text
[[98 12]
 [23 46]]
```

#### Decision Tree

```text
[[97 13]
 [29 40]]
```

#### Random Forest

```text
[[98 12]
 [21 48]]
```

Generated files:

```text
analytics/logistic_regression_confusion_matrix.png
analytics/decision_tree_confusion_matrix.png
analytics/random_forest_confusion_matrix.png
```

---

## 14. ROC-AUC Analysis

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.8437 |
| Decision Tree | 0.7971 |
| Random Forest | 0.8287 |

The ROC curve comparison is saved as:

```text
analytics/roc_curve_comparison.png
```

---

## 15. Class-Imbalance Handling

Three Logistic Regression approaches were compared:

1. Baseline
2. `class_weight="balanced"`
3. SMOTE

| Method | Precision | Recall | F1 Score |
|---|---:|---:|---:|
| Baseline | 0.7931 | 0.6667 | 0.7244 |
| Class Weight Balanced | 0.7297 | 0.7826 | 0.7552 |
| SMOTE | 0.7397 | 0.7826 | 0.7606 |

SMOTE was applied only to the training fold.

Before SMOTE:

```text
Class 0: 439
Class 1: 273
```

After SMOTE:

```text
Class 0: 439
Class 1: 439
```

Both class weighting and SMOTE increased recall compared with the baseline. SMOTE produced the highest F1 score of 0.7606.

The comparison is saved as:

```text
analytics/class_imbalance_comparison.csv
```

---

## 16. Random Forest Hyperparameter Tuning

`GridSearchCV` was used to tune:

```text
n_estimators
max_depth
max_features
```

The search used 5-fold cross-validation with F1 score as the optimization metric.

The script reports the best parameter combination and OOB score at runtime.

The tuned Random Forest results are generated by:

```text
analytics/final_model_comparison.csv
```

---

## 17. Fare Regression

A multivariate Linear Regression model was developed to predict `fare`.

Regression features included:

```text
survived
pclass
sex
age
sibsp
parch
embarked
```

The regression model uses numerical imputation/scaling and categorical imputation/one-hot encoding.

### Regression Metrics

The script reports:

- MAE
- RMSE
- R²
- Adjusted R²

The residual plot is saved as:

```text
analytics/fare_regression_residual_plot.png
```

### Heteroscedasticity

The residual analysis is used to determine whether residual variance is approximately constant or shows a non-random spread across fitted values.

---

## 18. Final Model Comparison

Classification and regression metrics are kept as separate groups because they represent different machine learning tasks.

### Classification Metrics

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8045 | 0.7931 | 0.6667 | 0.7244 | 0.8437 |
| Decision Tree | 0.7654 | 0.7547 | 0.5797 | 0.6557 | 0.7971 |
| Random Forest | 0.8156 | 0.8000 | 0.6957 | 0.7442 | 0.8287 |

### Regression Metrics

| Model | MAE | RMSE | R² | Adjusted R² |
|---|---:|---:|---:|---:|
| Linear Regression - Fare | Reported by script | Reported by script | Reported by script | Reported by script |

The complete comparison table is saved as:

```text
analytics/final_model_comparison.csv
```

---

## 19. Final Model Selection and Recommendation

Random Forest produced the highest baseline F1 score at **0.7442** and the highest baseline accuracy at **0.8156**.

Logistic Regression produced the highest ROC-AUC at **0.8437**.

For the saved classification pipeline, the Random Forest model is used as the final classifier because it provides strong accuracy and F1 performance in the baseline comparison.

The final pipeline includes preprocessing together with the estimator.

---

## 20. Complete Pipeline Persistence

The complete classification pipeline combines preprocessing with the final estimator:

```text
Raw input
    ↓
ColumnTransformer
    ↓
Numerical imputation
    ↓
Numerical scaling
    ↓
Categorical imputation
    ↓
One-Hot Encoding
    ↓
Random Forest
    ↓
Prediction
```

The complete fitted pipeline was saved using Joblib:

```text
analytics/best_classification_pipeline.joblib
```

The saved object contains both preprocessing and the final estimator, allowing raw unprocessed input to be supplied directly to the pipeline.

---

## 21. Pipeline Reload Verification

The saved pipeline was successfully reloaded using `joblib.load()` and tested on raw input.

Example:

```text
pclass  sex   age  sibsp  parch  fare  embarked
3       male  24.0 2      0      24.15 S
```

The script successfully produced a prediction and survival probability after reloading the pipeline.

---

## 22. Generated Output Files

```text
analytics/titanic.csv
analytics/class_imbalance_comparison.csv
analytics/correlation_heatmap.png
analytics/roc_curve_comparison.png
analytics/logistic_regression_confusion_matrix.png
analytics/decision_tree_confusion_matrix.png
analytics/random_forest_confusion_matrix.png
analytics/fare_regression_residual_plot.png
analytics/final_model_comparison.csv
analytics/best_classification_pipeline.joblib
```

---

## 23. How to Run

From the project root:

```bash
python analytics/01_eda.py
```

Then:

```bash
python analytics/02_modeling.py
```

The modeling pipeline performs:

1. Dataset loading
2. Feature and target selection
3. Class-balance analysis
4. Stratified train/test split
5. Numerical and categorical preprocessing
6. Model training
7. Model evaluation
8. ROC-AUC analysis
9. Class-imbalance handling
10. SMOTE
11. Random Forest GridSearchCV
12. OOB evaluation
13. Fare regression
14. Residual analysis
15. Final model comparison
16. Pipeline persistence
17. Pipeline reload verification

---

## 24. Design Decisions

### Data Cleaning

Median imputation was used for Age because its missingness was between 5% and 30%. Rows with very small missing percentages were removed. Deck was removed because of its very high missingness.

### Feature Selection

Leakage-prone and redundant features were excluded from classification.

### Preprocessing

A `ColumnTransformer` was used to apply separate transformations to numerical and categorical features.

### Train/Test Separation

The dataset was stratified before modeling preprocessing, and preprocessing was fitted only on the training data.

### Class Imbalance

Baseline, class weighting, and SMOTE approaches were compared quantitatively.

### Hyperparameter Tuning

Random Forest was tuned using GridSearchCV with F1 as the scoring metric.

### Model Persistence

The complete preprocessing-plus-estimator pipeline was saved so that raw future records can be processed end-to-end.

---

## 25. Module Status

The Analytics Pipeline has been executed successfully without runtime errors.

Completed components include:

- EDA
- Data cleaning
- Missing-value analysis
- Outlier analysis
- Bivariate analysis
- Multivariate analysis
- Correlation analysis
- Standardization checks
- Stratified train/test splitting
- Preprocessing
- Logistic Regression
- Decision Tree
- Random Forest
- Classification evaluation
- ROC-AUC analysis
- Class-imbalance comparison
- SMOTE
- Random Forest GridSearchCV
- OOB evaluation
- Fare regression
- Residual analysis
- Final model comparison
- Complete pipeline persistence
- Pipeline reload verification

The main modeling script completed successfully without runtime errors.


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
