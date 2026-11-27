# 🏡 House Price Prediction Using Machine Learning

A beginner-friendly Machine Learning project that predicts residential house prices using Python and Scikit-learn.

The project follows a structured Machine Learning workflow, covering dataset inspection, exploratory data analysis, data preprocessing, feature engineering, feature selection, model training, evaluation, and hyperparameter tuning.

---

## 📌 About the Project

This project uses the **Ames Housing Dataset**, which contains information about residential properties in Ames, Iowa, USA.

The goal is to understand how different property features — such as living area, overall quality, number of rooms, and garage size — relate to house prices and how Machine Learning models can use this information to make predictions.

### Dataset Overview

| Details | Information |
|---|---|
| Dataset | Ames Housing |
| Location | Ames, Iowa, USA |
| Total Records | 2,930 |
| Total Columns | 82 |
| Target Variable | `SalePrice` |

---

## 🛠️ Technologies Used

- **Python** – Main programming language
- **Pandas** – Data handling and analysis
- **NumPy** – Numerical operations
- **Scikit-learn** – Data preprocessing and Machine Learning
- **Matplotlib & Seaborn** – Data visualization
- **Streamlit** – Planned interactive web application

---

## 📂 Project Structure

```text
House-Price-Prediction-Using-ML/
│
├── data/
│   └── AmesHousing.csv
│
├── notebooks/
│   ├── __init__.py
│   ├── 01_dataset_inspection.py
│   ├── 02_feature_engineering_test.py
│   ├── 03_feature_selection.py
│   ├── 04_leakage_controlled_cv.py
│   ├── 05_model_training.py
│   └── 06_hyperparameter_tuning.py
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── missing_value_handler.py
│   ├── lot_frontage_imputer.py
│   └── preprocessing_pipeline.py
│
├── models/
│   ├── model_comparison.csv
│   └── hyperparameter_tuning_results.csv
│
├── app/
│
├── README.md
├── requirements.txt
├── .gitignore
└── .gitattributes
```

---

## 🚀 Project Progress

The project is being developed step by step, with a focus on understanding each stage of the Machine Learning workflow.

<details>
<summary><strong>✅ Phase 1 – Project Setup & Dataset Inspection</strong></summary>

- Set up the project structure and Python environment.
- Loaded and inspected the Ames Housing dataset.
- Analyzed dataset dimensions, data types, and statistical information.
- Checked missing values and duplicate records.
- Identified the target variable, `SalePrice`.
- Separated input features from the target.
- Excluded identifier columns (`Order` and `PID`) from model features.

**Dataset Findings:**

- Original dataset contains 2,930 records and 82 columns.
- The target variable is `SalePrice`.
- The dataset contains 79 input columns before excluding identifiers.

</details>

<details>
<summary><strong>✅ Phase 2 – Exploratory Data Analysis (EDA)</strong></summary>

- Studied the `SalePrice` distribution.
- Explored relationships between house prices and important numerical features.
- Performed correlation analysis.
- Analyzed numerical and categorical features.
- Investigated potential outliers using the IQR method.
- Studied numerical feature skewness.
- Explored house styles and neighborhood-level price differences.

**Key Findings:**

- `Overall Qual` has the strongest positive numerical correlation with `SalePrice` (approximately 0.799).
- The `SalePrice` distribution is right-skewed.
- 137 potential price outliers were identified.
- `1Story` is the most common house style.
- Neighborhoods show noticeable differences in average house prices.

Potential outliers were retained instead of being removed automatically.

</details>

<details>
<summary><strong>✅ Phase 3 – Data Cleaning & Preprocessing</strong></summary>

Implemented a reusable preprocessing pipeline using Scikit-learn.

### Data Preparation

- Separated input features and target variable.
- Created training, validation, and testing datasets.
- Preserved the original dataset.
- Implemented reusable preprocessing transformers.

### Missing Value Handling

- Created a custom transformer for special missing-value cases.
- Handled missing values involving garages, basements, fireplaces, masonry veneer, pools, and other property features.
- Used `None` for confirmed absent property features.
- Used `Unknown` where the actual category could not be determined.
- Implemented neighborhood-wise median imputation for `Lot Frontage`.
- Calculated imputation statistics using training data only.

### Preprocessing Pipeline

- Numerical median imputation.
- Categorical most-frequent imputation.
- One-Hot Encoding for categorical features.
- StandardScaler for numerical features.
- ColumnTransformer and Pipeline integration.
- Consistent feature transformation across datasets.

### Leakage Control

The preprocessing pipeline is fitted only on the training portion of the data. Validation and testing data are transformed using the learned training statistics.

A separate leakage-controlled cross-validation workflow was also implemented to ensure preprocessing is fitted independently inside each fold.

### Preprocessing Results

| Check | Result |
|---|---:|
| Training Records | 1,875 |
| Validation Records | 469 |
| Testing Records | 586 |
| Features After Feature Engineering | 85 |
| Final Preprocessed Features | 321 |
| Training Missing Values | 0 |
| Validation Missing Values | 0 |
| Testing Missing Values | 0 |
| Infinite Values | 0 |
| Feature Consistency Checks | Passed |
| Index Alignment Checks | Passed |

</details>

<details>
<summary><strong>✅ Phase 4 – Feature Engineering & Feature Selection</strong></summary>

### Feature Engineering

Created six additional features to capture useful information about residential properties.

| New Feature | Description |
|---|---|
| `TotalSF` | Total basement and above-ground floor area |
| `TotalBathrooms` | Combined full and half bathrooms |
| `HouseAgeAtSale` | Age of the house when sold |
| `RemodAgeAtSale` | Years since the last remodeling |
| `TotalPorchSF` | Combined porch area |
| `GarageAgeAtSale` | Age of the garage when sold |

**Feature Engineering Validation:**

- All six engineered features were tested.
- Feature calculations were independently verified.
- Exceptional missing values were preserved instead of being guessed.
- All feature validation checks passed.

### Feature Analysis

- Detected constant features — none found.
- Investigated low-variance features using VarianceThreshold.
- Identified highly correlated numerical feature pairs.
- Analyzed Random Forest feature importance.
- Compared All Features, Variance Selected, and Top 30 feature strategies.

### Leakage-Controlled Feature Selection

Implemented 5-fold cross-validation using raw training data.

For every fold:

- Created a fresh preprocessing pipeline.
- Fitted preprocessing only on that fold's training data.
- Applied feature selection using only fold training data.
- Trained a Random Forest Regressor.
- Evaluated predictions on the corresponding validation fold.

### Cross-Validation Results

| Feature Strategy | Mean Feature Count | Mean MAE | Mean R² |
|---|---:|---:|---:|
| **All Features** | 315.8 | **16,516.21** | 0.8700 |
| Variance Selected | 219.8 | 16,552.94 | 0.8713 |
| Top 30 | 30 | 17,051.16 | 0.8683 |

*Results are averages across five cross-validation folds.*

### Final Feature Selection Decision

**All Features** was selected because it achieved the lowest mean MAE across the five cross-validation folds.

- Final selected feature count: 321.
- Variance Selected reduced the feature count but did not improve mean MAE.
- Top 30 had a higher mean MAE than the other strategies.

### Independent Validation Results

| Metric | Result |
|---|---:|
| Selected Strategy | All Features |
| Selected Features | 321 |
| Validation MAE | 14,433.51 |
| Validation RMSE | 21,163.51 |
| Validation R² | 0.9177 |

The final test dataset has not been used for feature selection or model selection.

</details>

<details>
<summary><strong>✅ Phase 5 – Model Training</strong></summary>

Implemented and trained two regression models using the preprocessed training data.

### Models Implemented

1. Linear Regression
2. Random Forest Regressor

Both models were evaluated on the separate validation dataset.

### Validation Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 15,703.67 | 21,202.64 | 0.9174 |
| Random Forest | **14,433.51** | **21,163.51** | **0.9177** |

Random Forest performed better than Linear Regression across all three reported validation metrics.

</details>

<details>
<summary><strong>✅ Phase 6 – Model Evaluation</strong></summary>

Evaluated model predictions using visualizations and residual analysis.

### Evaluation Visualizations

- Actual vs Predicted plots for Linear Regression and Random Forest.
- Residuals vs Predicted plots.
- Residual distribution histograms.
- Reference line at zero for residual analysis.

### Evaluation Concepts

- Actual vs Predicted plots help compare predicted prices with actual prices.
- Residuals represent the difference between actual and predicted prices.
- Residual plots help inspect prediction errors and identify possible patterns.

</details>

<details>
<summary><strong>✅ Phase 7 – Hyperparameter Tuning</strong></summary>

Applied GridSearchCV to improve the Random Forest model.

### Tuning Setup

- Used a complete Scikit-learn Pipeline containing preprocessing and Random Forest.
- Used 5-fold cross-validation.
- Optimized negative Mean Absolute Error.
- Evaluated 16 hyperparameter combinations across 5 folds (80 fits).

### Hyperparameters Tested

| Parameter | Values |
|---|---|
| `n_estimators` | 100, 200 |
| `max_depth` | None, 20 |
| `min_samples_split` | 2, 5 |
| `min_samples_leaf` | 1, 2 |

### Best Parameters

- `n_estimators`: 200
- `max_depth`: 20
- `min_samples_split`: 2
- `min_samples_leaf`: 1

### Tuning Results

| Metric | Original Random Forest | Tuned Random Forest |
|---|---:|---:|
| MAE | 14,433.51 | **14,244.59** |
| RMSE | 21,163.51 | **20,705.88** |
| R² Score | 0.9177 | **0.9212** |

The tuned Random Forest improved all three validation metrics compared with the original Random Forest.

**Result:** The Tuned Random Forest was selected as the final candidate model for untouched test-set evaluation.

</details>

---

## 🔄 What's Next?

- [ ] Final evaluation using the untouched test dataset.
- [ ] Save the final trained model.
- [ ] Prepare the model for deployment.
- [ ] Build an interactive Streamlit application.
- [ ] Final documentation and GitHub review.

---

## 🎯 Learning Objectives

Through this project, I aim to understand:

- How to inspect and explore a real-world dataset.
- How to handle missing values and unusual records.
- How to prepare data for Machine Learning.
- How to build reusable preprocessing pipelines.
- How feature engineering and feature selection work.
- How to prevent data leakage during model development.
- How regression algorithms make predictions.
- How to evaluate and improve Machine Learning models.
- How to tune hyperparameters using cross-validation.
- How to build a simple application using a trained model.

---

## 📝 Important Note

This project uses historical US housing data from Ames, Iowa. It is intended for educational purposes and is not designed to estimate current property prices in India.

The project is being developed step by step as a learning exercise.

The `models/` and `app/` directories are used for model outputs and upcoming application development.