# 🏡 House Price Prediction Using Machine Learning

A beginner-friendly Machine Learning project that predicts residential house prices using Python and Scikit-learn.

The project follows a step-by-step Machine Learning workflow, starting with understanding the dataset and exploring the data, followed by preprocessing, feature engineering, model training, and evaluation.

---

## 📌 About the Project

This project uses the **Ames Housing Dataset**, which contains information about residential properties in Ames, Iowa, USA.

The goal is to understand how different property features — such as living area, overall quality, number of rooms, and garage size — relate to house prices and how Machine Learning models can use this information to make predictions.

**Dataset Overview**

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
│   ├── 01_dataset_inspection.py
│   └── 02_feature_engineering_test.py
│
├── src/
│   ├── data_preprocessing.py
│   └── feature_engineering.py
│
├── models/
│
├── app/
│
├── README.md
└── requirements.txt
```

---

## 🚀 Project Progress

The project is being developed step by step, with a focus on understanding each stage of the Machine Learning workflow.

<details>
<summary><strong>✅ Phase 1 – Dataset Inspection & Exploratory Data Analysis (EDA)</strong></summary>

- Loaded and inspected the Ames Housing dataset.
- Analyzed dataset dimensions, data types, and statistical information.
- Checked missing values and duplicate records.
- Studied the `SalePrice` distribution.
- Explored relationships between house prices and important features.
- Performed correlation analysis.
- Analyzed numerical and categorical features.
- Investigated potential outliers using the IQR method.
- Studied numerical feature skewness.

**Key Findings:**

- `Overall Qual` has the strongest positive numerical correlation with `SalePrice` (approximately 0.799).
- The `SalePrice` distribution is right-skewed.
- 137 potential price outliers were identified.
- `1Story` is the most common house style.
- Neighborhoods show noticeable differences in average house prices.

Potential outliers were retained instead of being removed automatically.

</details>

<details>
<summary><strong>✅ Phase 2 – Data Preparation & Missing Value Investigation</strong></summary>

- Separated input features and target variable.
- Excluded identifier columns (`Order` and `PID`).
- Implemented an 80/20 train-test split.
- Investigated missing values in numerical and categorical features.
- Studied special cases involving garages, basements, fireplaces, masonry veneer, pools, and other property features.
- Finalized appropriate missing-value handling strategies.

**Important decisions:**

- Use `None` when a property feature is confirmed to be absent.
- Use `Unknown` when the actual category cannot be determined.
- Use neighborhood-wise median imputation for `Lot Frontage`.
- Calculate imputation statistics using training data only.

</details>

<details>
<summary><strong>✅ Phase 3 – Data Preprocessing</strong></summary>

Implemented a preprocessing pipeline using Scikit-learn.

- Created separate working copies of training and testing data.
- Handled special missing-value cases.
- Implemented numerical median imputation.
- Implemented categorical most-frequent imputation.
- Applied One-Hot Encoding to categorical features.
- Applied StandardScaler to numerical features.
- Used ColumnTransformer and Pipeline.
- Fitted preprocessing transformations on training data only.
- Applied the learned transformations to testing data.

**Preprocessing Results:**

| Check | Result |
|---|---:|
| Training Records | 2,344 |
| Testing Records | 586 |
| Original Input Features | 85 |
| Final Encoded Features | 330 |
| Remaining Missing Values | 0 |
| Infinite Values | 0 |
| Automatic Validation Checks | Passed |

</details>

<details>
<summary><strong>✅ Phase 4 – Feature Engineering</strong></summary>

Created six additional features to capture useful information about residential properties.

| New Feature | Description |
|---|---|
| `TotalSF` | Total basement and above-ground floor area |
| `TotalBathrooms` | Combined full and half bathrooms |
| `HouseAgeAtSale` | Age of the house when sold |
| `RemodAgeAtSale` | Years since the last remodeling |
| `TotalPorchSF` | Combined porch area |
| `GarageAgeAtSale` | Age of the garage when sold |

**Validation:**

- All six engineered features were tested.
- Feature calculations were independently verified.
- Exceptional missing values were preserved instead of being guessed.
- All feature validation checks passed.

</details>

---

## 🔄 What's Next?

The next stages of the project are:

- [ ] Feature selection and analysis.
- [ ] Linear Regression implementation.
- [ ] Random Forest Regressor implementation.
- [ ] Model evaluation using MAE, RMSE, and R².
- [ ] Comparison of different models.
- [ ] Hyperparameter tuning and model improvement.
- [ ] Saving the trained model.
- [ ] Building an interactive Streamlit application.

---

## 🎯 Learning Objectives

Through this project, I aim to understand:

- How to inspect and explore a real-world dataset.
- How to handle missing values and unusual records.
- How to prepare data for Machine Learning.
- How feature engineering and feature selection work.
- How regression algorithms make predictions.
- How to evaluate and improve Machine Learning models.
- How to build a simple application using a trained model.

---

## 📝 Important Note

This project uses historical US housing data from Ames, Iowa. It is intended for educational purposes and is not designed to estimate current property prices in India.

**The project is being developed step by step as a learning exercise.**