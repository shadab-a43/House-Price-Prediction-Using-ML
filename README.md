

# House Price Prediction Using Machine Learning

An end-to-end Machine Learning project focused on predicting residential house prices using Python and Scikit-learn.

## Project Overview

The goal of this project is to understand and implement the complete Machine Learning workflow, from dataset inspection and Exploratory Data Analysis (EDA) to data preprocessing, model training, and evaluation.

This project uses the Ames Housing dataset to predict residential house sale prices.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit (planned)

## Dataset

- **Dataset:** Ames Housing Dataset
- **Records:** 2,930
- **Features:** 82 columns
- **Target Variable:** `SalePrice`

The dataset contains residential property information from Ames, Iowa, USA.

## Project Structure

```text
House-Price-Prediction-Using-ML/
│
├── data/
│   └── AmesHousing.csv
│
├── notebooks/
│   └── 01_dataset_inspection.py
│
├── src/
│   └── data_preprocessing.py
│
├── models/
├── app/
│
├── README.md
└── requirements.txt
```

## Project Progress

### Completed

- Project repository and folder structure created.
- Ames Housing dataset added.
- Dataset loaded and inspected using Pandas.
- Dataset dimensions and column information analyzed.
- Missing values identified and investigated.
- Duplicate row check completed (0 duplicate rows).
- Statistical summary of `SalePrice` generated.
- Initial Exploratory Data Analysis (EDA) completed.
- Data preprocessing phase started.
- Identifier columns (`Order` and `PID`) identified for exclusion from model features.
- Target variable `SalePrice` verified (no missing values).
- Train-test split implemented (80% training, 20% testing).
- Numerical and categorical features separated.
- Missing values in training features analyzed.

### EDA Analysis Completed

1. SalePrice Distribution Analysis.
2. Living Area vs SalePrice.
3. Overall Quality vs SalePrice.
4. Garage Area vs SalePrice.
5. Year Built vs SalePrice.
6. Correlation Analysis using a numerical feature heatmap.
7. Top 10 numerical features correlated with `SalePrice` identified.
8. Missing Values Visualization (Top 10 columns).
9. SalePrice Outlier Analysis using a Boxplot.
10. SalePrice Outlier Detection using the IQR Method.
11. `Gr Liv Area` Outlier Analysis using a Boxplot and IQR Method.
12. `Lot Area` Outlier Analysis using a Boxplot and IQR Method.
13. `Garage Area` Outlier Analysis using a Boxplot and IQR Method.
14. House Style Category Analysis.
15. House Style Distribution Visualization using a Count Plot.
16. House Style vs SalePrice Analysis using a Bar Plot.
17. Neighborhood Distribution and SalePrice Analysis.
18. Categorical Feature Analysis (`Bldg Type`, `Exterior 1st`, `Kitchen Qual`, and `MS Zoning`).
19. Feature Relationship Analysis using scatter plots.
20. Numerical Feature Skewness Analysis.

### Key EDA Observations

- `Overall Qual` has the strongest positive correlation with `SalePrice` among numerical features, with a correlation coefficient of approximately 0.799.
- SalePrice distribution is right-skewed, indicating some high-priced properties.
- The IQR method identified 137 potential SalePrice outliers, approximately 4.68% of the dataset.
- `Gr Liv Area` has 75 potential outliers (approximately 2.56%).
- `Lot Area` has 127 potential outliers (approximately 4.33%).
- `Garage Area` has 42 potential outliers (approximately 1.43%).
- `1Story` is the most common house style, with 1,481 houses.
- The dataset contains 8 different house styles.
- NoRidge has the highest average SalePrice among neighborhoods, while MeadowV has the lowest.
- `SalePrice` has a skewness value of approximately 1.744.
- Potential outliers will be retained for further evaluation rather than removed automatically.
- Missing values may represent the absence of certain property features and require careful handling.

### Data Preprocessing — In Progress

**Dataset Preparation**

- Original dataset loaded without modifying the source CSV.
- Identifier columns (`Order` and `PID`) excluded from model features.
- Target variable separated from input features.
- Train-test split completed (80% training, 20% testing).
- Training data contains 2,344 records; testing data contains 586 records.
- Numerical and categorical features separated (36 numerical and 43 categorical).
- Missing values analyzed in training features (11 numerical columns and 15 categorical columns).

**Missing Value Investigation Completed**

- Garage-related missing values investigated, including properties without garages and exceptional records.
- Basement-related missing values investigated, including properties without basements and exceptional records.
- Fireplace quality missing values investigated and confirmed to correspond to properties without fireplaces.
- Masonry veneer missing values investigated, including missing types, zero areas, positive areas, and records where both values are missing.

**Upcoming Preprocessing Tasks**

- Investigate `Lot Frontage` missing values using training data.
- Investigate remaining numerical and categorical missing values.
- Finalize missing-value imputation strategies.
- Implement numerical and categorical imputation.
- Build a leakage-safe preprocessing pipeline.
- Categorical feature encoding.
- Numerical feature scaling where appropriate.
- Feature engineering.

### Upcoming

- Complete data cleaning and preprocessing.
- Linear Regression model implementation.
- Random Forest Regressor implementation.
- Model evaluation using MAE, RMSE, and R².
- Model comparison.
- Streamlit application development.

## Learning Objective

To build a practical understanding of Machine Learning concepts by developing a complete house price prediction project step by step.

---

**Note:** This project uses US housing data from Ames, Iowa. It is an educational Machine Learning project and is not designed to estimate current Indian property prices.

