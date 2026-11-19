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
- Missing values identified and examined.
- Duplicate row check completed (0 duplicate rows).
- Statistical summary of `SalePrice` generated.
- Initial Exploratory Data Analysis (EDA) completed.

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

**Key EDA Observations:**

- `Overall Qual` has the strongest positive correlation with `SalePrice` among the numerical features, with a correlation coefficient of approximately 0.799.
- SalePrice distribution is right-skewed, indicating some high-priced properties.
- The IQR method identified 137 potential SalePrice outliers, approximately 4.68% of the dataset.
- Potential outliers will be retained for further evaluation rather than removed automatically.
- Missing values may represent the absence of certain property features and require careful handling.

### Upcoming

- Numerical feature outlier analysis (`Gr Liv Area`, `Lot Area`, `Garage Area`, etc.).
- Categorical feature analysis.
- Neighborhood and SalePrice relationship analysis.
- Further feature relationship and skewness analysis.
- Data cleaning and preprocessing.
- Feature engineering.
- Train-test split (80% training, 20% testing).
- Linear Regression model implementation.
- Random Forest Regressor implementation.
- Model evaluation using MAE, RMSE, and R².
- Model comparison.
- Streamlit application development.

## Learning Objective

To build a practical understanding of Machine Learning concepts by developing a complete house price prediction project step by step.

---

**Note:** This project uses US housing data from Ames, Iowa. It is an educational Machine Learning project and is not designed to estimate current Indian property prices.