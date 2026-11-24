import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Feature Engineering import
try:
    from .feature_engineering import add_engineered_features
except ImportError:
    from feature_engineering import add_engineered_features


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\n========== STEP 1: LOAD DATASET ==========")

df = pd.read_csv("data/AmesHousing.csv")

print("Dataset Shape:", df.shape)
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])


# ============================================================
# STEP 2: INITIAL MISSING VALUES ANALYSIS
# ============================================================

print("\n========== STEP 2: MISSING VALUES ANALYSIS ==========")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

print("\nMissing Values:")
print(missing_values.sort_values(ascending=False))

print("\nTotal Missing Values:", missing_values.sum())


# ============================================================
# STEP 3: TARGET AND FEATURE SEPARATION
# ============================================================

print("\n========== STEP 3: TARGET SEPARATION ==========")

X = df.drop(columns=["SalePrice", "Order", "PID"])
y = df["SalePrice"]

print("Feature Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# STEP 4: TRAIN-TEST SPLIT
# ============================================================

print("\n========== STEP 4: TRAIN-TEST SPLIT ==========")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================

print("\n========== STEP 5: FEATURE ENGINEERING ==========")

# Same feature engineering function on both datasets.
X_train = add_engineered_features(X_train)
X_test = add_engineered_features(X_test)

engineered_features = [
    "TotalSF",
    "TotalBathrooms",
    "HouseAgeAtSale",
    "RemodAgeAtSale",
    "TotalPorchSF",
    "GarageAgeAtSale"
]

print("New Engineered Features:")
print(engineered_features)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# Verify that all engineered features exist.
for column in engineered_features:
    assert column in X_train.columns
    assert column in X_test.columns

# Train and test must have identical feature columns.
assert X_train.columns.equals(X_test.columns)

print("\nFeature Engineering checks passed!")


# ============================================================
# STEP 6: NUMERICAL AND CATEGORICAL FEATURE SEPARATION
# ============================================================

print("\n========== STEP 6: FEATURE TYPE ANALYSIS ==========")

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("Numerical Features:", len(numerical_columns))
print("Categorical Features:", len(categorical_columns))

print("\nTotal Features:",
      len(numerical_columns) + len(categorical_columns))

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# STEP 7: CREATE WORKING COPIES
# ============================================================

print("\n========== STEP 7: WORKING COPIES ==========")

X_train_processed = X_train.copy()
X_test_processed = X_test.copy()

print("Training Shape:", X_train_processed.shape)
print("Testing Shape:", X_test_processed.shape)

print("\nMissing Values Before Preprocessing:")
print("Training:", X_train_processed.isnull().sum().sum())
print("Testing:", X_test_processed.isnull().sum().sum())


# ============================================================
# STEP 8: SPECIAL MISSING VALUE HANDLING
# ============================================================

print("\n========== STEP 8: SPECIAL MISSING VALUES ==========")


# ------------------------------------------------------------
# 8A: GARAGE
# ------------------------------------------------------------

print("\n--- Garage Handling ---")

garage_absent_train = (
    X_train_processed["Garage Cars"].eq(0)
    & X_train_processed["Garage Area"].eq(0)
)

garage_absent_test = (
    X_test_processed["Garage Cars"].eq(0)
    & X_test_processed["Garage Area"].eq(0)
)

garage_categorical = [
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond"
]

for column in garage_categorical:

    X_train_processed.loc[
        garage_absent_train & X_train_processed[column].isna(),
        column
    ] = "None"

    X_test_processed.loc[
        garage_absent_test & X_test_processed[column].isna(),
        column
    ] = "None"

    X_train_processed.loc[
        ~garage_absent_train & X_train_processed[column].isna(),
        column
    ] = "Unknown"

    X_test_processed.loc[
        ~garage_absent_test & X_test_processed[column].isna(),
        column
    ] = "Unknown"


# No garage means garage year is not applicable.
X_train_processed.loc[
    garage_absent_train, "Garage Yr Blt"
] = X_train_processed.loc[
    garage_absent_train, "Garage Yr Blt"
].fillna(0)

X_test_processed.loc[
    garage_absent_test, "Garage Yr Blt"
] = X_test_processed.loc[
    garage_absent_test, "Garage Yr Blt"
].fillna(0)


# ------------------------------------------------------------
# 8B: BASEMENT
# ------------------------------------------------------------

print("\n--- Basement Handling ---")

basement_absent_train = (
    X_train_processed["Total Bsmt SF"].eq(0)
)

basement_absent_test = (
    X_test_processed["Total Bsmt SF"].eq(0)
)

basement_categorical = [
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2"
]

for column in basement_categorical:

    X_train_processed.loc[
        basement_absent_train & X_train_processed[column].isna(),
        column
    ] = "None"

    X_test_processed.loc[
        basement_absent_test & X_test_processed[column].isna(),
        column
    ] = "None"

    X_train_processed.loc[
        ~basement_absent_train & X_train_processed[column].isna(),
        column
    ] = "Unknown"

    X_test_processed.loc[
        ~basement_absent_test & X_test_processed[column].isna(),
        column
    ] = "Unknown"


# Confirmed no-basement properties have zero basement measurements.
basement_numerical = [
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Total Bsmt SF",
    "Bsmt Unf SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath"
]

for column in basement_numerical:

    X_train_processed.loc[
        basement_absent_train, column
    ] = X_train_processed.loc[
        basement_absent_train, column
    ].fillna(0)

    X_test_processed.loc[
        basement_absent_test, column
    ] = X_test_processed.loc[
        basement_absent_test, column
    ].fillna(0)


# ------------------------------------------------------------
# 8C: MASONRY VENEER
# ------------------------------------------------------------

print("\n--- Masonry Veneer Handling ---")

for data in [X_train_processed, X_test_processed]:

    no_veneer = (
        data["Mas Vnr Type"].isna()
        & data["Mas Vnr Area"].eq(0)
    )

    unknown_veneer = (
        data["Mas Vnr Type"].isna()
        & ~data["Mas Vnr Area"].eq(0)
    )

    data.loc[no_veneer, "Mas Vnr Type"] = "None"
    data.loc[unknown_veneer, "Mas Vnr Type"] = "Unknown"


# ------------------------------------------------------------
# 8D: POOL QUALITY
# ------------------------------------------------------------

print("\n--- Pool Quality Handling ---")

for data in [X_train_processed, X_test_processed]:

    no_pool = data["Pool Area"].eq(0)

    data.loc[
        data["Pool QC"].isna() & no_pool,
        "Pool QC"
    ] = "None"

    data.loc[
        data["Pool QC"].isna() & ~no_pool,
        "Pool QC"
    ] = "Unknown"


# ------------------------------------------------------------
# 8E: FIREPLACE QUALITY
# ------------------------------------------------------------

print("\n--- Fireplace Quality Handling ---")

for data in [X_train_processed, X_test_processed]:

    no_fireplace = data["Fireplaces"].eq(0)

    data.loc[
        data["Fireplace Qu"].isna() & no_fireplace,
        "Fireplace Qu"
    ] = "None"

    data.loc[
        data["Fireplace Qu"].isna() & ~no_fireplace,
        "Fireplace Qu"
    ] = "Unknown"


# ------------------------------------------------------------
# 8F: OTHER ABSENCE-BASED CATEGORICAL FEATURES
# ------------------------------------------------------------

print("\n--- Other Absence-Based Features ---")

absence_columns = [
    "Alley",
    "Fence",
    "Misc Feature"
]

for column in absence_columns:

    X_train_processed[column] = (
        X_train_processed[column].fillna("None")
    )

    X_test_processed[column] = (
        X_test_processed[column].fillna("None")
    )


# ============================================================
# STEP 9: LOT FRONTAGE IMPUTATION
# ============================================================

print("\n========== STEP 9: LOT FRONTAGE IMPUTATION ==========")

# Learn neighborhood medians from training data only.
neighborhood_medians = (
    X_train_processed.groupby("Neighborhood")["Lot Frontage"]
    .median()
)

# Global training median as fallback.
global_frontage_median = (
    X_train_processed["Lot Frontage"].median()
)

# Apply training-derived values to both datasets.
for data in [X_train_processed, X_test_processed]:

    missing_frontage = data["Lot Frontage"].isna()

    neighborhood_values = (
        data.loc[missing_frontage, "Neighborhood"]
        .map(neighborhood_medians)
    )

    data.loc[missing_frontage, "Lot Frontage"] = (
        neighborhood_values.fillna(global_frontage_median)
    )

print("Global Training Median:", global_frontage_median)

print(
    "Training Missing:",
    X_train_processed["Lot Frontage"].isna().sum()
)

print(
    "Testing Missing:",
    X_test_processed["Lot Frontage"].isna().sum()
)


# ============================================================
# STEP 10: FINAL FEATURE TYPE LISTS
# ============================================================

print("\n========== STEP 10: FINAL FEATURE TYPES ==========")

print("Numerical Features:", len(numerical_columns))
print("Categorical Features:", len(categorical_columns))


# ============================================================
# STEP 11: NUMERICAL AND CATEGORICAL PIPELINES
# ============================================================

print("\n========== STEP 11: IMPUTATION PIPELINES ==========")

# Numerical:
# 1. Missing values -> Training median
# 2. Scaling -> StandardScaler

numerical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# Categorical:
# 1. Missing values -> Training most frequent category
# 2. Encoding -> OneHotEncoder

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


# ============================================================
# STEP 12: COLUMN TRANSFORMER
# ============================================================

print("\n========== STEP 12: COLUMN TRANSFORMER ==========")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_columns
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    ],
    remainder="drop"
)


# ============================================================
# STEP 13: FIT AND TRANSFORM
# ============================================================

print("\n========== STEP 13: FIT AND TRANSFORM ==========")

# Fit only on training data.
X_train_array = preprocessor.fit_transform(
    X_train_processed
)

# Apply learned transformations to testing data.
X_test_array = preprocessor.transform(
    X_test_processed
)

# Get generated feature names.
feature_names = preprocessor.get_feature_names_out()

# Convert arrays into DataFrames.
X_train_preprocessed = pd.DataFrame(
    X_train_array,
    columns=feature_names,
    index=X_train.index
)

X_test_preprocessed = pd.DataFrame(
    X_test_array,
    columns=feature_names,
    index=X_test.index
)


# ============================================================
# STEP 14: FINAL PREPROCESSING VALIDATION
# ============================================================

print("\n========== STEP 14: FINAL VALIDATION ==========")

print("\nOriginal Training Shape:", X_train.shape)
print("Processed Training Shape:", X_train_preprocessed.shape)

print("\nOriginal Testing Shape:", X_test.shape)
print("Processed Testing Shape:", X_test_preprocessed.shape)

print("\nRemaining Missing Values:")
print(
    "Training:",
    X_train_preprocessed.isna().sum().sum()
)

print(
    "Testing:",
    X_test_preprocessed.isna().sum().sum()
)

print("\nInfinite Values:")
print(
    "Training:",
    np.isinf(X_train_preprocessed.to_numpy()).sum()
)

print(
    "Testing:",
    np.isinf(X_test_preprocessed.to_numpy()).sum()
)

print("\nAll Processed Columns Numeric:",
      all(
          pd.api.types.is_numeric_dtype(dtype)
          for dtype in X_train_preprocessed.dtypes
      )
)


# ============================================================
# STEP 15: FINAL CONSISTENCY CHECKS
# ============================================================

print("\n========== STEP 15: FINAL REVIEW ==========")

# 1. Train and test columns must match.
same_columns = X_train_preprocessed.columns.equals(
    X_test_preprocessed.columns
)

print("Same Feature Columns:", same_columns)


# 2. Feature names must be unique.
unique_columns = X_train_preprocessed.columns.is_unique

print("Unique Feature Names:", unique_columns)


# 3. Target indexes must match feature indexes.
train_index_aligned = (
    X_train_preprocessed.index.equals(y_train.index)
)

test_index_aligned = (
    X_test_preprocessed.index.equals(y_test.index)
)

print("Training Index Aligned:", train_index_aligned)
print("Testing Index Aligned:", test_index_aligned)


# 4. All values must be finite.
train_finite = np.isfinite(
    X_train_preprocessed.to_numpy()
).all()

test_finite = np.isfinite(
    X_test_preprocessed.to_numpy()
).all()

print("Training Values Finite:", train_finite)
print("Testing Values Finite:", test_finite)


# 5. Original datasets must remain unchanged.
original_train_unchanged = (
    X_train.shape == (len(y_train), len(engineered_features) + X.shape[1])
)

original_test_unchanged = (
    X_test.shape == (len(y_test), len(engineered_features) + X.shape[1])
)

print("Original Training Shape Valid:", original_train_unchanged)
print("Original Testing Shape Valid:", original_test_unchanged)


# 6. Special categorical columns must have no missing values.
special_categorical = (
    garage_categorical
    + basement_categorical
    + ["Mas Vnr Type", "Pool QC", "Fireplace Qu"]
    + absence_columns
)

special_categorical_missing = (
    X_train_processed[special_categorical].isna().sum().sum()
    + X_test_processed[special_categorical].isna().sum().sum()
)

print(
    "Special Categorical Missing Values:",
    special_categorical_missing
)


# 7. Lot Frontage must have no missing values.
lot_frontage_missing = (
    X_train_processed["Lot Frontage"].isna().sum()
    + X_test_processed["Lot Frontage"].isna().sum()
)

print("Lot Frontage Missing Values:", lot_frontage_missing)


# 8. Target sizes must remain correct.
print("\nTraining Target Records:", len(y_train))
print("Testing Target Records:", len(y_test))


# ============================================================
# STEP 16: AUTOMATIC ASSERTIONS
# ============================================================

print("\n========== STEP 16: AUTOMATIC CHECKS ==========")

assert same_columns
assert unique_columns

assert train_index_aligned
assert test_index_aligned

assert train_finite
assert test_finite

assert original_train_unchanged
assert original_test_unchanged

assert special_categorical_missing == 0
assert lot_frontage_missing == 0

assert len(X_train_preprocessed) == len(y_train)
assert len(X_test_preprocessed) == len(y_test)

assert X_train_preprocessed.isna().sum().sum() == 0
assert X_test_preprocessed.isna().sum().sum() == 0

print("All final review checks passed!")


# ============================================================
# STEP 17: FINAL SUMMARY
# ============================================================

print("\n========== PREPROCESSING SUMMARY ==========")

print("Original Dataset:", df.shape)

print("Training Features:", X_train.shape)
print("Testing Features:", X_test.shape)

print("Training Target:", y_train.shape)
print("Testing Target:", y_test.shape)

print("Preprocessed Training:", X_train_preprocessed.shape)
print("Preprocessed Testing:", X_test_preprocessed.shape)

print("Numerical Input Features:", len(numerical_columns))
print("Categorical Input Features:", len(categorical_columns))

print("Engineered Features:", len(engineered_features))
print("Final Encoded Features:", len(feature_names))

print("\n========== PREPROCESSING COMPLETED SUCCESSFULLY ==========")
