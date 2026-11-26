
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

try:
    from .feature_engineering import add_engineered_features
except ImportError:
    from feature_engineering import add_engineered_features


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

print("\n========== STEP 1: LOAD DATASET ==========")

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "AmesHousing.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset Shape:", df.shape)
print("Dataset Loaded Successfully!")


# ============================================================
# STEP 2: INITIAL DATA ANALYSIS
# ============================================================

print("\n========== STEP 2: INITIAL DATA ANALYSIS ==========")

print("Total Records:", df.shape[0])
print("Total Columns:", df.shape[1])

print("Total Missing Values:", df.isnull().sum().sum())

print("\nTarget Variable: SalePrice")
print("Target Missing Values:", df["SalePrice"].isnull().sum())


# ============================================================
# STEP 3: SEPARATE FEATURES AND TARGET
# ============================================================

print("\n========== STEP 3: FEATURES AND TARGET ==========")

X = df.drop(
    columns=["SalePrice", "Order", "PID"],
    errors="ignore"
)

y = df["SalePrice"].copy()

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# STEP 4: TRAIN-VALIDATION-TEST SPLIT
# ============================================================

print("\n========== STEP 4: TRAIN-VALIDATION-TEST SPLIT ==========")

# First: Keep 20% data completely separate for final testing.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Second: Divide remaining 80% into training and validation.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42
)

print("Training Records:", X_train.shape[0])
print("Validation Records:", X_val.shape[0])
print("Testing Records:", X_test.shape[0])

print("Training Target:", y_train.shape)
print("Validation Target:", y_val.shape)
print("Testing Target:", y_test.shape)


# ============================================================
# STEP 5: FEATURE ENGINEERING
# ============================================================

print("\n========== STEP 5: FEATURE ENGINEERING ==========")

X_train = add_engineered_features(X_train)
X_val = add_engineered_features(X_val)
X_test = add_engineered_features(X_test)

print("Training Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Testing Shape:", X_test.shape)

print("Engineered Features Added Successfully!")

assert list(X_train.columns) == list(X_val.columns)
assert list(X_train.columns) == list(X_test.columns)

print("Feature Columns Match: True")


# ============================================================
# STEP 6: IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

print("\n========== STEP 6: FEATURE TYPE IDENTIFICATION ==========")

numerical_features = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))
print("Total Features:", len(numerical_features) + len(categorical_features))


# ============================================================
# STEP 7: CREATE WORKING COPIES
# ============================================================

print("\n========== STEP 7: CREATE WORKING COPIES ==========")

X_train_processed = X_train.copy()
X_val_processed = X_val.copy()
X_test_processed = X_test.copy()

print("Training Copy:", X_train_processed.shape)
print("Validation Copy:", X_val_processed.shape)
print("Testing Copy:", X_test_processed.shape)


# ============================================================
# STEP 8: SPECIAL MISSING VALUE HANDLING
# ============================================================

print("\n========== STEP 8: SPECIAL MISSING VALUE HANDLING ==========")

# ------------------------------------------------------------
# 8.1: GARAGE
# ------------------------------------------------------------

garage_categorical = [
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond"
]

for data in [
    X_train_processed,
    X_val_processed,
    X_test_processed
]:

    if "Garage Cars" in data.columns and "Garage Area" in data.columns:

        no_garage = (
            data["Garage Cars"].eq(0)
            & data["Garage Area"].eq(0)
        )

        for column in garage_categorical:

            if column in data.columns:

                data.loc[
                    no_garage & data[column].isna(),
                    column
                ] = "None"

                data.loc[
                    ~no_garage & data[column].isna(),
                    column
                ] = "Unknown"

print("Garage Missing Values Handled!")


# ------------------------------------------------------------
# 8.2: BASEMENT
# ------------------------------------------------------------

basement_categorical = [
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2"
]

for data in [
    X_train_processed,
    X_val_processed,
    X_test_processed
]:

    if "Total Bsmt SF" in data.columns:

        no_basement = data["Total Bsmt SF"].eq(0)

        for column in basement_categorical:

            if column in data.columns:

                data.loc[
                    no_basement & data[column].isna(),
                    column
                ] = "None"

                data.loc[
                    ~no_basement & data[column].isna(),
                    column
                ] = "Unknown"

print("Basement Missing Values Handled!")


# ------------------------------------------------------------
# 8.3: MASONRY VENEER
# ------------------------------------------------------------

for data in [
    X_train_processed,
    X_val_processed,
    X_test_processed
]:

    if (
        "Mas Vnr Type" in data.columns
        and "Mas Vnr Area" in data.columns
    ):

        missing_type = data["Mas Vnr Type"].isna()

        no_masonry = data["Mas Vnr Area"].eq(0)

        data.loc[
            missing_type & no_masonry,
            "Mas Vnr Type"
        ] = "None"

        data.loc[
            missing_type & ~no_masonry,
            "Mas Vnr Type"
        ] = "Unknown"

print("Masonry Veneer Missing Values Handled!")


# ------------------------------------------------------------
# 8.4: POOL QUALITY
# ------------------------------------------------------------

for data in [
    X_train_processed,
    X_val_processed,
    X_test_processed
]:

    if "Pool QC" in data.columns and "Pool Area" in data.columns:

        no_pool = data["Pool Area"].eq(0)

        data.loc[
            data["Pool QC"].isna() & no_pool,
            "Pool QC"
        ] = "None"

        data.loc[
            data["Pool QC"].isna() & ~no_pool,
            "Pool QC"
        ] = "Unknown"

print("Pool Missing Values Handled!")


# ------------------------------------------------------------
# 8.5: FIREPLACE QUALITY
# ------------------------------------------------------------

for data in [
    X_train_processed,
    X_val_processed,
    X_test_processed
]:

    if (
        "Fireplace Qu" in data.columns
        and "Fireplaces" in data.columns
    ):

        no_fireplace = data["Fireplaces"].eq(0)

        data.loc[
            data["Fireplace Qu"].isna() & no_fireplace,
            "Fireplace Qu"
        ] = "None"

        data.loc[
            data["Fireplace Qu"].isna() & ~no_fireplace,
            "Fireplace Qu"
        ] = "Unknown"

print("Fireplace Missing Values Handled!")


# ------------------------------------------------------------
# 8.6: OTHER ABSENCE-BASED CATEGORICAL FEATURES
# ------------------------------------------------------------

absence_categories = [
    "Alley",
    "Fence",
    "Misc Feature"
]

for data in [
    X_train_processed,
    X_val_processed,
    X_test_processed
]:

    for column in absence_categories:

        if column in data.columns:
            data[column] = data[column].fillna("None")

print("Other Absence-Based Categories Handled!")


# ============================================================
# STEP 9: LOT FRONTAGE IMPUTATION
# ============================================================

print("\n========== STEP 9: LOT FRONTAGE IMPUTATION ==========")

# Calculate neighborhood medians using TRAINING data only.

if "Lot Frontage" in X_train_processed.columns:

    neighborhood_medians = (
        X_train_processed
        .groupby("Neighborhood")["Lot Frontage"]
        .median()
    )

    global_median = X_train_processed["Lot Frontage"].median()

    for data in [
        X_train_processed,
        X_val_processed,
        X_test_processed
    ]:

        neighborhood_values = data["Neighborhood"].map(
            neighborhood_medians
        )

        data["Lot Frontage"] = (
            data["Lot Frontage"]
            .fillna(neighborhood_values)
            .fillna(global_median)
        )

    print("Neighborhood Medians Calculated From Training Data!")
    print("Global Training Median:", global_median)

print("Lot Frontage Imputation Completed!")


# ============================================================
# STEP 10: FINAL FEATURE TYPE IDENTIFICATION
# ============================================================

print("\n========== STEP 10: FINAL FEATURE TYPE IDENTIFICATION ==========")

numerical_features = X_train_processed.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X_train_processed.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

print("Numerical Features:", len(numerical_features))
print("Categorical Features:", len(categorical_features))
print("Total Features:", len(numerical_features) + len(categorical_features))


# ============================================================
# STEP 11: NUMERICAL AND CATEGORICAL PIPELINES
# ============================================================

print("\n========== STEP 11: CREATE PREPROCESSING PIPELINES ==========")

numerical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                keep_empty_features=True
            )
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent",
                keep_empty_features=True
            )
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

print("Numerical Pipeline Created!")
print("Categorical Pipeline Created!")


# ============================================================
# STEP 12: COLUMN TRANSFORMER
# ============================================================

print("\n========== STEP 12: COLUMN TRANSFORMER ==========")

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ],
    remainder="drop"
)

print("Column Transformer Created!")


# ============================================================
# STEP 13: FIT AND TRANSFORM DATA
# ============================================================

print("\n========== STEP 13: FIT AND TRANSFORM DATA ==========")

# IMPORTANT:
# Fit only on training data.
# Validation and test data are only transformed.

X_train_array = preprocessor.fit_transform(
    X_train_processed
)

X_val_array = preprocessor.transform(
    X_val_processed
)

X_test_array = preprocessor.transform(
    X_test_processed
)

# Get final feature names.

feature_names = preprocessor.get_feature_names_out()

# Convert arrays into DataFrames.

X_train_preprocessed = pd.DataFrame(
    X_train_array,
    columns=feature_names,
    index=X_train_processed.index
)

X_val_preprocessed = pd.DataFrame(
    X_val_array,
    columns=feature_names,
    index=X_val_processed.index
)

X_test_preprocessed = pd.DataFrame(
    X_test_array,
    columns=feature_names,
    index=X_test_processed.index
)

print("Training Preprocessed Shape:", X_train_preprocessed.shape)
print("Validation Preprocessed Shape:", X_val_preprocessed.shape)
print("Testing Preprocessed Shape:", X_test_preprocessed.shape)

print("Preprocessing Completed!")


# ============================================================
# STEP 14: CHECK FEATURE CONSISTENCY
# ============================================================

print("\n========== STEP 14: FEATURE CONSISTENCY CHECK ==========")

train_val_columns_match = (
    list(X_train_preprocessed.columns)
    == list(X_val_preprocessed.columns)
)

train_test_columns_match = (
    list(X_train_preprocessed.columns)
    == list(X_test_preprocessed.columns)
)

print("Training and Validation Columns Match:", train_val_columns_match)
print("Training and Testing Columns Match:", train_test_columns_match)

assert train_val_columns_match
assert train_test_columns_match

print("Feature Consistency Check Passed!")


# ============================================================
# STEP 15: CHECK INDEX ALIGNMENT
# ============================================================

print("\n========== STEP 15: INDEX ALIGNMENT CHECK ==========")

train_index_match = (
    X_train_preprocessed.index.equals(y_train.index)
)

val_index_match = (
    X_val_preprocessed.index.equals(y_val.index)
)

test_index_match = (
    X_test_preprocessed.index.equals(y_test.index)
)

print("Training Index Aligned:", train_index_match)
print("Validation Index Aligned:", val_index_match)
print("Testing Index Aligned:", test_index_match)

assert train_index_match
assert val_index_match
assert test_index_match

print("Index Alignment Check Passed!")


# ============================================================
# STEP 16: CHECK MISSING AND INFINITE VALUES
# ============================================================

print("\n========== STEP 16: DATA QUALITY CHECK ==========")

datasets = {
    "Training": X_train_preprocessed,
    "Validation": X_val_preprocessed,
    "Testing": X_test_preprocessed
}

for name, data in datasets.items():

    missing_values = data.isnull().sum().sum()

    infinite_values = np.isinf(
        data.to_numpy()
    ).sum()

    print(f"\n{name} Missing Values:", missing_values)
    print(f"{name} Infinite Values:", infinite_values)

    assert missing_values == 0
    assert infinite_values == 0

print("\nAll Data Quality Checks Passed!")


# ============================================================
# STEP 17: FINAL SUMMARY
# ============================================================

print("\n========== STEP 17: FINAL PREPROCESSING SUMMARY ==========")

print("Original Dataset:", df.shape)

print("\nTraining Data:")
print("Features:", X_train_preprocessed.shape)
print("Target:", y_train.shape)

print("\nValidation Data:")
print("Features:", X_val_preprocessed.shape)
print("Target:", y_val.shape)

print("\nTesting Data:")
print("Features:", X_test_preprocessed.shape)
print("Target:", y_test.shape)

print("\nTotal Final Features:", X_train_preprocessed.shape[1])

print("\n========== PREPROCESSING COMPLETED SUCCESSFULLY ==========")

print("Training, Validation and Testing datasets are ready!")
