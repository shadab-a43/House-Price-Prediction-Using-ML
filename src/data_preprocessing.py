
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split

try:
    from .feature_engineering import add_engineered_features
    from .preprocessing_pipeline import create_preprocessing_pipeline
except ImportError:
    from feature_engineering import add_engineered_features
    from preprocessing_pipeline import create_preprocessing_pipeline


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

X_train_val, X_test_raw, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

X_train_raw, X_val_raw, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42
)

print("Training Records:", X_train_raw.shape[0])
print("Validation Records:", X_val_raw.shape[0])
print("Testing Records:", X_test_raw.shape[0])

print("Training Target:", y_train.shape)
print("Validation Target:", y_val.shape)
print("Testing Target:", y_test.shape)


# ============================================================
# STEP 5: FEATURE ENGINEERING COMPATIBILITY
# ============================================================

print("\n========== STEP 5: FEATURE ENGINEERING ==========")

# Keep these variables for compatibility with the previous workflow.
# Actual pipeline will perform feature engineering internally.

X_train = add_engineered_features(X_train_raw)
X_val = add_engineered_features(X_val_raw)
X_test = add_engineered_features(X_test_raw)

print("Training Shape:", X_train.shape)
print("Validation Shape:", X_val.shape)
print("Testing Shape:", X_test.shape)

assert list(X_train.columns) == list(X_val.columns)
assert list(X_train.columns) == list(X_test.columns)

print("Feature Columns Match: True")


# ============================================================
# STEP 6: CREATE REUSABLE PREPROCESSING PIPELINE
# ============================================================

print("\n========== STEP 6: CREATE PREPROCESSING PIPELINE ==========")

preprocessor = create_preprocessing_pipeline(X_train_raw)

print(preprocessor)
print("\nReusable Preprocessing Pipeline Created!")


# ============================================================
# STEP 7: FIT AND TRANSFORM DATA
# ============================================================

print("\n========== STEP 7: FIT AND TRANSFORM DATA ==========")

# Fit the complete pipeline ONLY on training data.
X_train_array = preprocessor.fit_transform(X_train_raw)

# Validation and test data are only transformed.
X_val_array = preprocessor.transform(X_val_raw)
X_test_array = preprocessor.transform(X_test_raw)

# Get encoded feature names from the final ColumnTransformer.
column_transformer = preprocessor.named_steps["column_transformer"]
feature_names = column_transformer.get_feature_names_out()

# Convert NumPy arrays into DataFrames.
X_train_preprocessed = pd.DataFrame(
    X_train_array,
    columns=feature_names,
    index=X_train_raw.index
)

X_val_preprocessed = pd.DataFrame(
    X_val_array,
    columns=feature_names,
    index=X_val_raw.index
)

X_test_preprocessed = pd.DataFrame(
    X_test_array,
    columns=feature_names,
    index=X_test_raw.index
)

print("Training Preprocessed Shape:", X_train_preprocessed.shape)
print("Validation Preprocessed Shape:", X_val_preprocessed.shape)
print("Testing Preprocessed Shape:", X_test_preprocessed.shape)

print("Preprocessing Completed!")


# ============================================================
# STEP 8: FEATURE CONSISTENCY CHECK
# ============================================================

print("\n========== STEP 8: FEATURE CONSISTENCY CHECK ==========")

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
# STEP 9: INDEX ALIGNMENT CHECK
# ============================================================

print("\n========== STEP 9: INDEX ALIGNMENT CHECK ==========")

train_index_match = X_train_preprocessed.index.equals(y_train.index)
val_index_match = X_val_preprocessed.index.equals(y_val.index)
test_index_match = X_test_preprocessed.index.equals(y_test.index)

print("Training Index Aligned:", train_index_match)
print("Validation Index Aligned:", val_index_match)
print("Testing Index Aligned:", test_index_match)

assert train_index_match
assert val_index_match
assert test_index_match

print("Index Alignment Check Passed!")


# ============================================================
# STEP 10: DATA QUALITY CHECK
# ============================================================

print("\n========== STEP 10: DATA QUALITY CHECK ==========")

datasets = {
    "Training": X_train_preprocessed,
    "Validation": X_val_preprocessed,
    "Testing": X_test_preprocessed
}

for name, data in datasets.items():

    missing_values = data.isnull().sum().sum()

    infinite_values = np.isinf(data.to_numpy()).sum()

    print(f"\n{name} Missing Values:", missing_values)
    print(f"{name} Infinite Values:", infinite_values)

    assert missing_values == 0
    assert infinite_values == 0

print("\nAll Data Quality Checks Passed!")


# ============================================================
# STEP 11: FINAL PREPROCESSING SUMMARY
# ============================================================

print("\n========== STEP 11: FINAL PREPROCESSING SUMMARY ==========")

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
