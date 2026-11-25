
import pandas as pd
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score


# ==========================================
# PHASE 4: FEATURE SELECTION
# ==========================================

from src.data_preprocessing import (
    X_train_preprocessed,
    X_val_preprocessed,
    y_train,
    y_val
)


print("\n========== PHASE 4: FEATURE SELECTION ==========")


# ==========================================
# STEP 1: DATA OVERVIEW
# ==========================================

print("\n========== STEP 1: DATA OVERVIEW ==========")

print("\nTraining Data Shape:")
print(X_train_preprocessed.shape)

print("\nValidation Data Shape:")
print(X_val_preprocessed.shape)

print("\nTraining Target Shape:")
print(y_train.shape)

print("\nValidation Target Shape:")
print(y_val.shape)

print("\nTotal Features:")
print(X_train_preprocessed.shape[1])

print("\nFirst 20 Feature Names:")
print(X_train_preprocessed.columns[:20].tolist())

# Ensure feature columns match.
assert list(X_train_preprocessed.columns) == list(
    X_val_preprocessed.columns
)

assert X_train_preprocessed.index.equals(y_train.index)
assert X_val_preprocessed.index.equals(y_val.index)

print("\nTraining and Validation Data Alignment Passed!")


# ==========================================
# STEP 2: CONSTANT FEATURE DETECTION
# ==========================================

print("\n========== STEP 2: CONSTANT FEATURE DETECTION ==========")

# Check only training data.
constant_features = X_train_preprocessed.columns[
    X_train_preprocessed.nunique(dropna=False) <= 1
].tolist()

print("\nTotal Constant Features:", len(constant_features))

if constant_features:

    print("\nConstant Features:")

    for feature in constant_features:
        print(feature)

else:
    print("\nNo constant features found!")

print("\nConstant Feature Detection Completed!")


# ==========================================
# STEP 3: LOW VARIANCE DETECTION
# ==========================================

print("\n========== STEP 3: LOW VARIANCE DETECTION ==========")

# Fit variance selector using training data only.
variance_selector = VarianceThreshold(threshold=0.01)

variance_selector.fit(X_train_preprocessed)

# Features retained by variance threshold.
selected_features = X_train_preprocessed.columns[
    variance_selector.get_support()
].tolist()

# Features removed by variance threshold.
low_variance_features = X_train_preprocessed.columns[
    ~variance_selector.get_support()
].tolist()

print("\nVariance Threshold: 0.01")

print(
    "\nTotal Original Features:",
    X_train_preprocessed.shape[1]
)

print("Features Above Threshold:", len(selected_features))

print("Low Variance Features:", len(low_variance_features))

if low_variance_features:

    print("\nLow Variance Feature Names:")

    for feature in low_variance_features:
        print(feature)

else:
    print("\nNo low variance features found!")

print("\nLow Variance Analysis Completed!")


# ==========================================
# STEP 4: HIGHLY CORRELATED FEATURE DETECTION
# ==========================================

print("\n========== STEP 4: CORRELATION ANALYSIS ==========")

# Select numerical features only.
numerical_columns = [
    col for col in X_train_preprocessed.columns
    if col.startswith("numerical__")
]

X_train_numerical = X_train_preprocessed[numerical_columns]

# Calculate correlation using training data only.
correlation_matrix = X_train_numerical.corr().abs()

# Get upper triangle to avoid duplicate pairs.
upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

# Find pairs with correlation above 0.90.
correlation_threshold = 0.90

high_correlation_pairs = (
    upper_triangle.stack()
    .reset_index()
)

high_correlation_pairs.columns = [
    "Feature 1",
    "Feature 2",
    "Correlation"
]

high_correlation_pairs = high_correlation_pairs[
    high_correlation_pairs["Correlation"] > correlation_threshold
]

high_correlation_pairs = high_correlation_pairs.sort_values(
    by="Correlation",
    ascending=False
)

print("\nNumerical Features Checked:", len(numerical_columns))

print("Correlation Threshold:", correlation_threshold)

print(
    "\nHighly Correlated Feature Pairs:",
    len(high_correlation_pairs)
)

if not high_correlation_pairs.empty:

    print("\nHighly Correlated Pairs:")

    print(
        high_correlation_pairs.to_string(index=False)
    )

else:

    print("\nNo highly correlated pairs found!")

print("\nCorrelation Analysis Completed!")


# ==========================================
# STEP 5: RANDOM FOREST FEATURE IMPORTANCE
# ==========================================

print("\n========== STEP 5: RANDOM FOREST FEATURE IMPORTANCE ==========")

# Create Random Forest model.
rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train using training data only.
rf_model.fit(
    X_train_preprocessed,
    y_train
)

# Calculate feature importance.
feature_importance = pd.DataFrame({
    "Feature": X_train_preprocessed.columns,
    "Importance": rf_model.feature_importances_
})

# Sort features by importance.
feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nTotal Features Analyzed:", len(feature_importance))

print("\nTop 20 Important Features:")

print(
    feature_importance.head(20).to_string(index=False)
)

print("\nRandom Forest Feature Importance Completed!")


# ==========================================
# STEP 6: USE EXISTING VALIDATION DATA
# ==========================================

print("\n========== STEP 6: VALIDATION DATASET ==========")

# Validation data was already created in data_preprocessing.py.
# Do not split training data again.

X_train_fs = X_train_preprocessed.copy()
X_val_fs = X_val_preprocessed.copy()

y_train_fs = y_train.copy()
y_val_fs = y_val.copy()

print("\nTraining Features Shape:", X_train_fs.shape)

print("Validation Features Shape:", X_val_fs.shape)

print("\nTraining Target Shape:", y_train_fs.shape)

print("Validation Target Shape:", y_val_fs.shape)

print("\nExisting Validation Dataset Loaded Successfully!")


# ==========================================
# STEP 6.2: PERMUTATION FEATURE IMPORTANCE
# ==========================================

print("\n========== STEP 6.2: PERMUTATION IMPORTANCE ==========")

# Train Random Forest on training data.
rf_perm_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

rf_perm_model.fit(
    X_train_fs,
    y_train_fs
)

# Calculate permutation importance on actual validation data.
perm_results = permutation_importance(
    rf_perm_model,
    X_val_fs,
    y_val_fs,
    n_repeats=5,
    random_state=42,
    scoring="neg_mean_absolute_error",
    n_jobs=1
)

# Create importance DataFrame.
permutation_df = pd.DataFrame({
    "Feature": X_train_fs.columns,
    "Importance_Mean": perm_results.importances_mean,
    "Importance_Std": perm_results.importances_std
})

permutation_df = permutation_df.sort_values(
    by="Importance_Mean",
    ascending=False
).reset_index(drop=True)

print("\nTop 20 Permutation Important Features:")

print(
    permutation_df.head(20).to_string(index=False)
)

print("\nPermutation Feature Importance Completed!")


# ==========================================
# STEP 7.1: FEATURE SELECTION CANDIDATES
# ==========================================

print("\n========== STEP 7.1: FEATURE SELECTION CANDIDATES ==========")

# Select top 30 features using permutation importance.
top_30_features = permutation_df.head(30)["Feature"].tolist()

print(
    "\nTotal Original Features:",
    X_train_preprocessed.shape[1]
)

print("Selected Candidate Features:", len(top_30_features))

print("\nTop 30 Candidate Features:")

for index, feature in enumerate(top_30_features, start=1):
    print(f"{index}. {feature}")

print("\nFeature Selection Candidates Created!")


# ==========================================
# STEP 7.2: FEATURE GROUPS
# ==========================================

print("\n========== STEP 7.2: FEATURE GROUPS ==========")

# Group A: All original features.
X_train_all = X_train_fs.copy()
X_val_all = X_val_fs.copy()

# Group B: Top 30 permutation important features.
X_train_top30 = X_train_fs[top_30_features].copy()
X_val_top30 = X_val_fs[top_30_features].copy()

# Group C: Features retained by variance threshold.
variance_features = selected_features

X_train_variance = X_train_fs[variance_features].copy()
X_val_variance = X_val_fs[variance_features].copy()

print("\nGroup A - All Features:", X_train_all.shape)

print("Group B - Top 30 Features:", X_train_top30.shape)

print("Group C - Variance Selected Features:", X_train_variance.shape)

print("\nAll Feature Groups Created Successfully!")


# ==========================================
# STEP 7.3: FEATURE GROUP COMPARISON
# ==========================================

print("\n========== STEP 7.3: FEATURE GROUP COMPARISON ==========")

feature_groups = {
    f"All {X_train_all.shape[1]} Features": (
        X_train_all,
        X_val_all
    ),

    f"Top {len(top_30_features)} Features": (
        X_train_top30,
        X_val_top30
    ),

    f"Variance Selected {len(variance_features)}": (
        X_train_variance,
        X_val_variance
    )
}

comparison_results = []

for group_name, (X_tr, X_val_group) in feature_groups.items():

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    # Train on training data.
    model.fit(
        X_tr,
        y_train_fs
    )

    # Evaluate on validation data.
    predictions = model.predict(X_val_group)

    mae = mean_absolute_error(
        y_val_fs,
        predictions
    )

    r2 = r2_score(
        y_val_fs,
        predictions
    )

    comparison_results.append({
        "Feature Group": group_name,
        "Feature Count": X_tr.shape[1],
        "MAE": mae,
        "R2 Score": r2
    })

comparison_df = pd.DataFrame(comparison_results)

print("\nFeature Group Comparison:")

print(
    comparison_df.to_string(index=False)
)

print("\nFeature Group Comparison Completed!")


# ==========================================
# STEP 7.4: FEATURE SELECTION DECISION
# ==========================================

print("\n========== STEP 7.4: FEATURE SELECTION DECISION ==========")

# Choose the group with lowest MAE.
# If MAE is equal, prefer higher R2.

best_result = comparison_df.sort_values(
    by=["MAE", "R2 Score"],
    ascending=[True, False]
).iloc[0]

selected_group_name = best_result["Feature Group"]

if selected_group_name.startswith("All "):

    final_selected_features = X_train_all.columns.tolist()

elif selected_group_name.startswith("Top "):

    final_selected_features = top_30_features.copy()

else:

    final_selected_features = variance_features.copy()

print("\nSelected Feature Group:", selected_group_name)

print("Total Selected Features:", len(final_selected_features))

print("\nValidation MAE:", best_result["MAE"])

print("Validation R2 Score:", best_result["R2 Score"])

print("\nReason for Selection:")
print("1. Lowest validation MAE among the evaluated groups.")
print("2. R2 Score used as a tie-breaker.")
print("3. Selected feature count:", len(final_selected_features))

print("\nFeature Selection Decision Completed!")


# ==========================================
# STEP 7.5: TOP 30 FEATURE REVIEW
# ==========================================

print("\n========== STEP 7.5: TOP 30 FEATURE REVIEW ==========")

print("\nSelected Top 30 Features:")

for index, feature in enumerate(top_30_features, start=1):
    print(f"{index}. {feature}")

numerical_count = sum(
    feature.startswith("numerical__")
    for feature in top_30_features
)

categorical_count = sum(
    feature.startswith("categorical__")
    for feature in top_30_features
)

print("\nFeature Type Summary:")

print("Numerical Features:", numerical_count)

print("Categorical Features:", categorical_count)

print("Total Features:", len(top_30_features))

print("\nTop 30 Feature Review Completed!")


# ==========================================
# STEP 7.5: PART 2 - IMPORTANCE REVIEW
# ==========================================

print("\n========== STEP 7.5: PART 2 - IMPORTANCE REVIEW ==========")

top_30_review = permutation_df.head(30).copy()

top_30_review["Rank"] = range(1, len(top_30_review) + 1)

top_30_review = top_30_review[
    [
        "Rank",
        "Feature",
        "Importance_Mean",
        "Importance_Std"
    ]
]

print(top_30_review.to_string(index=False))

print("\nImportance Review Completed!")


# ==========================================
# STEP 7.5: PART 3 - BOTTOM 10 FEATURES
# ==========================================

print("\n========== STEP 7.5: PART 3 - BOTTOM 10 FEATURES ==========")

bottom_10_features = permutation_df.tail(10).copy()

print("\nBottom 10 Features:")

print(
    bottom_10_features[
        [
            "Feature",
            "Importance_Mean",
            "Importance_Std"
        ]
    ].to_string(index=False)
)

negative_features = permutation_df[
    permutation_df["Importance_Mean"] < 0
]

print("\nNegative Importance Features:", len(negative_features))

if len(negative_features) > 0:

    print("\nFeatures with Negative Importance:")

    print(
        negative_features[
            ["Feature", "Importance_Mean"]
        ].to_string(index=False)
    )

else:

    print("No negative importance features found.")

print("\nStep 7.5 Part 3 Completed!")


# ==========================================
# STEP 7.6: FINAL FEATURE SELECTION VERIFICATION
# ==========================================

print("\n========== STEP 7.6: FINAL FEATURE SELECTION VERIFICATION ==========")

# 1. Selected feature count.
print("\n1. Selected Feature Count:")

print("Selected Features:", len(final_selected_features))

# 2. Duplicate feature names.
print("\n2. Duplicate Feature Check:")

duplicate_count = (
    len(final_selected_features)
    - len(set(final_selected_features))
)

print("Duplicate Features:", duplicate_count)

# 3. Verify features exist in original training data.
print("\n3. Feature Availability Check:")

missing_features = [
    feature
    for feature in final_selected_features
    if feature not in X_train_preprocessed.columns
]

print("Missing Features:", len(missing_features))

# 4. Verify training and validation alignment.
print("\n4. Training and Validation Alignment:")

X_train_selected = X_train_fs[final_selected_features]

X_val_selected = X_val_fs[final_selected_features]

train_columns_match = (
    list(X_train_selected.columns)
    == final_selected_features
)

val_columns_match = (
    list(X_val_selected.columns)
    == final_selected_features
)

print("Training Columns Match:", train_columns_match)

print("Validation Columns Match:", val_columns_match)

# 5. Verify feature types.
print("\n5. Feature Type Check:")

final_numerical_count = sum(
    feature.startswith("numerical__")
    for feature in final_selected_features
)

final_categorical_count = sum(
    feature.startswith("categorical__")
    for feature in final_selected_features
)

print("Numerical Features:", final_numerical_count)

print("Categorical Features:", final_categorical_count)

print(
    "Total:",
    final_numerical_count + final_categorical_count
)

# 6. Verify target alignment.
print("\n6. Target Alignment Check:")

train_target_aligned = X_train_selected.index.equals(
    y_train_fs.index
)

val_target_aligned = X_val_selected.index.equals(
    y_val_fs.index
)

print("Training Target Aligned:", train_target_aligned)

print("Validation Target Aligned:", val_target_aligned)


# ==========================================
# FINAL VERIFICATION
# ==========================================

verification_passed = (
    len(final_selected_features) > 0
    and duplicate_count == 0
    and len(missing_features) == 0
    and train_columns_match
    and val_columns_match
    and final_numerical_count + final_categorical_count
        == len(final_selected_features)
    and train_target_aligned
    and val_target_aligned
)

print("\n========== FINAL VERIFICATION RESULT ==========")

if verification_passed:

    print("ALL FEATURE SELECTION CHECKS PASSED!")

    print("Selected Feature Group:", selected_group_name)

    print("Final Selected Features:", len(final_selected_features))

else:

    print("SOME CHECKS FAILED. REVIEW THE RESULTS.")

print("\nStep 7.6 Completed!")
