
import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.ensemble import RandomForestRegressor

from src.data_preprocessing import (
    X_train_raw,
    X_val_raw,
    X_train_preprocessed,
    X_val_preprocessed,
    y_train,
    y_val
)

from src.preprocessing_pipeline import create_preprocessing_pipeline


# ============================================================
# PHASE 4: FEATURE SELECTION
# ============================================================

print("\n========== PHASE 4: FEATURE SELECTION ==========")


# ============================================================
# STEP 1: DATA OVERVIEW
# ============================================================

print("\n========== STEP 1: DATA OVERVIEW ==========")

print("\nRaw Training Data:", X_train_raw.shape)
print("Raw Validation Data:", X_val_raw.shape)

print("\nPreprocessed Training Data:", X_train_preprocessed.shape)
print("Preprocessed Validation Data:", X_val_preprocessed.shape)

print("\nTraining Target:", y_train.shape)
print("Validation Target:", y_val.shape)

assert X_train_raw.index.equals(y_train.index)
assert X_val_raw.index.equals(y_val.index)

assert X_train_preprocessed.index.equals(y_train.index)
assert X_val_preprocessed.index.equals(y_val.index)

print("\nTraining and Validation Alignment Passed!")


# ============================================================
# STEP 2: CONSTANT FEATURE DETECTION
# ============================================================

print("\n========== STEP 2: CONSTANT FEATURE DETECTION ==========")

constant_features = X_train_preprocessed.columns[
    X_train_preprocessed.nunique(dropna=False) <= 1
].tolist()

print("\nTotal Features:", X_train_preprocessed.shape[1])
print("Constant Features:", len(constant_features))

if constant_features:

    print("\nConstant Feature Names:")

    for feature in constant_features:
        print(feature)

else:
    print("\nNo constant features found!")

print("\nConstant Feature Detection Completed!")


# ============================================================
# STEP 3: LOW VARIANCE ANALYSIS
# ============================================================

print("\n========== STEP 3: LOW VARIANCE ANALYSIS ==========")

variance_threshold = 0.01

exploratory_variance_selector = VarianceThreshold(
    threshold=variance_threshold
)

exploratory_variance_selector.fit(X_train_preprocessed)

variance_features = X_train_preprocessed.columns[
    exploratory_variance_selector.get_support()
].tolist()

low_variance_features = X_train_preprocessed.columns[
    ~exploratory_variance_selector.get_support()
].tolist()

print("\nVariance Threshold:", variance_threshold)
print("Original Features:", X_train_preprocessed.shape[1])
print("Retained Features:", len(variance_features))
print("Removed Features:", len(low_variance_features))

if low_variance_features:

    print("\nLow Variance Features:")

    for feature in low_variance_features:
        print(feature)

print("\nLow Variance Analysis Completed!")


# ============================================================
# STEP 4: HIGHLY CORRELATED FEATURE ANALYSIS
# ============================================================

print("\n========== STEP 4: CORRELATION ANALYSIS ==========")

numerical_columns = [
    col for col in X_train_preprocessed.columns
    if col.startswith("numerical__")
]

X_train_numerical = X_train_preprocessed[numerical_columns]

correlation_matrix = X_train_numerical.corr().abs()

upper_triangle = correlation_matrix.where(
    np.triu(
        np.ones(correlation_matrix.shape),
        k=1
    ).astype(bool)
)

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
    "\nHighly Correlated Pairs:",
    len(high_correlation_pairs)
)

if not high_correlation_pairs.empty:

    print("\nHighly Correlated Features:")

    print(high_correlation_pairs.to_string(index=False))

else:
    print("\nNo highly correlated pairs found!")

print("\nCorrelation Analysis Completed!")


# ============================================================
# STEP 5: EXPLORATORY RANDOM FOREST FEATURE IMPORTANCE
# ============================================================

print("\n========== STEP 5: RANDOM FOREST FEATURE IMPORTANCE ==========")

exploratory_rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

exploratory_rf_model.fit(
    X_train_preprocessed,
    y_train
)

feature_importance = pd.DataFrame({
    "Feature": X_train_preprocessed.columns,
    "Importance": exploratory_rf_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
).reset_index(drop=True)

print("\nTotal Features Analyzed:", len(feature_importance))

print("\nTop 20 Important Features:")

print(
    feature_importance.head(20).to_string(index=False)
)

print("\nExploratory Feature Importance Completed!")


# ============================================================
# STEP 6: LEAKAGE-CONTROLLED 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n========== STEP 6: LEAKAGE-CONTROLLED CROSS-VALIDATION ==========")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Three feature-selection strategies.
feature_group_names = [
    "All Features",
    "Variance Selected",
    "Top 30"
]

cv_results = {
    group: {
        "MAE": [],
        "RMSE": [],
        "R2": [],
        "Feature_Count": []
    }
    for group in feature_group_names
}


# ------------------------------------------------------------
# HELPER FUNCTION: EVALUATE RANDOM FOREST
# ------------------------------------------------------------

def evaluate_model(X_fold_train, X_fold_val, y_fold_train, y_fold_val):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_fold_train,
        y_fold_train
    )

    predictions = model.predict(X_fold_val)

    mae = mean_absolute_error(
        y_fold_val,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_fold_val,
            predictions
        )
    )

    r2 = r2_score(
        y_fold_val,
        predictions
    )

    return mae, rmse, r2


# ------------------------------------------------------------
# RUN CROSS-VALIDATION
# ------------------------------------------------------------

for fold_number, (train_indices, val_indices) in enumerate(
    kf.split(X_train_raw),
    start=1
):

    print(f"\n========== FOLD {fold_number} ==========")

    # Split RAW data.
    X_fold_train_raw = X_train_raw.iloc[train_indices]
    X_fold_val_raw = X_train_raw.iloc[val_indices]

    y_fold_train = y_train.iloc[train_indices]
    y_fold_val = y_train.iloc[val_indices]

    print("Fold Training Shape:", X_fold_train_raw.shape)
    print("Fold Validation Shape:", X_fold_val_raw.shape)

    # --------------------------------------------------------
    # 1. FRESH PREPROCESSING FOR EACH FOLD
    # --------------------------------------------------------

    fold_preprocessor = create_preprocessing_pipeline(
        X_fold_train_raw
    )

    X_fold_train_array = fold_preprocessor.fit_transform(
        X_fold_train_raw
    )

    X_fold_val_array = fold_preprocessor.transform(
        X_fold_val_raw
    )

    column_transformer = fold_preprocessor.named_steps[
        "column_transformer"
    ]

    fold_feature_names = column_transformer.get_feature_names_out()

    X_fold_train = pd.DataFrame(
        X_fold_train_array,
        columns=fold_feature_names,
        index=X_fold_train_raw.index
    )

    X_fold_val = pd.DataFrame(
        X_fold_val_array,
        columns=fold_feature_names,
        index=X_fold_val_raw.index
    )

    print("Preprocessing Completed for Fold:", fold_number)

    # --------------------------------------------------------
    # GROUP A: ALL FEATURES
    # --------------------------------------------------------

    all_mae, all_rmse, all_r2 = evaluate_model(
        X_fold_train,
        X_fold_val,
        y_fold_train,
        y_fold_val
    )

    cv_results["All Features"]["MAE"].append(all_mae)
    cv_results["All Features"]["RMSE"].append(all_rmse)
    cv_results["All Features"]["R2"].append(all_r2)
    cv_results["All Features"]["Feature_Count"].append(
        X_fold_train.shape[1]
    )

    print("\nAll Features:")
    print(f"MAE: {all_mae:.2f}")
    print(f"RMSE: {all_rmse:.2f}")
    print(f"R2: {all_r2:.4f}")

    # --------------------------------------------------------
    # GROUP B: VARIANCE SELECTED FEATURES
    # --------------------------------------------------------

    fold_variance_selector = VarianceThreshold(
        threshold=variance_threshold
    )

    X_fold_train_variance_array = fold_variance_selector.fit_transform(
        X_fold_train
    )

    X_fold_val_variance_array = fold_variance_selector.transform(
        X_fold_val
    )

    variance_support = fold_variance_selector.get_support()

    X_fold_train_variance = pd.DataFrame(
        X_fold_train_variance_array,
        columns=X_fold_train.columns[variance_support],
        index=X_fold_train.index
    )

    X_fold_val_variance = pd.DataFrame(
        X_fold_val_variance_array,
        columns=X_fold_val.columns[variance_support],
        index=X_fold_val.index
    )

    variance_mae, variance_rmse, variance_r2 = evaluate_model(
        X_fold_train_variance,
        X_fold_val_variance,
        y_fold_train,
        y_fold_val
    )

    cv_results["Variance Selected"]["MAE"].append(variance_mae)
    cv_results["Variance Selected"]["RMSE"].append(variance_rmse)
    cv_results["Variance Selected"]["R2"].append(variance_r2)
    cv_results["Variance Selected"]["Feature_Count"].append(
        X_fold_train_variance.shape[1]
    )

    print("\nVariance Selected:")
    print("Selected Features:", X_fold_train_variance.shape[1])
    print(f"MAE: {variance_mae:.2f}")
    print(f"RMSE: {variance_rmse:.2f}")
    print(f"R2: {variance_r2:.4f}")

    # --------------------------------------------------------
    # GROUP C: TOP 30 FEATURES
    # --------------------------------------------------------

    # Feature importance is learned ONLY from fold training data.
    fold_importance_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    fold_importance_model.fit(
        X_fold_train,
        y_fold_train
    )

    # Select the 30 most important features.
    top_feature_count = min(30, X_fold_train.shape[1])

    top_feature_indices = np.argsort(
        fold_importance_model.feature_importances_
    )[::-1][:top_feature_count]

    top_feature_names = X_fold_train.columns[
        top_feature_indices
    ].tolist()

    X_fold_train_top30 = X_fold_train[top_feature_names].copy()
    X_fold_val_top30 = X_fold_val[top_feature_names].copy()

    top30_mae, top30_rmse, top30_r2 = evaluate_model(
        X_fold_train_top30,
        X_fold_val_top30,
        y_fold_train,
        y_fold_val
    )

    cv_results["Top 30"]["MAE"].append(top30_mae)
    cv_results["Top 30"]["RMSE"].append(top30_rmse)
    cv_results["Top 30"]["R2"].append(top30_r2)
    cv_results["Top 30"]["Feature_Count"].append(
        X_fold_train_top30.shape[1]
    )

    print("\nTop 30 Features:")
    print("Selected Features:", X_fold_train_top30.shape[1])
    print(f"MAE: {top30_mae:.2f}")
    print(f"RMSE: {top30_rmse:.2f}")
    print(f"R2: {top30_r2:.4f}")


# ============================================================
# STEP 7: CROSS-VALIDATION COMPARISON
# ============================================================

print("\n========== STEP 7: 5-FOLD CV COMPARISON ==========")

cv_comparison_results = []

for group_name in feature_group_names:

    results = cv_results[group_name]

    cv_comparison_results.append({

        "Feature Group": group_name,

        "Mean Feature Count": np.mean(
            results["Feature_Count"]
        ),

        "Mean MAE": np.mean(
            results["MAE"]
        ),

        "Std MAE": np.std(
            results["MAE"]
        ),

        "Mean RMSE": np.mean(
            results["RMSE"]
        ),

        "Std RMSE": np.std(
            results["RMSE"]
        ),

        "Mean R2": np.mean(
            results["R2"]
        ),

        "Std R2": np.std(
            results["R2"]
        )
    })


cv_comparison_df = pd.DataFrame(cv_comparison_results)

print("\nFinal CV Results:")

print(
    cv_comparison_df.to_string(index=False)
)

print("\n5-Fold Cross-Validation Completed!")


# ============================================================
# STEP 8: SELECT CANDIDATE STRATEGY USING CV
# ============================================================

print("\n========== STEP 8: FEATURE SELECTION DECISION ==========")

# Lowest mean MAE is the primary selection criterion.
# Higher mean R2 is the tie-breaker.

best_cv_result = cv_comparison_df.sort_values(
    by=["Mean MAE", "Mean R2"],
    ascending=[True, False]
).iloc[0]

selected_strategy = best_cv_result["Feature Group"]

print("\nSelected Strategy:", selected_strategy)

print("Mean CV MAE:", best_cv_result["Mean MAE"])
print("Mean CV RMSE:", best_cv_result["Mean RMSE"])
print("Mean CV R2:", best_cv_result["Mean R2"])

print("\nStrategy Selected Using Training CV Results Only!")


# ============================================================
# STEP 9: FINAL FIT AND VALIDATION VERIFICATION
# ============================================================

print("\n========== STEP 9: VALIDATION VERIFICATION ==========")

# Fit preprocessing on the complete training split.
final_preprocessor = create_preprocessing_pipeline(
    X_train_raw
)

X_train_final_array = final_preprocessor.fit_transform(
    X_train_raw
)

X_val_final_array = final_preprocessor.transform(
    X_val_raw
)

final_column_transformer = final_preprocessor.named_steps[
    "column_transformer"
]

final_feature_names = final_column_transformer.get_feature_names_out()

X_train_final = pd.DataFrame(
    X_train_final_array,
    columns=final_feature_names,
    index=X_train_raw.index
)

X_val_final = pd.DataFrame(
    X_val_final_array,
    columns=final_feature_names,
    index=X_val_raw.index
)


# ------------------------------------------------------------
# APPLY THE SELECTED STRATEGY
# ------------------------------------------------------------

if selected_strategy == "All Features":

    final_selected_features = X_train_final.columns.tolist()

elif selected_strategy == "Variance Selected":

    final_variance_selector = VarianceThreshold(
        threshold=variance_threshold
    )

    final_variance_selector.fit(X_train_final)

    final_selected_features = X_train_final.columns[
        final_variance_selector.get_support()
    ].tolist()

elif selected_strategy == "Top 30":

    final_importance_model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    final_importance_model.fit(
        X_train_final,
        y_train
    )

    final_top_count = min(30, X_train_final.shape[1])

    final_top_indices = np.argsort(
        final_importance_model.feature_importances_
    )[::-1][:final_top_count]

    final_selected_features = X_train_final.columns[
        final_top_indices
    ].tolist()


# Create final selected datasets.
X_train_selected = X_train_final[
    final_selected_features
].copy()

X_val_selected = X_val_final[
    final_selected_features
].copy()


print("\nSelected Strategy:", selected_strategy)

print("Final Selected Feature Count:", len(final_selected_features))

print("Training Shape:", X_train_selected.shape)

print("Validation Shape:", X_val_selected.shape)


# ------------------------------------------------------------
# FINAL RANDOM FOREST VALIDATION
# ------------------------------------------------------------

final_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X_train_selected,
    y_train
)

validation_predictions = final_model.predict(
    X_val_selected
)

validation_mae = mean_absolute_error(
    y_val,
    validation_predictions
)

validation_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        validation_predictions
    )
)

validation_r2 = r2_score(
    y_val,
    validation_predictions
)

print("\nValidation Results:")

print(f"MAE: {validation_mae:.2f}")
print(f"RMSE: {validation_rmse:.2f}")
print(f"R2: {validation_r2:.4f}")

print("\nValidation Evaluation Completed!")


# ============================================================
# STEP 10: FINAL SELECTED FEATURE REVIEW
# ============================================================

print("\n========== STEP 10: FINAL SELECTED FEATURE REVIEW ==========")

print("\nFinal Selected Features:")

for index, feature in enumerate(
    final_selected_features,
    start=1
):
    print(f"{index}. {feature}")


final_numerical_count = sum(
    feature.startswith("numerical__")
    for feature in final_selected_features
)

final_categorical_count = sum(
    feature.startswith("categorical__")
    for feature in final_selected_features
)

print("\nFeature Type Summary:")

print("Numerical Features:", final_numerical_count)
print("Categorical Features:", final_categorical_count)
print("Total Features:", len(final_selected_features))


# ============================================================
# STEP 11: FINAL FEATURE SELECTION VERIFICATION
# ============================================================

print("\n========== STEP 11: FINAL VERIFICATION ==========")

duplicate_count = (
    len(final_selected_features)
    - len(set(final_selected_features))
)

missing_features = [
    feature
    for feature in final_selected_features
    if feature not in X_train_final.columns
]

train_columns_match = (
    list(X_train_selected.columns)
    == final_selected_features
)

val_columns_match = (
    list(X_val_selected.columns)
    == final_selected_features
)

train_target_aligned = X_train_selected.index.equals(
    y_train.index
)

val_target_aligned = X_val_selected.index.equals(
    y_val.index
)

print("\nSelected Feature Count:", len(final_selected_features))
print("Duplicate Features:", duplicate_count)
print("Missing Features:", len(missing_features))

print("Training Columns Match:", train_columns_match)
print("Validation Columns Match:", val_columns_match)

print("Training Target Aligned:", train_target_aligned)
print("Validation Target Aligned:", val_target_aligned)

verification_passed = (
    len(final_selected_features) > 0
    and duplicate_count == 0
    and len(missing_features) == 0
    and train_columns_match
    and val_columns_match
    and train_target_aligned
    and val_target_aligned
)

print("\n========== FINAL VERIFICATION RESULT ==========")

if verification_passed:

    print("ALL FEATURE SELECTION CHECKS PASSED!")

    print("Selected Strategy:", selected_strategy)

    print("Final Selected Features:", len(final_selected_features))

else:

    print("SOME CHECKS FAILED. REVIEW THE RESULTS!")


print("\n========== PHASE 4 FEATURE SELECTION ANALYSIS FINISHED ==========")
