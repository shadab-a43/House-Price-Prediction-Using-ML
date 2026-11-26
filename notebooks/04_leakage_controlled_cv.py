
import pandas as pd
import numpy as np

from pathlib import Path

from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.preprocessing_pipeline import create_preprocessing_pipeline


# ============================================================
# STEP 1: LOAD RAW DATA
# ============================================================

print("\n========== STEP 1: LOAD RAW DATA ==========")

DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "AmesHousing.csv"
)

df = pd.read_csv(DATA_PATH)

X = df.drop(
    columns=["SalePrice", "Order", "PID"],
    errors="ignore"
)

y = df["SalePrice"].copy()

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# STEP 2: RECREATE ORIGINAL TRAINING SPLIT
# ============================================================

print("\n========== STEP 2: TRAINING DATA SPLIT ==========")

# Keep the original 20% test set separate.
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Recreate the original training and validation split.
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.20,
    random_state=42
)

print("Training Records:", len(X_train))
print("Validation Records:", len(X_val))
print("Test Records:", len(X_test))


# ============================================================
# STEP 3: CREATE 5-FOLD CROSS-VALIDATION
# ============================================================

print("\n========== STEP 3: 5-FOLD CROSS-VALIDATION ==========")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

fold_results = []


# ============================================================
# STEP 4: TRAIN AND EVALUATE EACH FOLD
# ============================================================

print("\n========== STEP 4: FOLD-WISE TRAINING ==========")

for fold_number, (train_indices, val_indices) in enumerate(
    kf.split(X_train),
    start=1
):

    print(f"\n---------- FOLD {fold_number} ----------")

    # Raw data for this fold.
    X_fold_train = X_train.iloc[train_indices]
    X_fold_val = X_train.iloc[val_indices]

    y_fold_train = y_train.iloc[train_indices]
    y_fold_val = y_train.iloc[val_indices]

    print("Fold Training Shape:", X_fold_train.shape)
    print("Fold Validation Shape:", X_fold_val.shape)

    # Create a fresh preprocessing pipeline for this fold.
    preprocessing = create_preprocessing_pipeline(
        X_fold_train
    )

    # Complete model pipeline.
    model_pipeline = Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # Fit preprocessing and model only on fold training data.
    model_pipeline.fit(
        X_fold_train,
        y_fold_train
    )

    # Predict on fold validation data.
    predictions = model_pipeline.predict(
        X_fold_val
    )

    # Calculate evaluation metrics.
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

    fold_results.append({
        "Fold": fold_number,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2:   {r2:.4f}")


# ============================================================
# STEP 5: CROSS-VALIDATION SUMMARY
# ============================================================

print("\n========== STEP 5: CV RESULTS SUMMARY ==========")

results_df = pd.DataFrame(fold_results)

print("\nFold-wise Results:")
print(results_df.to_string(index=False))

print("\nMean MAE:", results_df["MAE"].mean())
print("Std MAE:", results_df["MAE"].std())

print("\nMean RMSE:", results_df["RMSE"].mean())
print("Std RMSE:", results_df["RMSE"].std())

print("\nMean R2:", results_df["R2"].mean())
print("Std R2:", results_df["R2"].std())


# ============================================================
# STEP 6: FINAL VERIFICATION
# ============================================================

print("\n========== STEP 6: FINAL VERIFICATION ==========")

assert len(fold_results) == 5
assert results_df[["MAE", "RMSE", "R2"]].notna().all().all()

print("Number of Folds:", len(fold_results))
print("All Fold Results Available:", True)
print("Untouched Test Records:", len(X_test))
print("Test Data Used During CV: False")

print("\n========== LEAKAGE-CONTROLLED CV COMPLETED ==========")
