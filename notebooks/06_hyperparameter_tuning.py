
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from src.data_preprocessing import (
    X_train_raw,
    y_train,
    X_val_raw,
    y_val
)

from src.preprocessing_pipeline import create_preprocessing_pipeline


print("Training Data:", X_train_raw.shape)
print("Validation Data:", X_val_raw.shape)

print("\nHyperparameter tuning setup completed!")


# Create preprocessing pipeline
preprocessor = create_preprocessing_pipeline(X_train_raw)

# Combine preprocessing and model
model_pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", RandomForestRegressor(random_state=42, n_jobs=-1))
])

# Define cross-validation
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Define parameters to test
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 20],
    "model__min_samples_split": [2, 5],
    "model__min_samples_leaf": [1, 2]
}

print("\nPipeline Created Successfully!")
print("Total Parameter Combinations:", 16)
print("Cross Validation Folds:", cv.get_n_splits())



# Create Grid Search
grid_search = GridSearchCV(
    estimator=model_pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1,
    verbose=2
)

print("\n========== HYPERPARAMETER TUNING STARTED ==========")

# Train and search
grid_search.fit(X_train_raw, y_train)

print("\n========== TUNING COMPLETED ==========")

print("\nBest Parameters:")
print(grid_search.best_params_)

print("\nBest Cross Validation MAE:")
print(round(-grid_search.best_score_, 2))



from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Get best tuned model
best_model = grid_search.best_estimator_

# Predict validation data
tuned_predictions = best_model.predict(X_val_raw)

# Calculate metrics
tuned_mae = mean_absolute_error(y_val, tuned_predictions)
tuned_rmse = np.sqrt(mean_squared_error(y_val, tuned_predictions))
tuned_r2 = r2_score(y_val, tuned_predictions)

print("\n========== TUNED MODEL VALIDATION RESULTS ==========")
print("MAE:", round(tuned_mae, 2))
print("RMSE:", round(tuned_rmse, 2))
print("R2 Score:", round(tuned_r2, 4))

# Compare with original Random Forest
print("\n========== ORIGINAL VS TUNED RANDOM FOREST ==========")

print("Original Random Forest MAE: 14433.51")
print("Tuned Random Forest MAE:", round(tuned_mae, 2))



import pandas as pd

tuning_results = pd.DataFrame({
    "Model": ["Original Random Forest", "Tuned Random Forest"],
    "MAE": [14433.514925, tuned_mae],
    "RMSE": [21163.514574, tuned_rmse],
    "R2 Score": [0.917665, tuned_r2]
})

print("\n========== TUNING COMPARISON ==========")
print(tuning_results.to_string(index=False))

tuning_results.to_csv(
    "models/hyperparameter_tuning_results.csv",
    index=False
)

print("\nTuning results saved successfully!")
