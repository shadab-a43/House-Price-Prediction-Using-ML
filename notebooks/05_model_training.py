import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.data_preprocessing import (
    X_train_preprocessed,
    y_train,
    X_val_preprocessed,
    y_val
)

# Check training data
print("Training Data Shape:", X_train_preprocessed.shape)
print("Target Shape:", y_train.shape)

# Model 1: Linear Regression
linear_model = LinearRegression()

# Model 2: Random Forest Regressor
random_forest_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

# Train both models
print("\nTraining Linear Regression...")
linear_model.fit(X_train_preprocessed, y_train)
print("Linear Regression Training Completed!")

print("\nTraining Random Forest...")
random_forest_model.fit(X_train_preprocessed, y_train)
print("Random Forest Training Completed!")

print("\nBoth models trained successfully!")


from src.data_preprocessing import X_val_preprocessed

# Linear Regression predictions
linear_predictions = linear_model.predict(X_val_preprocessed)

# Random Forest predictions
random_forest_predictions = random_forest_model.predict(
    X_val_preprocessed
)

print("\n========== VALIDATION PREDICTIONS ==========")

print("Actual Prices:")
print(y_val.head(5).values)

print("\nLinear Regression Predictions:")
print(linear_predictions[:5])

print("\nRandom Forest Predictions:")
print(random_forest_predictions[:5])

print("\nPrediction Shapes:")
print("Linear Regression:", linear_predictions.shape)
print("Random Forest:", random_forest_predictions.shape)


# Calculate Linear Regression metrics
linear_mae = mean_absolute_error(y_val, linear_predictions)
linear_rmse = np.sqrt(mean_squared_error(y_val, linear_predictions))
linear_r2 = r2_score(y_val, linear_predictions)

# Calculate Random Forest metrics
rf_mae = mean_absolute_error(y_val, random_forest_predictions)
rf_rmse = np.sqrt(mean_squared_error(y_val, random_forest_predictions))
rf_r2 = r2_score(y_val, random_forest_predictions)

# Display comparison
print("\n========== MODEL COMPARISON ==========")

print("\nLinear Regression:")
print("MAE:", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R2 Score:", round(linear_r2, 4))

print("\nRandom Forest:")
print("MAE:", round(rf_mae, 2))
print("RMSE:", round(rf_rmse, 2))
print("R2 Score:", round(rf_r2, 4))


import pandas as pd

model_results = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [linear_mae, rf_mae],
    "RMSE": [linear_rmse, rf_rmse],
    "R2 Score": [linear_r2, rf_r2]
})

print("\n========== FINAL RESULTS TABLE ==========")
print(model_results.to_string(index=False))

# Save results
results_path = "models/model_comparison.csv"
model_results.to_csv(results_path, index=False)

print("\nResults saved to:", results_path)


import matplotlib.pyplot as plt

# Actual prices
actual_prices = y_val

# Find common graph limits
min_price = min(
    actual_prices.min(),
    linear_predictions.min(),
    random_forest_predictions.min()
)

max_price = max(
    actual_prices.max(),
    linear_predictions.max(),
    random_forest_predictions.max()
)

# Create two graphs
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Linear Regression
axes[0].scatter(actual_prices, linear_predictions, alpha=0.5)
axes[0].plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)
axes[0].set_title("Linear Regression")
axes[0].set_xlabel("Actual Sale Price")
axes[0].set_ylabel("Predicted Sale Price")

# Random Forest
axes[1].scatter(actual_prices, random_forest_predictions, alpha=0.5)
axes[1].plot(
    [min_price, max_price],
    [min_price, max_price],
    linestyle="--"
)
axes[1].set_title("Random Forest")
axes[1].set_xlabel("Actual Sale Price")
axes[1].set_ylabel("Predicted Sale Price")

plt.suptitle("Actual vs Predicted House Prices")
plt.tight_layout()
plt.show()


# Calculate residuals
linear_residuals = y_val - linear_predictions
rf_residuals = y_val - random_forest_predictions

# Create residual plots
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Linear Regression
axes[0].scatter(linear_predictions, linear_residuals, alpha=0.5)
axes[0].axhline(y=0, linestyle="--")
axes[0].set_title("Linear Regression Residuals")
axes[0].set_xlabel("Predicted Price")
axes[0].set_ylabel("Residuals")

# Random Forest
axes[1].scatter(random_forest_predictions, rf_residuals, alpha=0.5)
axes[1].axhline(y=0, linestyle="--")
axes[1].set_title("Random Forest Residuals")
axes[1].set_xlabel("Predicted Price")
axes[1].set_ylabel("Residuals")

plt.tight_layout()
plt.show()


# Plot residual distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(linear_residuals, bins=30, edgecolor="black")
axes[0].set_title("Linear Regression Residual Distribution")
axes[0].set_xlabel("Residual")
axes[0].set_ylabel("Frequency")

axes[1].hist(rf_residuals, bins=30, edgecolor="black")
axes[1].set_title("Random Forest Residual Distribution")
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")

plt.tight_layout()
plt.show()

print("\n========== RESIDUAL SUMMARY ==========")
print("Linear Regression Mean Residual:", round(linear_residuals.mean(), 2))
print("Random Forest Mean Residual:", round(rf_residuals.mean(), 2))
