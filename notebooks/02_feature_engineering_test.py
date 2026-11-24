
import sys
import numpy as np
from pathlib import Path
import pandas as pd

# Project root ko Python import path mein add karna
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import add_engineered_features


# Load original dataset
df = pd.read_csv("data/AmesHousing.csv")

# Separate input features
X = df.drop(columns=["SalePrice", "Order", "PID"])

# Apply feature engineering
X_new = add_engineered_features(X)


# Display new features
new_features = [
    "TotalSF",
    "TotalBathrooms",
    "HouseAgeAtSale",
    "RemodAgeAtSale",
    "TotalPorchSF",
    "GarageAgeAtSale"
]

print("\nNew Engineered Features:")
print(X_new[new_features].head(10))


# Check dataset shapes
print("\nOriginal Shape:", X.shape)
print("New Shape:", X_new.shape)


# Check missing values
print("\nMissing Values in New Features:")
print(X_new[new_features].isnull().sum())


# Check feature creation
print("\nFeature Creation Check:")
for feature in new_features:
    print(feature, ":", feature in X_new.columns)


print("\n--- Missing Value Investigation ---")

# TotalSF missing records
print("\nTotalSF Missing Records:")
print(
    X_new.loc[
        X_new["TotalSF"].isnull(),
        ["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"]
    ]
)

# TotalBathrooms missing records
print("\nTotalBathrooms Missing Records:")
print(
    X_new.loc[
        X_new["TotalBathrooms"].isnull(),
        [
            "Full Bath",
            "Half Bath",
            "Bsmt Full Bath",
            "Bsmt Half Bath"
        ]
    ]
)

# GarageAgeAtSale missing records
print("\nGarageAgeAtSale Missing Records:")
print(
    X_new.loc[
        X_new["GarageAgeAtSale"].isnull(),
        ["Garage Cars", "Garage Area", "Garage Yr Blt", "Yr Sold"]
    ]
)



print("\n--- Feature Calculation Audit ---")

# Check TotalSF
print("\nTotalSF Calculation (Row 0):")
print(
    X.loc[0, ["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"]].sum(),
    "=",
    X_new.loc[0, "TotalSF"]
)

# Check TotalBathrooms
print("\nTotalBathrooms Calculation (Row 0):")
print(
    X.loc[0, "Full Bath"]
    + 0.5 * X.loc[0, "Half Bath"]
    + X.loc[0, "Bsmt Full Bath"]
    + 0.5 * X.loc[0, "Bsmt Half Bath"],
    "=",
    X_new.loc[0, "TotalBathrooms"]
)

# Check House Age
print("\nHouseAgeAtSale Calculation (Row 0):")
print(
    X.loc[0, "Yr Sold"] - X.loc[0, "Year Built"],
    "=",
    X_new.loc[0, "HouseAgeAtSale"]
)

# Check Remodeling Age
print("\nRemodAgeAtSale Calculation (Row 0):")
print(
    X.loc[0, "Yr Sold"] - X.loc[0, "Year Remod/Add"],
    "=",
    X_new.loc[0, "RemodAgeAtSale"]
)

# Check Porch Area
print("\nTotalPorchSF Calculation (Row 0):")
print(
    X.loc[0, [
        "Open Porch SF",
        "Enclosed Porch",
        "3Ssn Porch",
        "Screen Porch"
    ]].sum(),
    "=",
    X_new.loc[0, "TotalPorchSF"]
)

# Check Garage Age
print("\nGarageAgeAtSale Calculation (Row 0):")
print(
    X.loc[0, "Yr Sold"] - X.loc[0, "Garage Yr Blt"],
    "=",
    X_new.loc[0, "GarageAgeAtSale"]
)


print("\n--- Complete Dataset Validation ---")

# 1. TotalSF
expected_total_sf = X[
    ["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"]
].sum(axis=1, min_count=3)

check_total_sf = np.allclose(
    X_new["TotalSF"],
    expected_total_sf,
    equal_nan=True
)


# 2. TotalBathrooms
bath_data = X.copy()

no_basement = bath_data["Total Bsmt SF"].eq(0)

bath_data.loc[
    no_basement,
    ["Bsmt Full Bath", "Bsmt Half Bath"]
] = bath_data.loc[
    no_basement,
    ["Bsmt Full Bath", "Bsmt Half Bath"]
].fillna(0)

expected_bathrooms = (
    bath_data["Full Bath"]
    + 0.5 * bath_data["Half Bath"]
    + bath_data["Bsmt Full Bath"]
    + 0.5 * bath_data["Bsmt Half Bath"]
)

check_bathrooms = np.allclose(
    X_new["TotalBathrooms"],
    expected_bathrooms,
    equal_nan=True
)


# 3. HouseAgeAtSale
expected_house_age = X["Yr Sold"] - X["Year Built"]

check_house_age = np.allclose(
    X_new["HouseAgeAtSale"],
    expected_house_age,
    equal_nan=True
)


# 4. RemodAgeAtSale
expected_remod_age = X["Yr Sold"] - X["Year Remod/Add"]

check_remod_age = np.allclose(
    X_new["RemodAgeAtSale"],
    expected_remod_age,
    equal_nan=True
)


# 5. TotalPorchSF
porch_columns = [
    "Open Porch SF",
    "Enclosed Porch",
    "3Ssn Porch",
    "Screen Porch"
]

expected_porch = X[porch_columns].sum(
    axis=1,
    min_count=4
)

check_porch = np.allclose(
    X_new["TotalPorchSF"],
    expected_porch,
    equal_nan=True
)


# 6. GarageAgeAtSale
expected_garage_age = X["Yr Sold"] - X["Garage Yr Blt"]

no_garage = (
    X["Garage Cars"].eq(0)
    & X["Garage Area"].eq(0)
)

expected_garage_age.loc[no_garage] = 0

check_garage_age = np.allclose(
    X_new["GarageAgeAtSale"],
    expected_garage_age,
    equal_nan=True
)


# Final results
checks = {
    "TotalSF": check_total_sf,
    "TotalBathrooms": check_bathrooms,
    "HouseAgeAtSale": check_house_age,
    "RemodAgeAtSale": check_remod_age,
    "TotalPorchSF": check_porch,
    "GarageAgeAtSale": check_garage_age
}

for feature, result in checks.items():
    print(feature, ":", "PASSED" if result else "FAILED")

print("\nAll Features Validated:", all(checks.values()))
