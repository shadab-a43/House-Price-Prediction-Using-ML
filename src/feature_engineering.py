
import pandas as pd


def add_engineered_features(X):

    # Original data ko safe rakhne ke liye copy
    X = X.copy()

    # 1. Total Floor Area
    X["TotalSF"] = X[
        ["Total Bsmt SF", "1st Flr SF", "2nd Flr SF"]
    ].sum(axis=1, min_count=3)

    # Identify properties without a basement
    no_basement = X["Total Bsmt SF"].eq(0)

    # Missing basement bathroom values ko zero karna
    basement_bath_columns = [
        "Bsmt Full Bath",
        "Bsmt Half Bath"
    ]

    X.loc[no_basement, basement_bath_columns] = (
        X.loc[no_basement, basement_bath_columns].fillna(0)
    )

    # 2. Total Bathrooms
    
    X["TotalBathrooms"] = (
        X["Full Bath"]
        + (0.5 * X["Half Bath"])
        + X["Bsmt Full Bath"]
        + (0.5 * X["Bsmt Half Bath"])
    )

    # 3. House Age at Sale
    X["HouseAgeAtSale"] = (
        X["Yr Sold"] - X["Year Built"]
    )

    # 4. Years Since Remodeling
    X["RemodAgeAtSale"] = (
        X["Yr Sold"] - X["Year Remod/Add"]
    )

    # 5. Total Porch Area
    porch_columns = [
        "Open Porch SF",
        "Enclosed Porch",
        "3Ssn Porch",
        "Screen Porch"
    ]

    X["TotalPorchSF"] = X[
        porch_columns
    ].sum(axis=1, min_count=4)

    # 6. Garage Age at Sale
    X["GarageAgeAtSale"] = (
        X["Yr Sold"] - X["Garage Yr Blt"]
    )

    # No-garage properties ke liye garage age = 0
    no_garage = (
        X["Garage Cars"].eq(0)
        & X["Garage Area"].eq(0)
    )

    X.loc[no_garage, "GarageAgeAtSale"] = 0

    return X
