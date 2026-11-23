
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# STEP 1: LOAD DATASET
# ============================================================

df = pd.read_csv("data/AmesHousing.csv")

print("\n========== DATASET INFORMATION ==========")

print("Dataset Shape:", df.shape)
print("Total Rows:", df.shape[0])
print("Total Columns:", df.shape[1])


# ============================================================
# STEP 2: COMPLETE MISSING VALUES ANALYSIS
# ============================================================

print("\n========== MISSING VALUES ANALYSIS ==========")

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]

print("\nMissing Values:")
print(missing_values.sort_values(ascending=False))

print("\nTotal Missing Values:", missing_values.sum())

missing_info = pd.DataFrame({
    "Data Type": df[missing_values.index].dtypes,
    "Missing Count": missing_values,
    "Missing Percentage": (
        missing_values / len(df) * 100
    ).round(2)
})

print("\nMissing Values Data Types:")
print(missing_info)


# ============================================================
# STEP 3: ELECTRICAL INVESTIGATION
# ============================================================

print("\n========== ELECTRICAL INVESTIGATION ==========")

print("\nElectrical Value Counts:")
print(df["Electrical"].value_counts())

print("\nMissing Electrical Record:")

print(
    df[df["Electrical"].isnull()][
        ["Electrical", "Neighborhood", "Year Built",
         "Overall Qual", "SalePrice"]
    ]
)


# ============================================================
# STEP 4: GARAGE INVESTIGATION
# ============================================================

print("\n========== GARAGE INVESTIGATION ==========")

garage_missing = df[df["Garage Yr Blt"].isnull()]

print("\nGarage Year Missing Records:")

print(
    garage_missing[
        ["Garage Type", "Garage Yr Blt", "Garage Cars",
         "Garage Area", "Garage Finish"]
    ].head(10)
)

print("\nTotal Missing Garage Year:", len(garage_missing))

print("\n--- Garage Missing Value Verification ---")

print("Garage Type Missing:",
      garage_missing["Garage Type"].isnull().sum())

print("Garage Cars = 0:",
      (garage_missing["Garage Cars"] == 0).sum())

print("Garage Area = 0:",
      (garage_missing["Garage Area"] == 0).sum())

print("Garage Finish Missing:",
      garage_missing["Garage Finish"].isnull().sum())

print("Garage Qual Missing:",
      garage_missing["Garage Qual"].isnull().sum())

print("Garage Cond Missing:",
      garage_missing["Garage Cond"].isnull().sum())

print("\n--- Garage Exceptional Records ---")

garage_exceptions = garage_missing[
    garage_missing["Garage Type"].notnull()
    | garage_missing["Garage Cars"].ne(0)
    | garage_missing["Garage Area"].ne(0)
]

print(
    garage_exceptions[
        ["Garage Type", "Garage Yr Blt", "Garage Cars",
         "Garage Area", "Garage Finish", "Garage Qual",
         "Garage Cond", "SalePrice"]
    ].to_string()
)


# ============================================================
# STEP 5: BASEMENT INVESTIGATION
# ============================================================

print("\n========== BASEMENT INVESTIGATION ==========")

basement_missing = df[df["Bsmt Qual"].isnull()]

print("\nTotal Missing Basement Quality:",
      len(basement_missing))

print("\nBasement Missing Records:")

print(
    basement_missing[
        ["Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
         "BsmtFin Type 1", "Total Bsmt SF",
         "Bsmt Full Bath", "Bsmt Half Bath", "SalePrice"]
    ].head(10)
)

print("\n--- Basement Verification ---")

print("Bsmt Cond Missing:",
      basement_missing["Bsmt Cond"].isnull().sum())

print("Bsmt Exposure Missing:",
      basement_missing["Bsmt Exposure"].isnull().sum())

print("BsmtFin Type 1 Missing:",
      basement_missing["BsmtFin Type 1"].isnull().sum())

print("Total Bsmt SF = 0:",
      (basement_missing["Total Bsmt SF"] == 0).sum())

print("Bsmt Full Bath = 0:",
      (basement_missing["Bsmt Full Bath"] == 0).sum())

print("Bsmt Half Bath = 0:",
      (basement_missing["Bsmt Half Bath"] == 0).sum())

print("\n--- Basement Exceptional Records ---")

basement_exceptions = basement_missing[
    basement_missing["Total Bsmt SF"].ne(0)
    | basement_missing["Bsmt Full Bath"].ne(0)
    | basement_missing["Bsmt Half Bath"].ne(0)
]

print("Exceptional Records:", len(basement_exceptions))

print(
    basement_exceptions[
        ["Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
         "BsmtFin Type 1", "BsmtFin SF 1",
         "BsmtFin SF 2", "Bsmt Unf SF",
         "Total Bsmt SF", "Bsmt Full Bath",
         "Bsmt Half Bath", "SalePrice"]
    ].to_string()
)

print("\n--- Basement Type 2 Investigation ---")

type2_missing = df[df["BsmtFin Type 2"].isnull()]

type2_exceptions = type2_missing[
    type2_missing["Bsmt Qual"].notnull()
    | type2_missing["Total Bsmt SF"].gt(0)
]

print("Total Type 2 Missing:", len(type2_missing))
print("Exceptional Records:", len(type2_exceptions))

print(
    type2_exceptions[
        ["Bsmt Qual", "BsmtFin Type 1",
         "BsmtFin Type 2", "Total Bsmt SF", "SalePrice"]
    ].to_string()
)

print("\n--- Basement Type 2 Index 444 ---")

print(
    df.loc[444, [
        "Bsmt Qual",
        "BsmtFin Type 1",
        "BsmtFin SF 1",
        "BsmtFin Type 2",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF"
    ]].to_string()
)


# ============================================================
# STEP 6: MASONRY VENEER INVESTIGATION
# ============================================================

print("\n========== MASONRY VENEER INVESTIGATION ==========")

masonry_missing = df[df["Mas Vnr Type"].isnull()]

print("Missing Masonry Type:", len(masonry_missing))

print("\nMissing Masonry Records:")

print(
    masonry_missing[
        ["Mas Vnr Type", "Mas Vnr Area",
         "Overall Qual", "SalePrice"]
    ].head(10)
)

print("\nMas Vnr Area = 0:",
      (masonry_missing["Mas Vnr Area"] == 0).sum())

masonry_exceptions = masonry_missing[
    masonry_missing["Mas Vnr Area"].ne(0)
    | masonry_missing["Mas Vnr Area"].isnull()
]

print("\n--- Masonry Exceptional Records ---")

print("Exceptional Records:", len(masonry_exceptions))

print(
    masonry_exceptions[
        ["Mas Vnr Type", "Mas Vnr Area",
         "Overall Qual", "Year Built", "SalePrice"]
    ].to_string()
)

print("\n--- Masonry Detailed Inspection ---")

print(
    masonry_exceptions[
        ["Mas Vnr Type", "Mas Vnr Area",
         "Exterior 1st", "Exterior 2nd"]
    ].to_string()
)

print("\n--- Masonry Area Median Analysis ---")

print("Missing Area:", df["Mas Vnr Area"].isnull().sum())

print("Overall Median:", df["Mas Vnr Area"].median())

print(
    "Positive Area Median:",
    df.loc[df["Mas Vnr Area"] > 0, "Mas Vnr Area"].median()
)

print("\nMasonry Area Statistics:")
print(df["Mas Vnr Area"].describe())


# ============================================================
# STEP 7: PROPERTY FEATURE INVESTIGATION
# ============================================================

print("\n========== PROPERTY FEATURE INVESTIGATION ==========")

property_columns = [
    "Pool QC",
    "Fence",
    "Alley",
    "Misc Feature",
    "Fireplace Qu"
]

for column in property_columns:

    print(f"\n--- {column} ---")

    print("Missing:", df[column].isnull().sum())

    print("Existing Values:")
    print(df[column].value_counts().head(10))


# ============================================================
# STEP 8: LOT FRONTAGE INVESTIGATION
# ============================================================

print("\n========== LOT FRONTAGE INVESTIGATION ==========")

print("Missing Lot Frontage:",
      df["Lot Frontage"].isnull().sum())

print("\nMissing Lot Frontage Records:")

print(
    df[df["Lot Frontage"].isnull()][
        ["Neighborhood", "Lot Area",
         "Lot Frontage", "SalePrice"]
    ].head(10)
)

lot_frontage_analysis = df.groupby(
    "Neighborhood"
)["Lot Frontage"].agg(["count", "median"])

print("\nLot Frontage by Neighborhood:")

print(lot_frontage_analysis.sort_values("count"))

print("\nNeighborhoods with No Lot Frontage Data:")

print(
    lot_frontage_analysis[
        lot_frontage_analysis["count"] == 0
    ]
)


# ============================================================
# STEP 9: IDENTIFIER COLUMNS INVESTIGATION
# ============================================================

print("\n========== IDENTIFIER COLUMNS ==========")

identifier_columns = ["Order", "PID"]

for column in identifier_columns:

    print(f"\n{column}")

    print("Unique Values:", df[column].nunique())
    print("Total Records:", len(df))
    print("Missing Values:", df[column].isnull().sum())
    print("Sample Values:", df[column].head(10).tolist())


# ============================================================
# STEP 10: TARGET VARIABLE INVESTIGATION
# ============================================================

print("\n========== TARGET VARIABLE ==========")

target = "SalePrice"

print("Target Column:", target)
print("Data Type:", df[target].dtype)
print("Missing Values:", df[target].isnull().sum())
print("Unique Values:", df[target].nunique())

print("\nTarget Statistics:")
print(df[target].describe())

print("Minimum SalePrice:", df[target].min())
print("Maximum SalePrice:", df[target].max())


# ============================================================
# STEP 11: TRAIN-TEST SPLIT
# ============================================================

X = df.drop(columns=["SalePrice", "Order", "PID"])
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")

print("Original Dataset:", df.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ============================================================
# STEP 12: NUMERICAL AND CATEGORICAL FEATURE SEPARATION
# ============================================================

numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\n========== FEATURE TYPE ANALYSIS ==========")

print("Numerical Features:", len(numerical_columns))
print("Categorical Features:", len(categorical_columns))

print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================================
# STEP 13: TRAINING DATA MISSING VALUES
# ============================================================

numerical_missing = (
    X_train[numerical_columns].isnull().sum()
)

numerical_missing = numerical_missing[
    numerical_missing > 0
]

categorical_missing = (
    X_train[categorical_columns].isnull().sum()
)

categorical_missing = categorical_missing[
    categorical_missing > 0
]

print("\n========== TRAINING MISSING VALUES ==========")

print("\nNumerical Missing Values:")
print(numerical_missing.sort_values(ascending=False))

print("\nNumerical Columns with Missing Values:",
      len(numerical_missing))

print("\nCategorical Missing Values:")
print(categorical_missing.sort_values(ascending=False))

print("\nCategorical Columns with Missing Values:",
      len(categorical_missing))


# ============================================================
# STEP 14: FIREPLACE INVESTIGATION
# ============================================================

print("\n========== FIREPLACE INVESTIGATION ==========")

fireplace_missing = X_train["Fireplace Qu"].isnull()

print("Missing Fireplace Quality:",
      fireplace_missing.sum())

print("\nFireplaces Count for Missing Quality:")

print(
    X_train.loc[
        fireplace_missing, "Fireplaces"
    ].value_counts(dropna=False).sort_index()
)

print("\nMissing Quality but Existing Fireplace:")

print(
    X_train.loc[
        fireplace_missing & (X_train["Fireplaces"] > 0),
        ["Fireplaces", "Fireplace Qu"]
    ]
)


# ============================================================
# STEP 15: TRAINING MASONRY INVESTIGATION
# ============================================================

print("\n========== TRAINING MASONRY INVESTIGATION ==========")

masonry_type_missing = X_train["Mas Vnr Type"].isnull()

print("Missing Masonry Type:", masonry_type_missing.sum())

print("\nMasonry Area Values for Missing Types:")

print(
    X_train.loc[
        masonry_type_missing, "Mas Vnr Area"
    ].value_counts(dropna=False).sort_index()
)

print("\nMissing Type and Missing Area:")

print(
    X_train.loc[
        masonry_type_missing
        & X_train["Mas Vnr Area"].isnull(),
        ["Mas Vnr Type", "Mas Vnr Area",
         "Exterior 1st", "Exterior 2nd"]
    ]
)

print("\nMissing Type but Positive Area:")

print(
    X_train.loc[
        masonry_type_missing
        & (X_train["Mas Vnr Area"] > 0),
        ["Mas Vnr Type", "Mas Vnr Area",
         "Exterior 1st", "Exterior 2nd"]
    ]
)

print("\n--- Masonry Area Median Analysis ---")

print("Missing Area:",
      X_train["Mas Vnr Area"].isnull().sum())

print("Overall Training Median:",
      X_train["Mas Vnr Area"].median())

print("Positive Area Median:",
      X_train.loc[
          X_train["Mas Vnr Area"] > 0,
          "Mas Vnr Area"
      ].median())

print("\nArea Statistics:")
print(X_train["Mas Vnr Area"].describe())


# ============================================================
# STEP 16: LOT FRONTAGE TRAINING INVESTIGATION
# ============================================================

print("\n========== LOT FRONTAGE TRAINING INVESTIGATION ==========")

print("Missing Lot Frontage:",
      X_train["Lot Frontage"].isnull().sum())

print("\nMissing Values by Neighborhood:")

print(
    X_train.groupby("Neighborhood")["Lot Frontage"]
    .apply(lambda x: x.isnull().sum())
    .sort_values(ascending=False)
)

neighborhood_medians = (
    X_train.groupby("Neighborhood")["Lot Frontage"].median()
)

global_median = X_train["Lot Frontage"].median()

print("\nNeighborhood-wise Lot Frontage Median:")
print(neighborhood_medians.sort_values())

print("\nGlobal Training Median:", global_median)

print("\nNeighborhoods without a Median:")

print(neighborhood_medians[neighborhood_medians.isnull()])

print("\nMissing Records in Those Neighborhoods:")

print(
    X_train.loc[
        X_train["Lot Frontage"].isnull()
        & X_train["Neighborhood"].isin(
            neighborhood_medians[
                neighborhood_medians.isnull()
            ].index
        ),
        "Neighborhood"
    ].value_counts()
)


# ============================================================
# STEP 17: REMAINING NUMERICAL MISSING VALUES
# ============================================================

print("\n========== NUMERICAL MISSING RECORDS ==========")

columns_to_check = [
    "Garage Yr Blt",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Total Bsmt SF",
    "Bsmt Unf SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath",
    "Garage Cars",
    "Garage Area"
]

for column in columns_to_check:

    missing_records = X_train[
        X_train[column].isnull()
    ]

    print(f"\n{column}: {len(missing_records)} missing")

    if not missing_records.empty:

        print(
            missing_records[
                [
                    "Neighborhood",
                    "Year Built",
                    "Garage Type",
                    "Garage Cars",
                    "Garage Area",
                    "Total Bsmt SF",
                    "Bsmt Full Bath",
                    "Bsmt Half Bath"
                ]
            ]
        )


# ============================================================
# STEP 18: EXCEPTIONAL RECORD INSPECTION
# ============================================================

print("\n========== EXCEPTIONAL RECORD INSPECTION ==========")

exception_columns = [
    "Neighborhood",
    "Year Built",
    "Overall Qual",

    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2",
    "BsmtFin SF 1",
    "BsmtFin SF 2",
    "Bsmt Unf SF",
    "Total Bsmt SF",
    "Bsmt Full Bath",
    "Bsmt Half Bath",

    "Garage Type",
    "Garage Yr Blt",
    "Garage Finish",
    "Garage Cars",
    "Garage Area",
    "Garage Qual",
    "Garage Cond"
]

for index in [1341, 2236]:

    print(f"\n--- Record Index {index} ---")

    print(
        X_train.loc[
            index, exception_columns
        ].to_string()
    )


# ============================================================
# STEP 19: BASEMENT EXCEPTION DETAILED INSPECTION
# ============================================================

print("\n========== BASEMENT EXCEPTION: INDEX 1341 ==========")

basement_detail_columns = [
    "Neighborhood",
    "Year Built",
    "Overall Qual",
    "Foundation",
    "Bldg Type",
    "House Style",
    "1st Flr SF",
    "2nd Flr SF",
    "Gr Liv Area"
]

print(
    df.loc[
        1341, basement_detail_columns
    ].to_string()
)

print("\nSale Price:")
print(df.loc[1341, "SalePrice"])


# ============================================================
# STEP 20: FINAL MISSING VALUE SUMMARY
# ============================================================

print("\n========== FINAL TRAINING MISSING SUMMARY ==========")

print("\nNumerical Missing Values:")

print(
    X_train[numerical_columns]
    .isnull()
    .sum()
    .loc[lambda x: x > 0]
    .sort_values(ascending=False)
)

print("\nCategorical Missing Values:")

print(
    X_train[categorical_columns]
    .isnull()
    .sum()
    .loc[lambda x: x > 0]
    .sort_values(ascending=False)
)

print("\nTotal Numerical Missing Cells:",
      X_train[numerical_columns].isnull().sum().sum())

print("Total Categorical Missing Cells:",
      X_train[categorical_columns].isnull().sum().sum())

print("\n========== INVESTIGATION COMPLETE ==========")
print("No original dataset values have been modified.")
print("No imputation, encoding, or scaling has been performed.")


# ============================================================
# STEP 21: REMAINING CATEGORICAL MISSING VALUES INVESTIGATION
# ============================================================

print("\n========== STEP 21: CATEGORICAL INVESTIGATION ==========")

# 1. Pool Investigation
print("\n--- Pool QC Investigation ---")

pool_missing = X_train["Pool QC"].isnull()

print("Missing Pool QC:", pool_missing.sum())

print("\nPool Area for Missing Pool QC:")
print(
    X_train.loc[pool_missing, "Pool Area"]
    .value_counts(dropna=False)
    .sort_index()
)

print("\nMissing Pool QC but Pool Area > 0:")
print(
    X_train.loc[
        pool_missing & (X_train["Pool Area"] > 0),
        ["Pool QC", "Pool Area"]
    ]
)

# 2. Garage Investigation
print("\n--- Garage Investigation ---")

garage_columns = [
    "Garage Type",
    "Garage Finish",
    "Garage Qual",
    "Garage Cond"
]

for column in garage_columns:
    print(f"\n{column}:")
    print("Missing:", X_train[column].isnull().sum())

print("\nGarage Type vs Garage Cars:")
print(
    pd.crosstab(
        X_train["Garage Type"].fillna("MISSING"),
        X_train["Garage Cars"].fillna(-1)
    )
)

# 3. Basement Investigation
print("\n--- Basement Investigation ---")

basement_columns = [
    "Bsmt Qual",
    "Bsmt Cond",
    "Bsmt Exposure",
    "BsmtFin Type 1",
    "BsmtFin Type 2"
]

for column in basement_columns:
    print(f"\n{column}:")
    print("Missing:", X_train[column].isnull().sum())

print("\nMissing Basement Quality vs Total Basement SF:")
print(
    X_train.loc[
        X_train["Bsmt Qual"].isnull(),
        [
            "Bsmt Qual",
            "Bsmt Cond",
            "Bsmt Exposure",
            "BsmtFin Type 1",
            "BsmtFin Type 2",
            "Total Bsmt SF"
        ]
    ].value_counts(dropna=False)
)

# 4. Alley Investigation
print("\n--- Alley Investigation ---")
print(X_train["Alley"].value_counts(dropna=False))

# 5. Fence Investigation
print("\n--- Fence Investigation ---")
print(X_train["Fence"].value_counts(dropna=False))

# 6. Misc Feature Investigation
print("\n--- Misc Feature Investigation ---")
print(X_train["Misc Feature"].value_counts(dropna=False))

# 7. Masonry Verification
print("\n--- Masonry Verification ---")
print(
    X_train.groupby(
        X_train["Mas Vnr Type"].isnull()
    )["Mas Vnr Area"].agg(
        ["count", "median", "min", "max"]
    )
)

print("\n========== STEP 21 COMPLETE ==========")
