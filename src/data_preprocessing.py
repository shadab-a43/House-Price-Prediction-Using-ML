
import pandas as pd

# Load dataset
df = pd.read_csv("data/AmesHousing.csv")

# Dataset information
print("Dataset Shape:", df.shape)

# Missing values analysis
missing_values = df.isnull().sum()

missing_values = missing_values[missing_values > 0]

print("\nMissing Values:")
print(missing_values.sort_values(ascending=False))

print("\nTotal Missing Values:", missing_values.sum())



# Classify columns with missing values
print("\nMissing Values Data Types:")

missing_info = pd.DataFrame({
    "Data Type": df[missing_values.index].dtypes,
    "Missing Count": missing_values,
    "Missing Percentage": (
        missing_values / len(df) * 100
    ).round(2)
})

print(missing_info)


# Inspect special missing value cases

print("\n--- Electrical Missing Record ---")

print(
    df[df["Electrical"].isnull()][
        ["Electrical", "Overall Qual", "Year Built", "SalePrice"]
    ]
)

print("\n--- Garage Year Missing Records ---")

garage_missing = df[df["Garage Yr Blt"].isnull()]

print(
    garage_missing[
        ["Garage Type", "Garage Yr Blt", "Garage Cars",
         "Garage Area", "Garage Finish"]
    ].head(10)
)

print("\nTotal records with missing Garage Year:",
      len(garage_missing))



# Verify missing garage records

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



# Inspect remaining 2 garage records

print("\n--- Remaining Garage Records ---")

print(
    garage_missing[
        (garage_missing["Garage Type"].notnull()) |
        (garage_missing["Garage Cars"] != 0) |
        (garage_missing["Garage Area"] != 0)
    ][
        ["Garage Type", "Garage Yr Blt", "Garage Cars",
         "Garage Area", "Garage Finish", "Garage Qual",
         "Garage Cond", "SalePrice"]
    ]
)


# Inspect missing basement records

print("\n--- Basement Missing Records ---")

basement_missing = df[df["Bsmt Qual"].isnull()]

print("Total Missing Basement Quality:",
      len(basement_missing))

print(
    basement_missing[
        ["Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
         "BsmtFin Type 1", "Total Bsmt SF",
         "Bsmt Full Bath", "Bsmt Half Bath", "SalePrice"]
    ].head(10)
)


# Verify missing basement records

print("\n--- Basement Verification ---")

print("Total records:", len(basement_missing))

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



# Inspect basement exceptions

print("\n--- Basement Exceptional Records ---")

basement_exceptions = basement_missing[
    (basement_missing["Total Bsmt SF"] != 0) |
    (basement_missing["Bsmt Full Bath"] != 0) |
    (basement_missing["Bsmt Half Bath"] != 0)
]

print("Exceptional records:", len(basement_exceptions))

print(
    basement_exceptions[
        ["Bsmt Qual", "Bsmt Cond", "Bsmt Exposure",
         "BsmtFin Type 1", "BsmtFin SF 1",
         "BsmtFin SF 2", "Bsmt Unf SF",
         "Total Bsmt SF", "Bsmt Full Bath",
         "Bsmt Half Bath", "SalePrice"]
    ].to_string()
)


# Inspect masonry veneer missing values

print("\n--- Masonry Veneer Missing Records ---")

masonry_missing = df[df["Mas Vnr Type"].isnull()]

print("Total records:", len(masonry_missing))

print(
    masonry_missing[
        ["Mas Vnr Type", "Mas Vnr Area", "Overall Qual", "SalePrice"]
    ].head(10)
)

print("\nMas Vnr Area = 0:",
      (masonry_missing["Mas Vnr Area"] == 0).sum())

print("\n--- Basement Type 2 Missing Records ---")

print(
    df[df["BsmtFin Type 2"].isnull()][
        ["Bsmt Qual", "BsmtFin Type 1",
         "BsmtFin Type 2", "Total Bsmt SF"]
    ].head(10)
)




# Inspect remaining masonry veneer records

print("\n--- Masonry Veneer Exceptions ---")

masonry_exceptions = masonry_missing[
    (masonry_missing["Mas Vnr Area"] != 0) |
    (masonry_missing["Mas Vnr Area"].isnull())
]

print("Exceptional records:", len(masonry_exceptions))

print(
    masonry_exceptions[
        ["Mas Vnr Type", "Mas Vnr Area",
         "Overall Qual", "Year Built", "SalePrice"]
    ].to_string()
)


# Inspect basement Type 2 exceptions

print("\n--- Basement Type 2 Exceptions ---")

type2_missing = df[df["BsmtFin Type 2"].isnull()]

type2_exceptions = type2_missing[
    (type2_missing["Bsmt Qual"].notnull()) |
    (type2_missing["Total Bsmt SF"] > 0)
]

print("Total Type 2 missing:", len(type2_missing))
print("Exceptional records:", len(type2_exceptions))

print(
    type2_exceptions[
        ["Bsmt Qual", "BsmtFin Type 1",
         "BsmtFin Type 2", "Total Bsmt SF", "SalePrice"]
    ].to_string()
)



# Inspect masonry veneer exceptions in detail

print("\n--- Masonry Veneer Detailed Inspection ---")

print(
    masonry_exceptions[
        ["Mas Vnr Type", "Mas Vnr Area",
         "Exterior 1st", "Exterior 2nd"]
    ].to_string()
)


# Inspect Basement Type 2 exception

print("\n--- Basement Type 2 Detailed Inspection ---")

print(
    df.loc[444, [
        "Bsmt Qual",
        "BsmtFin Type 1",
        "BsmtFin SF 1",
        "BsmtFin Type 2",
        "BsmtFin SF 2",
        "Bsmt Unf SF",
        "Total Bsmt SF"
    ]]
)


# Inspect missing values in property features

print("\n--- Property Feature Missing Values ---")

property_columns = [
    "Pool QC",
    "Fence",
    "Alley",
    "Misc Feature",
    "Fireplace Qu"
]

for column in property_columns:
    print(f"\n{column}:")
    print("Missing:", df[column].isnull().sum())
    print("Existing values:")
    print(df[column].value_counts().head(10))


# Inspect Lot Frontage missing records

print("\n--- Lot Frontage Inspection ---")

print("Missing:", df["Lot Frontage"].isnull().sum())

print(
    df[df["Lot Frontage"].isnull()][
        ["Neighborhood", "Lot Area", "Lot Frontage", "SalePrice"]
    ].head(10)
)


# Inspect Electrical categories

print("\n--- Electrical Inspection ---")

print("Electrical value counts:")
print(df["Electrical"].value_counts())

print("\nMissing Electrical record:")

print(
    df[df["Electrical"].isnull()][
        ["Electrical", "Neighborhood", "Year Built",
         "Overall Qual", "SalePrice"]
    ]
)


# Check Lot Frontage availability by Neighborhood

print("\n--- Lot Frontage by Neighborhood ---")

lot_frontage_analysis = df.groupby("Neighborhood")[
    "Lot Frontage"
].agg(["count", "median"])

print(lot_frontage_analysis.sort_values("count"))

print("\nNeighborhoods with no Lot Frontage data:",
      lot_frontage_analysis["count"].eq(0).sum())


# Inspect identifier columns

print("\n--- Identifier Columns Analysis ---")

identifier_columns = ["Order", "PID"]

for column in identifier_columns:
    print(f"\n{column}")
    print("Unique Values:", df[column].nunique())
    print("Total Records:", len(df))
    print("Missing Values:", df[column].isnull().sum())
    print("Sample Values:")
    print(df[column].head(10).tolist())


# Target Variable Verification

print("\n--- Target Variable Analysis ---")

target = "SalePrice"

print("Target Column:", target)
print("Data Type:", df[target].dtype)
print("Missing Values:", df[target].isnull().sum())
print("Unique Values:", df[target].nunique())

print("\nTarget Statistics:")
print(df[target].describe())

print("\nMinimum SalePrice:", df[target].min())
print("Maximum SalePrice:", df[target].max())


# Step 17: Train-Test Split

from sklearn.model_selection import train_test_split

# Separate features and target
X = df.drop(columns=["SalePrice", "Order", "PID"])
y = df["SalePrice"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Display split information
print("\n--- Train-Test Split Results ---")

print("Original Dataset:", df.shape)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# Step 18: Separate Numerical and Categorical Features

# Identify numerical columns
numerical_columns = X_train.select_dtypes(
    include=["number"]
).columns.tolist()

# Identify categorical columns
categorical_columns = X_train.select_dtypes(
    exclude=["number"]
).columns.tolist()

print("\n--- Feature Type Analysis ---")

print("Total Numerical Features:", len(numerical_columns))
print("Total Categorical Features:", len(categorical_columns))

print("\nNumerical Features:")
print(numerical_columns)

print("\nCategorical Features:")
print(categorical_columns)



# Step 18.2: Missing Values in Training Data

numerical_missing = X_train[numerical_columns].isnull().sum()
numerical_missing = numerical_missing[numerical_missing > 0]

categorical_missing = X_train[categorical_columns].isnull().sum()
categorical_missing = categorical_missing[categorical_missing > 0]

print("\n--- Numerical Missing Values ---")
print(numerical_missing.sort_values(ascending=False))

print("\nTotal Numerical Columns with Missing Values:",
      len(numerical_missing))

print("\n--- Categorical Missing Values ---")
print(categorical_missing.sort_values(ascending=False))

print("\nTotal Categorical Columns with Missing Values:",
      len(categorical_missing))

