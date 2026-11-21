
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. DATASET LOADING AND INSPECTION
# ==========================================

# Dataset path
file_path = "data/AmesHousing.csv"

# Read CSV file
df = pd.read_csv(file_path)

# First 5 rows
print("\nFirst 5 Rows:")
print(df.head())

# Dataset shape
print("\nDataset Shape:")
print(df.shape)

print("\nTotal Rows:", df.shape[0])
print("Total Columns:", df.shape[1])

# Column names
print("\nColumn Names:")
print(df.columns.tolist())

# Dataset information
print("\nDataset Information:")
df.info()


# ==========================================
# 2. MISSING VALUES ANALYSIS
# ==========================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nTotal Missing Values:")
print(df.isnull().sum().sum())

# Missing values count
missing_values = df.isnull().sum()

missing_values = missing_values[
    missing_values > 0
].sort_values(ascending=False)

print("\nColumns with Missing Values:")
print(missing_values)

# Missing values report
missing_percentage = (
    df.isnull().sum() / len(df)
) * 100

missing_report = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": missing_percentage
})

missing_report = missing_report[
    missing_report["Missing Count"] > 0
]

missing_report = missing_report.sort_values(
    by="Missing Percentage",
    ascending=False
)

print("\nMissing Values Report:")
print(missing_report.round(2).to_string())


# ==========================================
# 3. DUPLICATE ROW ANALYSIS
# ==========================================

duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows:", duplicate_count)


# ==========================================
# 4. SALEPRICE STATISTICAL ANALYSIS
# ==========================================

print("\nSalePrice Statistics:")
print(df["SalePrice"].describe())

print("\nLowest SalePrice:", df["SalePrice"].min())
print("Highest SalePrice:", df["SalePrice"].max())
print("Average SalePrice:", df["SalePrice"].mean())
print("Median SalePrice:", df["SalePrice"].median())


# ==========================================
# 5. SALEPRICE DISTRIBUTION
# ==========================================

plt.figure(figsize=(10, 5))

sns.histplot(
    df["SalePrice"],
    bins=40,
    kde=True
)

plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price ($)")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.show()


# ==========================================
# 6. FEATURE VS SALEPRICE ANALYSIS
# ==========================================

# Living Area vs SalePrice
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Gr Liv Area",
    y="SalePrice"
)

plt.title("Living Area vs House Sale Price")
plt.xlabel("Living Area (Square Feet)")
plt.ylabel("Sale Price ($)")

plt.tight_layout()
plt.show()


# Overall Quality vs SalePrice
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Overall Qual",
    y="SalePrice",
    s=15,
    alpha=0.5
)

plt.title("Overall Quality vs House Sale Price")
plt.xlabel("Overall Quality (1-10)")
plt.ylabel("Sale Price ($)")

plt.tight_layout()
plt.show()


# Garage Area vs SalePrice
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Garage Area",
    y="SalePrice",
    s=15,
    alpha=0.5
)

plt.title("Garage Area vs House Sale Price")
plt.xlabel("Garage Area (Square Feet)")
plt.ylabel("Sale Price ($)")

plt.tight_layout()
plt.show()


# Year Built vs SalePrice
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Year Built",
    y="SalePrice",
    s=15,
    alpha=0.5
)

plt.title("Year Built vs House Sale Price")
plt.xlabel("Year Built")
plt.ylabel("Sale Price ($)")

plt.tight_layout()
plt.show()


# ==========================================
# 7. CORRELATION ANALYSIS
# ==========================================

numeric_df = df.select_dtypes(include=["number"])

correlation = numeric_df.corr()

plt.figure(figsize=(14, 10))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False,
    linewidths=0.5
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()


# Top 10 features correlated with SalePrice
saleprice_corr = correlation["SalePrice"].sort_values(
    ascending=False
)

print("\nTop 10 Features Correlated with SalePrice:")

print(
    saleprice_corr
    .drop("SalePrice")
    .head(10)
)


# ==========================================
# 8. MISSING VALUES VISUALIZATION
# ==========================================

missing_percent = (
    df.isnull().mean() * 100
)

missing_percent = missing_percent[
    missing_percent > 0
].sort_values(ascending=False).head(10)

plt.figure(figsize=(10, 6))

sns.barplot(
    x=missing_percent.values,
    y=missing_percent.index
)

plt.title("Top 10 Columns with Missing Values")
plt.xlabel("Missing Values (%)")
plt.ylabel("Features")

plt.tight_layout()
plt.show()


# ==========================================
# 9. SALEPRICE OUTLIER ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["SalePrice"])

plt.title("Outlier Analysis of SalePrice")
plt.xlabel("Sale Price ($)")

plt.tight_layout()
plt.show()


# SalePrice IQR Method
Q1 = df["SalePrice"].quantile(0.25)
Q3 = df["SalePrice"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["SalePrice"] < lower_bound) |
    (df["SalePrice"] > upper_bound)
]

print("\nSalePrice Outlier Analysis")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Total Potential Outliers:", len(outliers))

outlier_percentage = (
    len(outliers) / len(df)
) * 100

print("Outlier Percentage:", outlier_percentage, "%")


# ==========================================
# 10. GR LIV AREA OUTLIER ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["Gr Liv Area"])

plt.title("Outlier Analysis of Gr Liv Area")
plt.xlabel("Above Ground Living Area (sq ft)")

plt.tight_layout()
plt.show()


Q1 = df["Gr Liv Area"].quantile(0.25)
Q3 = df["Gr Liv Area"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Gr Liv Area"] < lower_bound) |
    (df["Gr Liv Area"] > upper_bound)
]

print("\nGr Liv Area Outlier Analysis")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Total Potential Outliers:", len(outliers))

outlier_percentage = (
    len(outliers) / len(df)
) * 100

print("Outlier Percentage:", outlier_percentage, "%")


# ==========================================
# 11. LOT AREA OUTLIER ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["Lot Area"])

plt.title("Outlier Analysis of Lot Area")
plt.xlabel("Lot Area (sq ft)")

plt.tight_layout()
plt.show()


Q1 = df["Lot Area"].quantile(0.25)
Q3 = df["Lot Area"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Lot Area"] < lower_bound) |
    (df["Lot Area"] > upper_bound)
]

print("\nLot Area Outlier Analysis")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Total Potential Outliers:", len(outliers))

outlier_percentage = (
    len(outliers) / len(df)
) * 100

print("Outlier Percentage:", outlier_percentage, "%")


# ==========================================
# 12. GARAGE AREA OUTLIER ANALYSIS
# ==========================================

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["Garage Area"])

plt.title("Outlier Analysis of Garage Area")
plt.xlabel("Garage Area (sq ft)")

plt.tight_layout()
plt.show()


Q1 = df["Garage Area"].quantile(0.25)
Q3 = df["Garage Area"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[
    (df["Garage Area"] < lower_bound) |
    (df["Garage Area"] > upper_bound)
]

print("\nGarage Area Outlier Analysis")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)
print("Total Potential Outliers:", len(outliers))

outlier_percentage = (
    len(outliers) / len(df)
) * 100

print("Outlier Percentage:", outlier_percentage, "%")


# ==========================================
# 13. HOUSE STYLE ANALYSIS
# ==========================================

print("\nHouse Style Categories:")
print(df["House Style"].value_counts())


# House Style Count Plot
plt.figure(figsize=(10, 6))

sns.countplot(
    data=df,
    x="House Style",
    order=df["House Style"].value_counts().index
)

plt.title("Distribution of House Styles")
plt.xlabel("House Style")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.show()


# House Style vs SalePrice
house_style_prices = (
    df.groupby("House Style")["SalePrice"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage SalePrice by House Style:")
print(house_style_prices.round(2))


plt.figure(figsize=(10, 6))

sns.barplot(
    x=house_style_prices.index,
    y=house_style_prices.values,
    errorbar=None
)

plt.title("Average SalePrice by House Style")
plt.xlabel("House Style")
plt.ylabel("Average SalePrice")

plt.tight_layout()
plt.show()


# ==========================================
# 14. NEIGHBORHOOD ANALYSIS
# ==========================================

print("\nTotal Neighborhoods:")
print(df["Neighborhood"].nunique())


# Houses in each neighborhood
neighborhood_counts = (
    df["Neighborhood"].value_counts()
)

print("\nHouses in Each Neighborhood:")
print(neighborhood_counts)


# Neighborhood Count Plot
plt.figure(figsize=(12, 6))

sns.countplot(
    data=df,
    x="Neighborhood",
    order=neighborhood_counts.index
)

plt.title("House Distribution by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("Number of Houses")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# Average SalePrice by Neighborhood
neighborhood_prices = (
    df.groupby("Neighborhood")["SalePrice"]
    .mean()
    .sort_values(ascending=False)
)

print("\nAverage SalePrice by Neighborhood:")
print(neighborhood_prices.round(2))


# Neighborhood vs SalePrice
plt.figure(figsize=(14, 6))

sns.barplot(
    x=neighborhood_prices.index,
    y=neighborhood_prices.values,
    errorbar=None
)

plt.title("Average SalePrice by Neighborhood")
plt.xlabel("Neighborhood")
plt.ylabel("Average SalePrice")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()


# ==========================================
# 15. FURTHER CATEGORICAL FEATURE ANALYSIS
# ==========================================

categorical_features = [
    "Bldg Type",
    "Exterior 1st",
    "Kitchen Qual",
    "MS Zoning"
]

for feature in categorical_features:

    print(f"\nCategory Distribution: {feature}")

    print(df[feature].value_counts())

    # Category Count Plot
    plt.figure(figsize=(10, 5))

    sns.countplot(
        data=df,
        x=feature,
        order=df[feature].value_counts().index
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Number of Houses")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    # Average SalePrice by Category
    category_prices = (
        df.groupby(feature)["SalePrice"]
        .mean()
        .sort_values(ascending=False)
    )

    print(f"\nAverage SalePrice by {feature}:")
    print(category_prices.round(2))

    plt.figure(figsize=(10, 5))

    sns.barplot(
        x=category_prices.index,
        y=category_prices.values,
        errorbar=None
    )

    plt.title(f"Average SalePrice by {feature}")
    plt.xlabel(feature)
    plt.ylabel("Average SalePrice")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


# ==========================================
# 16. FEATURE RELATIONSHIP ANALYSIS
# ==========================================

relationship_features = [
    "Overall Qual",
    "Gr Liv Area",
    "Total Bsmt SF",
    "Garage Cars",
    "Year Built"
]

for feature in relationship_features:

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=df,
        x=feature,
        y="SalePrice",
        alpha=0.5
    )

    plt.title(f"{feature} vs SalePrice")
    plt.xlabel(feature)
    plt.ylabel("SalePrice")

    plt.tight_layout()
    plt.show()


# Correlation of selected features with SalePrice
print("\nFeature Correlation with SalePrice:")

feature_correlation = (
    df[relationship_features + ["SalePrice"]]
    .corr(numeric_only=True)["SalePrice"]
    .drop("SalePrice")
    .sort_values(ascending=False)
)

print(feature_correlation.round(3))


# ==========================================
# 17. SKEWNESS ANALYSIS
# ==========================================

# Select numerical columns
numeric_df = df.select_dtypes(include=["number"])

# Remove identifier columns and target
numeric_features = numeric_df.drop(
    columns=["Order", "PID", "SalePrice"],
    errors="ignore"
)

# Calculate skewness
skewness = numeric_features.skew()

# Sort by absolute skewness
skewness = skewness.reindex(
    skewness.abs().sort_values(ascending=False).index
)

print("\nNumerical Feature Skewness:")
print(skewness.round(3))


# Identify highly skewed features
highly_skewed = skewness[
    skewness.abs() > 1
]

print("\nHighly Skewed Features (Absolute Skewness > 1):")
print(highly_skewed.round(3))


# Visualize top 6 highly skewed features
top_skewed_features = highly_skewed.head(6).index

for feature in top_skewed_features:

    plt.figure(figsize=(8, 5))

    sns.histplot(
        data=df,
        x=feature,
        kde=True
    )

    plt.title(f"Distribution of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


# SalePrice Skewness separately
saleprice_skewness = df["SalePrice"].skew()

print("\nSalePrice Skewness:")
print(round(saleprice_skewness, 3))


print("\n==========================================")
print("EDA ANALYSIS SCRIPT COMPLETED")
print("==========================================")
