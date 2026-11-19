
import pandas as pd

# Dataset ka path
file_path = "data/AmesHousing.csv"

# CSV file read karna
df = pd.read_csv(file_path)

# Dataset ki first 5 rows dekhna
print(df.head())

# Total rows aur columns
print("\nDataset Shape:")
print(df.shape)

# Total rows
print("\nTotal Rows:", df.shape[0])

# Total columns
print("Total Columns:", df.shape[1])

# Saare column names
print("\nColumn Names:")
print(df.columns.tolist())


# Dataset ki information
print("\nDataset Information:")
df.info()

# Missing values count
print("\nMissing Values:")
print(df.isnull().sum())

# Total missing values
print("\nTotal Missing Values:")
print(df.isnull().sum().sum())


# Missing values in each column
missing_values = df.isnull().sum()

# Only columns having missing values
missing_values = missing_values[missing_values > 0]

# Sort from highest to lowest
missing_values = missing_values.sort_values(ascending=False)

print("\nColumns with Missing Values:")
print(missing_values)


# Missing values percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

# Combine count and percentage
missing_report = pd.DataFrame({
    "Missing Count": df.isnull().sum(),
    "Missing Percentage": missing_percentage
})

# Only columns with missing values
missing_report = missing_report[missing_report["Missing Count"] > 0]

# Sort by missing percentage
missing_report = missing_report.sort_values(
    by="Missing Percentage",
    ascending=False
)

print("\nMissing Values Report:")
print(missing_report.round(2).to_string())


# Check duplicate rows
duplicate_count = df.duplicated().sum()

print("\nDuplicate Rows:", duplicate_count)

# SalePrice Statistical Summary
print("\nSalePrice Statistics:")
print(df["SalePrice"].describe())

print("\nLowest SalePrice:", df["SalePrice"].min())
print("Highest SalePrice:", df["SalePrice"].max())
print("Average SalePrice:", df["SalePrice"].mean())
print("Median SalePrice:", df["SalePrice"].median())


import matplotlib.pyplot as plt
import seaborn as sns

# SalePrice Distribution
plt.figure(figsize=(10, 5))

sns.histplot(df["SalePrice"], bins=40, kde=True)

plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price ($)")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.show()


import matplotlib.pyplot as plt
import seaborn as sns

# Visualize SalePrice Distribution
plt.figure(figsize=(10, 5))

sns.histplot(df["SalePrice"], bins=40, kde=True)

plt.title("Distribution of House Sale Prices")
plt.xlabel("Sale Price ($)")
plt.ylabel("Number of Houses")

plt.tight_layout()
plt.show()



# Relationship between Living Area and Sale Price

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


# Relationship between Overall Quality and Sale Price
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

# Relationship between Garage Area and Sale Price
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

# Relationship between Year Built and Sale Price
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

# Correlation Analysis
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

# Top 10 Features Correlated with SalePrice
saleprice_corr = correlation["SalePrice"].sort_values(
    ascending=False
)

print("\nTop 10 Features Correlated with SalePrice:")
print(saleprice_corr.drop("SalePrice").head(10))

# Missing Values Visualization

missing_percent = (df.isnull().mean() * 100)

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

# Outlier Analysis - SalePrice

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["SalePrice"])

plt.title("Outlier Analysis of SalePrice")
plt.xlabel("Sale Price ($)")

plt.tight_layout()
plt.show()


# Outlier Analysis - IQR Method

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


outlier_percentage = (len(outliers) / len(df)) * 100

print("Outlier Percentage:", outlier_percentage, "%")


# Outlier Analysis - Gr Liv Area

plt.figure(figsize=(10, 5))

sns.boxplot(x=df["Gr Liv Area"])

plt.title("Outlier Analysis of Gr Liv Area")
plt.xlabel("Above Ground Living Area (sq ft)")

plt.tight_layout()
plt.show()
