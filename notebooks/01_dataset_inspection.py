
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