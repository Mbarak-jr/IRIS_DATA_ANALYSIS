import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Load dataset
try:
    iris_data = load_iris(as_frame=True)
    df = iris_data.frame
    print("Dataset loaded successfully.")
except Exception as e:
    print("Error loading dataset:", e)

# Display first few rows
print("\n\ud83d\udd0d First 5 rows of the dataset:")
print(df.head())

# Explore data structure
print("\n\ud83e\udde0 Data types and info:")
print(df.info())

# Check for missing values
print("\n❓ Missing values:")
print(df.isnull().sum())

# Clean the data (no missing values in Iris, but example for filling)
df.fillna(method='ffill', inplace=True)

# Basic statistics
print("\n\ud83d\udcc8 Descriptive statistics:")
print(df.describe())

# Group by species and compute mean
grouped = df.groupby('target').mean()
print("\n\ud83d\udcca Mean values grouped by target (species):")
print(grouped)

# Map target to species name for readability
df['species'] = df['target'].map(lambda x: iris_data.target_names[x])

# Interesting pattern: Mean petal length by species
print("\n\ud83c\udf38 Mean Petal Length by Species:")
print(df.groupby('species')['petal length (cm)'].mean())

# Set plot style
sns.set(style="whitegrid")

# 1. Line Chart: Sepal Length over Observations
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length')
plt.title("Sepal Length over Observations")
plt.xlabel("Observation")
plt.ylabel("Sepal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()

# 2. Bar Chart: Average Petal Length by Species
plt.figure(figsize=(8, 5))
sns.barplot(x='species', y='petal length (cm)', data=df, ci=None)
plt.title("Average Petal Length by Species")
plt.xlabel("Species")
plt.ylabel("Petal Length (cm)")
plt.tight_layout()
plt.show()

# 3. Histogram: Distribution of Sepal Width
plt.figure(figsize=(8, 5))
sns.histplot(df['sepal width (cm)'], kde=True, bins=20)
plt.title("Distribution of Sepal Width")
plt.xlabel("Sepal Width (cm)")
plt.tight_layout()
plt.show()

# 4. Scatter Plot: Sepal Length vs. Petal Length
plt.figure(figsize=(8, 5))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
plt.title("Sepal Length vs. Petal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Petal Length (cm)")
plt.legend()
plt.tight_layout()
plt.show()
