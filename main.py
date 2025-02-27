
import pandas as pd
import numpy as np

# reading file
df = pd.read_csv('employee_attrition_dataset.csv')



# Display the first 12 rows
print("---------> a")

print("First 12 rows of the dataset:")
print(df.head(12))

# Display the last 12 rows
print("\nLast 12 rows of the dataset:")
print(df.tail(12))

print("---------> b")

# Print the total number of rows and columns
rows, cols = df.shape
print(f"\nTotal Rows: {rows}, Total Columns: {cols}")

print("---------> c")

# List all column names along with their corresponding data types
print("\nColumn names and their data types:")
print(df.dtypes)

print("---------> d")

# Print the name of the first column.
print("\nName of the First Column:", df.columns[0])
print("---------> f")

# Generate a summary of the dataset (non-null counts and data types).
print("\nDataset Summary:")
df.info()
print("---------> g")

# Display distinct values in the 'Department' column
distinct_departments = df['Department'].unique()
print("Distinct values in the 'Department' column:")
print(distinct_departments)

# Identify the most frequently occurring value in the 'Department' column
most_frequent_department = df['Department'].mode()[0]
print("\nMost frequently occurring value in the 'Department' column:", most_frequent_department)


print("---------> h")

# Calculate and present the mean, median, standard deviation, and percentiles for the 'Age' column
mean_age = df['Age'].mean()
median_age = df['Age'].median()
std_dev_age = df['Age'].std()
percentiles_age = df['Age'].quantile([0.25, 0.50, 0.75])

print("\nStatistics for the 'Age' column:")
print(f"Mean: {mean_age}")
print(f"Median: {median_age}")
print(f"Standard Deviation: {std_dev_age}")
print("Percentiles:")
print(percentiles_age)

