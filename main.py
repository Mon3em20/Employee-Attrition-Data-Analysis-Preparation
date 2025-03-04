
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
print("---------> e")

# Generate a summary of the dataset (non-null counts and data types).
print("\nDataset Summary:")
df.info()
print("---------> f")

# Display distinct values in the 'Department' column
distinct_departments = df['Department'].unique()
print("Distinct values in the 'Department' column:")
print(distinct_departments)

# Identify the most frequently occurring value in the 'Department' column
most_frequent_department = df['Department'].mode()[0]
print("\nMost frequently occurring value in the 'Department' column:", most_frequent_department)


print("---------> g")

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


print("----------> a")
# Apply a filter to select rows where 'Age' exceeds 30
filtered_df = df[df['Age'] > 30]

print("\nRows where 'Age' exceeds 30:")
print(filtered_df)

print("----------> b")
# Identify records where the 'Department' column starts with the letter 'S'
matching_records = df[df['Department'].str.startswith('S')]

# Count how many records match this condition
matching_count = matching_records.shape[0]

print("\nRecords where 'Department' starts with 'S':")
print(matching_records)
print(f"\nTotal number of records where 'Department' starts with 'S': {matching_count}")

print("----------> c")

# Determine the total number of duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"\nTotal number of duplicate rows: {duplicate_rows}")

# Remove duplicate rows
df = df.drop_duplicates()

# Verify that duplicates have been removed
print(f"\nTotal number of rows after removing duplicates: {df.shape[0]}")


grouped_data = df.groupby(['Attrition', 'Department']).size()
print("Grouped data by Attrition and Department:\n", grouped_data)

#Check for Missing Values and Replace Them if Found:
missing_values = df.isnull().sum()
df_filled_missing = df.copy()
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype == 'object':
            df_filled_missing[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df_filled_missing[column].fillna(df[column].median(), inplace=True)
print("Missing values in each column:\n", missing_values)


#divide a Numerical Column into 5 Equal-Width Bins and Count the Records in Each Bin:
bins = pd.cut(df['Age'], bins=5)
bin_counts = bins.value_counts()
print("Bin counts for Age:\n", bin_counts)




#Convert the data type of a numerical column from integer to string.

print("--------------> d")



# Convert the data type of the 'Age' column from integer to string

df['Age'] = df['Age'].astype(str)

# Verify the conversion
print(df.dtypes)



#Group the dataset based on two selected categorical features and analyze the results.
print("--------------> e")

# Group the dataset by 'Attrition' and 'Department'
grouped_data = df.groupby(['Attrition', 'Department']).size()

# Display the grouped data
print("Grouped data by Attrition and Department:\n", grouped_data)




# Check for missing values

print("--------------> f")
missing_values = df.isnull().any()

print("Missing values in each column:\n", missing_values)








