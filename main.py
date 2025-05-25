import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

df = pd.read_csv('employee_attrition_dataset (2).csv')



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
print("\n---------> e")

# Generate a summary of the dataset (non-null counts and data types).
print("\nDataset Summary:")
df.info()
print("\n---------> f")

# Display distinct values in the 'Department' column
distinct_departments = df['Department'].unique()
print("Distinct values in the 'Department' column:")
print(distinct_departments)

print("\n---------> g")

# Identify the most frequently occurring value in the 'Department' column
most_frequent_department = df['Department'].mode()[0]
print("\nMost frequently occurring value in the 'Department' column:", most_frequent_department)


print("\n---------> h")

# Calculate and present the mean, median, standard deviation, and percentiles for the 'Age' column
mean_age = df['Age'].mean()
median_age = df['Age'].median()
std_dev_age = df['Age'].std()
percentiles_age =np.percentile(df['Age'], [20])

print("\nStatistics for the 'Age' column:")
print(f"Mean: {mean_age}")
print(f"Median: {median_age}")
print(f"Standard Deviation: {std_dev_age}")
print("Percentiles:")
print(percentiles_age)



####################################       Data Preparation Tasks       ##################################################

print("\n####################################       Data Preparation Tasks       ###########################################\n")



print("\n----------> a")

#a) Apply a filter to select rows where 'Age' exceeds 30
filtered_df = df[df['Age'] > 57]

print("\nRows where 'Age' exceeds 57:")
print(filtered_df)

print("----------> b")
#b) Identify records where the 'Department' column starts with the letter 'S'
matching_records = df[df['Department'].str.startswith('M')]

# Count how many records match this condition
matching_count = matching_records.shape[0]

print("\nRecords where 'Department' starts with 'S':")
print(matching_records)
print(f"\nTotal number of records where 'Department' starts with 'M': {matching_count}")

print("----------> c")

#c) Determine the total number of duplicate rows
duplicate_rows = df.duplicated().sum()
print(f"\nTotal number of duplicate rows: {duplicate_rows}")

# Remove duplicate rows
df = df.drop_duplicates()

# Verify that duplicates have been removed
print(f"\nTotal number of rows after removing duplicates: {df.shape[0]}")



print("--------------> d")



#D)Convert the data type of the 'Age' column from integer to string

df['Age'] = df['Age'].astype(str)

# Verify the conversion
print(df.dtypes)



#E)Group the dataset based on two selected categorical features and analyze the results.
print("--------------> e")

# Group the dataset by 'Attrition' and 'Department'
grouped_data = df.groupby(['Attrition', 'Department']).size()

# Display the grouped data
print("Grouped data by Attrition and Department:\n", grouped_data)




#F) Check for missing values

print("--------------> f")

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



print("--------------> G")
#G)Replace Missing Values with Median or Mode:

# Replace missing values with median or mode as appropriate
df_filled_missing = df.copy()
for column in df.columns:
    if df[column].isnull().sum() > 0:
        if df[column].dtype == 'object':
            df_filled_missing[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df_filled_missing[column].fillna(df[column].median(), inplace=True)

# Verify that there are no more missing values
missing_values_after = df_filled_missing.isnull().sum()
print("Missing values after filling:\n", missing_values_after)
#check if correct




#H) Divide a Numerical Column into 5 Equal-Width Bins and Count the Records in Each Bin:

print ("----------------------> h")

df = pd.read_csv('employee_attrition_dataset (2).csv')


# Choose the numerical column
numerical_column = 'Age'

# Divide the column into 5 equal-width bins
bins = pd.cut(df[numerical_column], bins=5)

# Count the number of records in each bin
bin_counts = bins.value_counts()



print(bin_counts)


# I) identify and Print the Row with the Maximum Value of a Selected Numerical Feature:


print ("----------------------> i")


# Load the data into a DataFrame
df = pd.read_csv('employee_attrition_dataset (2).csv')

# Select the numerical feature
feature = 'Age'

# Find the index of the row with the maximum value of the selected feature
max_index = df[feature].idxmax()

# Print the row corresponding to the maximum value
print(df.loc[max_index])



#J)Construct a Boxplot for a Significant Attribute:

print ("----------------> i")

# Load the data into a DataFrame
df = pd.read_csv('employee_attrition_dataset (2).csv')


# Select the numerical feature (column) for which you want to find the maximum value
feature = 'Age'  # Replace 'Age' with the desired column name

# Identify the row with the maximum value for the selected feature
max_value_row = df.loc[df[feature].idxmax()]

# Print the corresponding row
print(max_value_row)


print ("----------------> j")

# Create a boxplot for the Age attribute
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['Age'])
plt.title('Boxplot of Age')
plt.xlabel('Age')
plt.show()


print ("----------------> k")

# Generate a histogram for the 'Age' attribute
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=10, edgecolor='black')
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

#The age histogram helps identify the predominant age group,
# plan age-specific training, assess retirement potential,
# and analyze age-related trends, aiding workforce management decisions.



print ("----------------> L")

# Create a scatterplot
plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Monthly_Income'], alpha=0.5)
plt.title('Scatterplot of Age vs Monthly Income')
plt.xlabel('Age')
plt.ylabel('Monthly Income')
plt.grid(True)
plt.show()

#A scatterplot shows the correlation between age and monthly income:
# rising points indicate a positive correlation,
# falling points indicate a negative correlation,
# and scattered points suggest no correlation.


# m) Normalize the numerical attributes using StandardScaler to achieve standardized data.
print ("----------------> M")


# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
updated_df= pd.read_csv('employee_attrition_dataset (2).csv')

# Initialize the StandardScaler
scaler = StandardScaler()

# Apply the scaler to the numerical columns
updated_df[numerical_cols] = scaler.fit_transform(df[numerical_cols])


# Print the transformed dataset to the console
print(updated_df)



#n) Perform PCA (Principal Component Analysis) to reduce dimensionality to two components, and
 #visualize the dataset before and after applying PCA.


print ("----------------> N")


# Load the dataset
file_path = 'employee_attrition_dataset (2).csv'
data = pd.read_csv(file_path)

# Visualize the dataset using a pair plot
sns.pairplot(data)
plt.show()

# Standardize the data
features = data.select_dtypes(include=[float, int])  # Select only numerical columns
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply PCA to reduce the dimensionality to two components
pca = PCA(n_components=2)
principal_components = pca.fit_transform(scaled_features)

# Create a DataFrame with the principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])

# Visualize the transformed data using a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - First Two Principal Components')
plt.grid()
plt.show()

# Plot the two principal component axes (eigenvectors)
plt.figure(figsize=(8, 6))
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.quiver(0, 0, pca.components_[0, 0], pca.components_[0, 1], angles='xy', scale_units='xy', scale=1, color='r')
plt.quiver(0, 0, pca.components_[1, 0], pca.components_[1, 1], angles='xy', scale_units='xy', scale=1, color='b')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA - Principal Component Axes')
plt.grid()
plt.show()


print ("----------------> o")




#O) Analyze the Correlation Between Numerical Features Using a Heatmap:
# Calculate the correlation matrix
corr_matrix = df.corr(numeric_only=True)

# Display the correlation matrix using a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()




print("####################################  Practical Analytical Questions ##########################################\n")

#a) Use Python to calculate and display the correlation matrix, and identify potential features relevant
#for classification.

print ("----------------> A\n")
corr_matrix = df.corr(numeric_only=True)

# Display the correlation matrix
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# Identify highly correlated features (absolute value > 0.5)
high_corr = corr_matrix.abs() > 0.5
print("Highly correlated features (absolute value > 0.5):\n")
print(high_corr)




#Use Python to find the class distribution of a selected categorical feature and analyze the results.


print ("\n----------------> B\n")


# Analyze the class distribution of the 'Gender' feature
gender_distribution = df['Gender'].value_counts()

# Print the distribution
print(gender_distribution)

# Plot the distribution
plt.figure(figsize=(8, 6))
gender_distribution.plot(kind='bar', color=['blue', 'pink'])
plt.title('Gender Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# Analytical insights
total = gender_distribution.sum()
male_percentage = (gender_distribution['Male'] / total) * 100
female_percentage = (gender_distribution['Female'] / total) * 100

print(f"Male: {male_percentage:.2f}%")
print(f"Female: {female_percentage:.2f}%")

if abs(male_percentage - female_percentage) > 10:
    print("The dataset is imbalanced.")
else:
    print("The dataset is balanced.")




#Apply Python techniques to create new features from existing ones (feature engineering) and explain
#the significance of the new features.


print ("\n----------------> C\n")
# Create Age Group feature
def age_group(Age):
    if Age < 30:
        return 'Under 30'
    elif 30 <= Age < 40:
        return '30-39'
    elif 40 <= Age < 50:
        return '40-49'
    else:
        return '50 and above'

df['Age Group'] = df['Age'].apply(age_group)



# Create Income per Year of Experience feature
df['Income per Year'] = df['Monthly_Income'] / (df['Years_at_Company'] + 1)

# Create Is Manager feature
df['Is Manager'] = df['Job_Role'].apply(lambda x: 1 if 'Manager' in x else 0)



print(df.iloc[:, -3:])


#Age Group: Helps analyze attrition trends across different age groups.
#Income per Year of Experience: Normalizes income to identify potential disparities affecting attrition.
#Is Manager: Distinguishes managerial roles to evaluate their impact on attrition.
