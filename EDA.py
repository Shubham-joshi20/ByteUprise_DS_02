
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the Titanic dataset
df = pd.read_excel(r'C:\Users\Shubm\OneDrive\Desktop\ByteUprise_DS_02\titanic3.xls')
# View the first few rows of the dataset
print(df.head())

# Get information about the dataset
print(df.info())

# Statistical summary of numerical columns
print(df.describe())

# Count missing values in each column
print(df.isnull().sum())
# Fill missing values 
df['age'] = df['age'].fillna(df['age'].mean())

# Drop rows with missing values 
df.dropna(subset=['embarked'], inplace=True)


# Scatter plot of age vs fare with hue based on Survival
sns.scatterplot(x='age', y='fare', hue='survived', data=df, palette='Set1')
plt.title('Age vs Fare with Survival')
plt.show()

# Bar chart of Survival counts by pclass
sns.countplot(x='pclass', hue='survived', data=df, palette='Set2')
plt.title('Survival Counts by Passenger Class')
plt.show()

# Bar chart of female and male survived 
sns.countplot(x='sex', hue='survived', data=df, palette='Set3')
plt.title("Sex vs Survived")
plt.show()

# Age distribution by passenger class
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='age', data=df, palette='Set1')
plt.title('Age Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Age')
plt.show()

# Passenger count by embarked port
sns.countplot(x='embarked', data=df, palette='Set2')
plt.title('Passenger Count by Embarked Port')
plt.xlabel('Embarked Port')
plt.ylabel('Passenger Count')
plt.show()

# Survival rates by Sex
survival_by_sex = df.groupby('sex')['survived'].mean()
print(survival_by_sex)

# Fare distribution by passenger class
plt.figure(figsize=(10, 6))
sns.boxplot(x='pclass', y='fare', data=df, palette='Set3')
plt.title('Fare Distribution by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Fare')
plt.show()

# Create a new feature 'FamilySize' combining SibSp and Parch
df['FamilySize'] = df['sibsp'] + df['parch'] + 1

# Summary of survival rates by pclass
survival_by_pclass = df.groupby('pclass')['survived'].mean()
print(survival_by_pclass)

# Plotting survival rates by pclass  
survival_by_pclass.plot(kind='bar', color=['skyblue', 'salmon', 'lightgreen'])
plt.title('Survival Rates by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Plotting pie chart
gender_counts = df['survived'].value_counts()
labels = ['Not Survived', 'Survived']  # Updated labels
colors = ['lightcoral', 'lightblue']  # Different colors
plt.pie(gender_counts.values, labels=labels, colors=colors, autopct='%1.1f%%')
plt.title('Survived')
plt.show()
