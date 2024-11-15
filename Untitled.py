#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Provide a valid file path for your dataset
dataset = pd.read_csv("Advertising.csv")

# Display the first 10 rows of the dataset
print(dataset.head(10))

# Display the shape of the dataset
print(dataset.shape)

# Check for missing values
print(dataset.isna().sum())

# Check for duplicate rows
print(dataset.duplicated().any())

# Create subplots for boxplots
fig, axs = plt.subplots(3, figsize=(5, 5))
plt1 = sns.boxplot(dataset['TV'], ax=axs[0])
plt2 = sns.boxplot(dataset['Newspaper'], ax=axs[1])
plt3 = sns.boxplot(dataset['Radio'], ax=axs[2])
plt.tight_layout()

# Distribution plot for 'sales'
sns.displot(dataset['Sales'])
plt.show()

# Pairplot for selected variables
sns.pairplot(dataset, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', height=4, aspect=1, kind='scatter')
plt.show()

# Heatmap for correlation matrix
sns.heatmap(dataset.corr(), annot=True)
plt.show()

# Define features and target variable
X = dataset[['TV']]
y = dataset['Sales']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Create and train the linear regression model
slr = LinearRegression()
slr.fit(X_train, y_train)

# Print the intercept and coefficient
print('Intercept:', slr.intercept_)
print('Coefficient:', slr.coef_)

# Plot the regression line
plt.scatter(X_train, y_train)
plt.plot(X_train, slr.intercept_ + slr.coef_ * X_train, 'r')
plt.show()

# Make predictions on the test set
y_pred_slr = slr.predict(X_test)
print("Prediction for test set:", y_pred_slr)

# Calculate and print the R squared value
print('R squared value of the model: {:.2f}'.format(slr.score(X, y) * 100))


# In[9]:


import pandas as pd
data={'Name':['John','Emma','Sort','lisa','Tom'],
     'Age':[25,30,28,32,27],
     'Country':['USA','Canada','India','UK','Australia'],
      'Salary':[50000,60000,70000,80000,65000]
     }
df=pd.DataFrame(data)
print("original Dataframe")
print(df)
name_age=df[['Name','Age']]
print("Name and Age columns")
print(name_age)
filtered_df=df[df['Country']=='USA']
print("\n filtered DataFrame(country='USA')")
print(filtered_df)
sorted_df=df.sort_values("Salary", ascending =False)
print("\n sorted DataFrame(by salary in descending order)")
print(sorted_df)
average_Salary=df['Salary'].mean()
print("\n Average Salary",average_Salary)
df['Experience']=[3,6,4,8,5]
print("\n DataFrame with added experience")
print(df)
df.loc[df['Name']=='Emma','Salary']=65000
print("\n DataFrame with updating emma salary")
print(df)
df = df.drop('Experience', axis=1)
print("\nDataFrame after deleting the Experience column")
print(df)


# In[ ]:




