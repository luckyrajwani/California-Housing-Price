#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


# Step 1: Data Collection
data = pd.read_csv('california_housing.csv')


# In[3]:


# Step 2: Data Exploration
print(data.head())  # Display the first few rows of the dataset
print(data.shape)  # Check the dimensions of the dataset
print(data.info())  # Get information about the dataset, including data types and missing values


# In[4]:


# Step 3: Data Cleaning
data = data.dropna()  # Drop rows with missing values
data = data.reset_index(drop=True)  # Reset index after dropping rows


# In[5]:


# Step 4: Descriptive Statistics
print(data.describe())  # Calculate summary statistics for numerical variables


# In[6]:


# Step 5: Data Visualization
sns.histplot(data['median_house_value'], kde=True)  # Histogram of median house value
plt.xlabel('Median House Value')
plt.ylabel('Count')
plt.title('Distribution of Median House Value')
plt.show()


# In[7]:


sns.scatterplot(x='total_rooms', y='median_house_value', data=data)  # Scatter plot of total rooms vs. median house value
plt.xlabel('Total Rooms')
plt.ylabel('Median House Value')
plt.title('Relationship between Total Rooms and Median House Value')
plt.show()


# In[8]:


# Step 6: Feature Engineering (Example: Creating a new feature)
data['avg_rooms_per_household'] = data['total_rooms'] / data['households']


# In[9]:


# Step 7: Correlation Analysis
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)  # Heatmap of correlation matrix
plt.title('Correlation Matrix')
plt.show()


# In[10]:


# Step 8: Hypothesis Testing (Example: T-test for mean house values in different districts)
districts = data['ocean_proximity'].unique()
for district in districts:
    subset = data[data['ocean_proximity'] == district]
    print(f"Mean house value in {district}: {subset['median_house_value'].mean()}")


# In[11]:


# Step 9: Key Findings (Example: Top 5 districts with highest median house values)
top_districts = data.nlargest(5, 'median_house_value')
print("Top 5 districts with highest median house values:")
print(top_districts[['ocean_proximity', 'median_house_value']])


# In[12]:


# Step 10: Conclusion
print("EDA project on California Housing Prices dataset is complete.")


# In[ ]:




