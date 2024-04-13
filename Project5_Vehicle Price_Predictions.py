#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMPORTING LIBRARIES


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD


# In[3]:


# IMPORTING THE DATA SET


# In[4]:


# use the link in the project description on the website to find the dataset on Kaggle. It is too big to upload on GitHub
file_path = "/Users/lassewillen/Downloads/car_prices.csv"
car_data = pd.read_csv(file_path)
car_data.head(3)


# In[5]:


car_data.describe().round()


# In[6]:


# DATA CLEANING


# In[7]:


#dropping missing values 
# no duplicate removal as it is highly plausbile that duplicates exist in such a dataset
car_data.dropna(inplace=True)


# In[8]:


car_data.isna().sum()


# In[9]:


from datetime import datetime

# Step 1: Removing rows where 'make', 'model', or 'vin' is missing
car_data = car_data.dropna(subset=['make', 'model', 'vin'])

# Step 2: Converting 'saledate' to datetime format
car_data['saledate'] = pd.to_datetime(car_data['saledate'], errors='coerce')

# Step 3: Removing rows with missing values in 'odometer', 'mmr', and 'sellingprice'
car_data = car_data.dropna(subset=['odometer', 'mmr', 'sellingprice'])


# In[10]:


# Further cleaning steps

# Normalizing textual data: Converting 'make', 'model', 'trim', 'body', 'color', and 'interior' to lowercase
text_columns = ['make', 'model', 'trim', 'body', 'color', 'interior']
car_data[text_columns] = car_data[text_columns].apply(lambda x: x.str.lower())

# Handling 'transmission' missing values by categorizing them as 'unknown'
car_data['transmission'] = car_data['transmission'].fillna('unknown')

# Imputing 'condition' missing values with the median of the 'condition' column
condition_median = car_data['condition'].median()
car_data['condition'] = car_data['condition'].fillna(condition_median)

# Final check on the current state of missing values after further cleaning
final_missing_values_summary = car_data.isnull().sum()

# Displaying the summary of cleaned data
final_cleaned_summary = {
    "Cleaned rows after further steps": car_data.shape[0],
    "Current missing values after further steps": final_missing_values_summary
}

final_cleaned_summary


# In[11]:


# Import necessary libraries
from scipy import stats
# Define a function to detect and count outliers for each numerical column using the IQR method
def count_outliers_iqr(dataframe, columns):
    outlier_counts = {}
    for column in columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
        outlier_counts[column] = len(outliers)
    return outlier_counts

# List of numerical columns to check for outliers
numerical_columns = ['year', 'condition', 'odometer', 'mmr', 'sellingprice']

# Count outliers in each numerical column of car_data
outlier_counts = count_outliers_iqr(car_data, numerical_columns)

outlier_counts



# In[12]:


# Define a function to calculate the value range of outliers for each numerical column using the IQR method
def outlier_value_range(dataframe, columns):
    outlier_ranges = {}
    for column in columns:
        q1 = dataframe[column].quantile(0.25)
        q3 = dataframe[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        outliers = dataframe[(dataframe[column] < lower_bound) | (dataframe[column] > upper_bound)]
        if not outliers.empty:
            min_outlier = outliers[column].min()
            max_outlier = outliers[column].max()
        else:
            min_outlier = max_outlier = None
        outlier_ranges[column] = (min_outlier, max_outlier)
    return outlier_ranges

# Calculate the value range of outliers for each numerical column in car_data
outlier_value_ranges = outlier_value_range(car_data, numerical_columns)

outlier_value_ranges


# In[13]:


# outliers are not ectreme instead of the odometers
# Set the threshold for extreme odometer readings. Delete these rows
odometer_threshold = 500000
# outliers for prices are okay, odometer is not realistic. Therefore delete rows with odometer ratings
# Correcting the DataFrame name to car_data and removing rows with odometer readings above the threshold
car_data = car_data[car_data['odometer'] <= odometer_threshold]

# Verify the correction by checking the shape of the cleaned dataset
car_data.shape


# In[14]:


# SUPERVISED ALGORITHM


# In[15]:


y = car_data['sellingprice']

feature_columns = ['year', 'make', 'model', 'odometer', 'condition']
X = car_data[feature_columns] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['make', 'model']
numerical_features = ['year', 'odometer', 'condition']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])


model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(y_pred[:5])  
print(f'Linear Regression MSE: {mean_squared_error(y_test, y_pred)}')
print(f'Linear Regression R² score: {r2_score(y_test, y_pred)}')


# In[16]:


# The model's predictions have an average squared error of 41.1 million, indicating substantial deviations from actual selling prices.
# With an R² score of 0.547, the model explains about 54.7% of the variance in car selling prices from the selected features.


# In[17]:


#update the model pipeline to use a random forrest regressor 
y = car_data['sellingprice']
X = car_data.drop(['sellingprice', 'vin', 'saledate'], axis=1)

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)
print(y_pred[:5]) 
print(f'Random Forest Regression MSE: {mean_squared_error(y_test, y_pred)}')
print(f'Random Forest Regression R² score: {r2_score(y_test, y_pred)}')


# The Random Forest model significantly improves prediction accuracy with an MSE of ~2.38 million and explains ~97.38% of the variance in car selling prices (R² score).\n# Predicted selling prices for the first five cars in the test set range from approximately $4,049 to $22,966, indicating varied price predictions across the dataset.

# In[19]:


# UNSUPERVISED ALGORTHM 
# to assess car data and cluster it based on characteristics not including sellingprice


# In[20]:


y = car_data['sellingprice']
X = car_data.drop('sellingprice', axis=1)

# Preprocessing
categorical_features = ['make', 'model']
numerical_features = ['year', 'odometer', 'condition']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Elbow Method to determine number of clusters
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('cluster', kmeans)])
    pipeline.fit(X)
    sse.append(kmeans.inertia_)

plt.plot(range(1, 11), sse)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('SSE')
plt.show()

pipeline.fit(X)


# In[21]:


# optimal_k determined based on Elbow method
optimal_k = 4  
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('cluster', kmeans)])


# In[22]:


# Prepare the features by dropping the target variable 'sellingprice'
X = car_data.drop('sellingprice', axis=1)

# Define which features are categorical and which are numerical
categorical_features = ['make', 'model']
numerical_features = ['year', 'odometer', 'condition']

# Create a preprocessor that scales numerical features and encodes categorical features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),  # Scale numerical features to have mean=0 and variance=1
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # Convert categorical features into one-hot vectors
    ])

# Create a pipeline that preprocesses the data and then applies KMeans clustering
pipeline = Pipeline([
    ('preprocessor', preprocessor),  # First, preprocess the data
    ('cluster', KMeans(n_clusters=4, random_state=42))  # Then cluster it into 4 groups
])

# Fit the pipeline to the data
pipeline.fit(X)

# Use TruncatedSVD to reduce the dimensionality of the preprocessed data to 2D for visualization
svd = TruncatedSVD(n_components=2, random_state=42)
X_reduced = svd.fit_transform(pipeline.named_steps['preprocessor'].transform(X))  # Transform the data to 2D

# Visualize the clusters
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=pipeline.named_steps['cluster'].labels_, cmap='viridis')  # Scatter plot of the two principal components colored by cluster
plt.title('2D Visualization of Car Data Clusters')  # Title of the plot
plt.xlabel('Component 1')  # X-axis label
plt.ylabel('Component 2')  # Y-axis label
plt.colorbar(label='Cluster Label')  # Color bar to show cluster labels
plt.show()  # Display the plot


# In[ ]:




