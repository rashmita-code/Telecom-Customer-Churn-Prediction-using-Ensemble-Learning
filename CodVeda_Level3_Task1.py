#!/usr/bin/env python
# coding: utf-8

# # Telecom Customer Churn Prediction using Ensemble Learning
#   
# ## Model: Random Forest Classifier  
# 
# ---
# 
# ## Project Objective
# ### To develop a robust classification model that predicts customer churn by:
# - Optimizing model parameters
# - Evaluating performance using multiple metrics
# - Identifying key influencing features
# 
# ---
# 
# ## Approach
# ### This project leverages:
# - Pre-divided dataset (train & test)
# - Ensemble learning techniques
# - Statistical validation (cross-validation)
# 

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RandomizedSearchCV, cross_validate
from sklearn.metrics import confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Reading datasets
df_train = pd.read_csv("churn-bigml-80.csv")
df_test  = pd.read_csv("churn-bigml-20.csv")

print(f"Training Data: {df_train.shape}")
print(f"Testing Data: {df_test.shape}")


# In[3]:


df_train.describe(include='all')


# ## Merging datasets to ensure uniform transformations

# In[4]:


data = pd.concat([df_train, df_test], ignore_index=True)


# ## Data Preparation

# In[5]:


# Handling duplicates
data.drop_duplicates(inplace=True)

# Fill missing values smartly
for column in data.columns:
    if data[column].dtype == 'object':
        data[column].fillna(data[column].value_counts().idxmax(), inplace=True)
    else:
        data[column].fillna(data[column].median(), inplace=True)


# ## Transforming Categorical Variables

# In[6]:


# Binary conversion
binary_map = {'Yes': 1, 'No': 0}
data.replace(binary_map, inplace=True)

# Encoding categorical variables
encoder = LabelEncoder()

categorical_cols = data.select_dtypes(include='object').columns

for col in categorical_cols:
    data[col] = encoder.fit_transform(data[col])


# ## Define Inputs & Output

# In[7]:


target = 'Churn'

features = data.drop(columns=[target])
labels = data[target]


# ## Train-Test Split

# In[8]:


split_index = len(df_train)

X_train = features.iloc[:split_index]
X_test  = features.iloc[split_index:]

y_train = labels.iloc[:split_index]
y_test  = labels.iloc[split_index:]


# ## Model Setup

# In[9]:


model = RandomForestClassifier(
    random_state=42,
    class_weight='balanced_subsample'
)


# ## Parameter Search using Random Sampling

# In[10]:


param_grid = {
    'n_estimators': np.arange(100, 600, 100),
    'max_depth': [None, 15, 25, 35],
    'min_samples_split': [2, 4, 8],
    'min_samples_leaf': [1, 2, 3]
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_grid,
    n_iter=15,
    cv=4,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

optimized_model = search.best_estimator_


# ## Validation Scores

# In[11]:


cv_results = cross_validate(
    optimized_model,
    X_train,
    y_train,
    cv=5,
    scoring=['accuracy', 'f1']
)

print("Average Accuracy:", cv_results['test_accuracy'].mean())
print("Average F1 Score:", cv_results['test_f1'].mean())


# ## Predictions

# In[12]:


predictions = optimized_model.predict(X_test)


# ## Performance Report

# In[13]:


print(classification_report(y_test, predictions))


# ## Confusion Matrix

# In[14]:


cmatrix = confusion_matrix(y_test, predictions)

plt.figure(figsize=(5,5))
sns.heatmap(cmatrix, annot=True, fmt='d', cmap='coolwarm', cbar=False)
plt.title("Prediction Confusion Matrix")
plt.show()


# ## Feature Importance

# In[15]:


importance = pd.DataFrame({
    'Feature': features.columns,
    'Score': optimized_model.feature_importances_
}).sort_values(by='Score', ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x='Score', y='Feature', data=importance.head(10))
plt.title("Top Influential Features")
plt.show()


# ## Observations
# 
# - Ensemble learning significantly boosts predictive performance.
# - Hyperparameter tuning refines model efficiency.
# - Key features strongly influence churn behavior.
# - Model generalizes well due to cross-validation.
# 
# ---
# 
# ## Summary
# 
# This implementation:
# 
# 1 Uses optimized ensemble learning  
# 2 Applies structured preprocessing  
# 3 Validates performance rigorously  
# 4 Provides business insights  
# 

# In[ ]:




