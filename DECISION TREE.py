# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from pandas.api.types import is_string_dtype
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


cwd = os.getcwd()

data = pd.read_csv("C://Users//User//Downloads//Pre-processing and machine learning notes and code files/DT & RF sample//Automobile_data.csv")
data

data.replace('?',np.nan, inplace =True) 

string_col = data.select_dtypes(exclude = np.number).columns.tolist()

string_col

num_cols = ["normalized-losses", "bore", "stroke", "horsepower", "peak-rpm","price"]

#convert into numeric data types
for i in num_cols:
    data[i]= pd.to_numeric(data[i],errors = "raise")

#catogorical conversion
for i in data:
    if is_string_dtype(data[i]):
        data[i]= data[i].astype("category").cat.as_unordered() 

#categorical code conversion
for i in data:
    if (str(data[i].dtype) == 'category'):
        data[i] = data[i].cat.codes 

#imputation
data.fillna(data.median(),inplace = True)

#modeling
X = data.drop('symboling', axis =1)
Y = data['symboling']

#Train_test_split
x_train, x_val, y_train, y_val = train_test_split(X,Y, test_size = 0.2, random_state=100)

#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(x_train,y_train)

print(dt.score(x_val,y_val))
accuracy_score(x_val,y_val)



lr = LogisticRegression()
lr.fit(x_train, y_train)







