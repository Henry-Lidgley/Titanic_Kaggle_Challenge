# Titanic_Kaggle_Challenge
Submission of predictions using XGBoost in Python for the Kaggle Titantic Challenge.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 20:13:30 2018

@author: henrylidgley
"""

# XGBoost on Titanic Kaggle Competition

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

y_train = X_train.iloc[:, 1].values
submission = X_test.iloc[:, 0].values
submission = pd.DataFrame(submission) 
submission.columns = ['PassengerId'] 

# Delete redundant features
X_train = X_train.drop(X_train.columns[[1, 3, 8, 10]], axis=1)
X_test = X_test.drop(X_test.columns[[2, 7, 9]], axis=1)

# Create train dummy variables for categorical features
dummies = pd.get_dummies(X_train['Pclass']) 
dummies.columns = ['1st Class', '2nd Class', '3rd Class'] 
#dummies = dummies.drop(dummies.columns[0], axis=1)
X_train = X_train.drop('Pclass', axis=1)
X_train = X_train.join(dummies)   

dummies = pd.get_dummies(X_train['Sex']) 
#dummies = dummies.drop(dummies.columns[0], axis=1)
X_train = X_train.drop('Sex', axis=1)
X_train = X_train.join(dummies) 

dummies = pd.get_dummies(X_train['Embarked']) 
#dummies = dummies.drop(dummies.columns[0], axis=1)
X_train = X_train.drop('Embarked', axis=1)
X_train = X_train.join(dummies) 

# Create test dummy variables for categorical features
dummies = pd.get_dummies(X_test['Pclass']) 
dummies.columns = ['1st Class', '2nd Class', '3rd Class'] 
#dummies = dummies.drop(dummies.columns[0], axis=1)
X_test = X_test.drop('Pclass', axis=1)
X_test = X_test.join(dummies)   

dummies = pd.get_dummies(X_test['Sex']) 
#dummies = dummies.drop(dummies.columns[0], axis=1)
X_test = X_test.drop('Sex', axis=1)
X_test = X_test.join(dummies) 

dummies = pd.get_dummies(X_test['Embarked']) 
#dummies = dummies.drop(dummies.columns[0], axis=1)
X_test = X_test.drop('Embarked', axis=1)
X_test = X_test.join(dummies) 

# Fitting XGBoost to the Training set
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred) 
y_pred.columns = ['Survived'] 
submission = submission.join(y_pred) 

# Exporting dataset to csv
submission.to_csv('Lidgley_Titanic_Submission.csv', index=False, sep=',')
