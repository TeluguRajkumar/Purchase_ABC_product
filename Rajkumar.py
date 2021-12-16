#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 09:06:37 2021

@author: rajkumar
"""
#Import libraries
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

#!pip install imblearn

from sklearn.metrics import accuracy_score, f1_score,auc

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier
#Reading the dataset
df= pd.read_csv("/Users/rajkumar/Desktop/DSdataset.csv") 
df.shape
df.head()
df.info()
df.isna()
##Exploratory Data Analysis
fig, axes = plt.subplots(2, 2, figsize=(10, 10)) 
sns.countplot(ax=axes[0,0],x='Group',hue='Purchased_ABC_product',data=df,palette="mako") 
sns.countplot(ax=axes[0,1],x='Category',hue='Purchased_ABC_product',data=df,palette="mako") 
sns.countplot(ax=axes[1,0],x='Rating',hue='Purchased_ABC_product',data=df,palette="mako") 
sns.countplot(ax=axes[1,1],x='Customer_ID',hue='Purchased_ABC_product',data=df,palette="mako")
sns.countplot(x='Var1',hue='Purchased_ABC_product',data=df,palette="mako")
sns.countplot(x='Var2',hue='Purchased_ABC_product',data=df,palette="mako")
Purchased_ABC_product = df.loc[:,"Purchased_ABC_product"].value_counts().rename('Count')
plt.xlabel("Purchased_ABC_product")
plt.ylabel('Count')
sns.barplot(Purchased_ABC_product.index , Purchased_ABC_product.values,palette="mako")
sns.displot(df['Var1'])
sns.distplot(df['Var2'])

##Data preprocessing
def data_prep(df):

    df= df.drop(columns=['Customer_ID','Category','Rating'])

    df=pd.get_dummies(df,columns=['Var1'] ,prefix='Var')

    df=pd.get_dummies(df,columns=['Var2'] ,prefix='Var')
    df["Purchased_ABC_product"] = pd.cut(df['Purchased_ABC_product'], bins=[0, 1])

    df['Purchased_ABC_product']= df['Purchased_ABC_product'].cat.codes
    df.drop(columns=['Region_Code'],inplace= True)
    return df
df1=data_prep(df)
df1.head()
##Select Feature
Features= ['Customer_ID','Category','Var1','Var2','Rating']
#Train-Test split
from sklearn.model_selection import train_test_split 
X_train, X_test, Y_train, Y_test = train_test_split(df1[Features],df1['Purchased_ABC_product'],
                                   test_size = 0.3, random_state = 101) 
X_train.shape,X_test.shape
##Handle Imbalance Data Problem
from imblearn.under_sampling import RandomUnderSampler

RUS = RandomUnderSampler(sampling_strategy=.5,random_state=3,)

X_train,Y_train  = RUS.fit_resample(df1[Features],df1['Purchased_ABC_product'])
###Model training and prediction
def performance_met(model,X_train,Y_train,X_test,Y_test):

    acc_train=accuracy_score(Y_train, model.predict(X_train))

    f1_train=f1_score(Y_train, model.predict(X_train))

    acc_test=accuracy_score(Y_test, model.predict(X_test))

    f1_test=f1_score(Y_test, model.predict(X_test))

    print("train score: accuracy:{} f1:{}".format(acc_train,f1_train))

    print("test score: accuracy:{} f1:{}".format(acc_test,f1_test))

##Logistic Regression
model = LogisticRegression()
model.fit(X_train,Y_train) 
performance_met(model,X_train,Y_train,X_test,Y_test)

#Decision Tree
model_DT=DecisionTreeClassifier(random_state=1) 
model_DT.fit(X_train,Y_train) 
performance_met(model_DT,X_train,Y_train,X_test,Y_test)

##Random forest
Forest= RandomForestClassifier(random_state=1) 
Forest.fit(X_train,Y_train) 
performance_met(Forest,X_train,Y_train,X_test,Y_test)

##Hyperparameter tuning
rf= RandomForestClassifier(random_state=1)

parameters = {

    'bootstrap': [True],

'max_depth': [20, 25],

'min_samples_leaf': [3, 4],

'min_samples_split': [100,300],

}

grid_search_1 = GridSearchCV(rf, parameters, cv=3, verbose=2, n_jobs=-1)

grid_search_1.fit(X_train, Y_train)

performance_met(grid_search_1,X_train,Y_train,X_test,Y_test)

#Code to Calculate SHAP Values
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("/Users/rajkumar/Desktop/DSdataset.csv")
y = (data['Purchased_ABC_product'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

my_model.predict_proba(data_for_prediction_array)
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)


























