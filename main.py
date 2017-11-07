#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:27:39 2017

@author: alexkirwan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics

def preProccess(df):
    # Imputation transformer for completing missing values.
    imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
    scaler = MinMaxScaler()
    
    # Remove fetures that do not contribute 
    # to the prediction of the final class.
    df = df.drop(["Name", "Cabin", "Ticket"],1)
    
    imputer.fit( df[['Age', 'Fare']] )
    tf = imputer.transform( df[['Age', 'Fare']] )
    df['Age'] = tf[:, 0]
    df['Fare']= tf[:, 1]
    
    df = df.dropna(axis=0, subset=['Embarked'], how='all')
    
    # Encoding Categorical Variable
    sex_mapping = {'male':0, 'female':1}
    df['Sex'] = df['Sex'].map(sex_mapping)
    
    emb_mapping = {'S': 0, 'C': 1, 'Q': 2} 
    df['Embarked'] = df['Embarked'].map(emb_mapping)
    
    df[["Age", "SibSp", "Parch", "Fare", "Pclass"]] = scaler.fit_transform(df[["Age", "SibSp", "Parch", "Fare", "Pclass"]])
    return df



def main():
    # Read dataframes
    df = pd.read_csv("./train.csv")
    tdf = pd.read_csv("./test.csv")
    
    df = preProccess(df)
    tdf = preProccess(tdf)
    #print(df)
    dfc = df['Survived']
    df = df.drop(['Survived'], 1)
    
    dfid = df['PassengerId']
    df = df.drop(['PassengerId'], 1)
    
    tdfid = tdf['PassengerId']
    tdf = tdf.drop(['PassengerId'], 1)
    
    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(df, dfc) 
    
    results = neigh.predict(tdf)
    
    resultSeries = pd.Series(data = results, name = 'Survived', dtype='int64')
    df = pd.DataFrame({"PassengerId":tdfid, "Survived":resultSeries})
    df.to_csv("submission.csv", index=False, header=True)
    
    return

main()