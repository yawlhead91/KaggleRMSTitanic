#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:27:39 2017

@author: alexkirwan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer

def preProccess(df, tdf):
    # Imputation transformer for completing missing values.
    imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
    
    # Remove fetures that do not contribute 
    # to the prediction of the final class.
    df = df.drop(["Name", "Cabin", "Ticket"],1)
    tdf = tdf.drop(["Name", "Cabin", "Ticket"], 1)
    
    imputer.fit( df[['Age', 'Fare']] )
    df['Age'] = imputer.transform( df[['Age', 'Fare']] )

    df.dropna(axis=0,subset=['Embarked'])
    
    # Encoding Categorical Variable
    sex_mapping = {'male':0, 'female':1}
    df['Sex'] = df['Sex'].map(sex_mapping)
    tdf['Sex'] = df['Sex'].map(sex_mapping)
    
    emb_mapping = {'S': 0, 'C': 1, 'Q': 2} 
    df['Embarked'] = df['Embarked'].map(emb_mapping)
    tdf['Embarked'] = df['Embarked'].map(emb_mapping)
    
    return df, tdf



def main():
    # Read dataframes
    df = pd.read_csv("./train.csv")
    tdf = pd.read_csv("./test.csv")
    
    df, tdf = preProccess(df, tdf)
    
    dft = df['Survived']
    df = df.drop(['Survived'], 1)
    
    dfid = df['PassengerId']
    df = df.drop(['PassengerId'], 1)
    
    tdfid = tdf['PassengerId']
    tdf = tdf.drop(['PassengerId'], 1)
    
    
    
    print(dft)
    
    return

main()