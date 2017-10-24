#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:27:39 2017

@author: alexkirwan
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer


def main():
    imputer = Imputer(missing_values = 'NaN', strategy='mean', axis=0)
    
    # Read dataframes
    df = pd.read_csv("./train.csv")
    tdf = pd.read_csv("./test.csv")
    
    # Remove fetures that do not contribute 
    # to the prediction of the final class.
    df = df.drop(["Name", "Cabin", "Ticket"],1)
    tdf = tdf.drop(["Name", "Cabin", "Ticket"], 1)
    
    imputer.fit( df[['Age', 'Fare']] )
    df['Age'] = imputer.transform( df[['Age', 'Fare']] )
    df.dropna(axis=0,subset=['Embarked'])

    print(df)
    
    
    return

main()