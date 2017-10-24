#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 14:27:39 2017

@author: alexkirwan
"""

import numpy as np
import pandas as pd


def main():
    # Read dataframes
    traind = pd.read_csv("./train.csv")
    testd = pd.read_csv("./test.csv")
    
    # Remove fetures that do not contribute 
    # to the prediction of the final class.
    traind.drop(["Name", "Cabin", "Ticket"], axis=1)
    testd.drop(["Name", "Cabin", "Ticket"], axis=1)
    
    
    return

main()