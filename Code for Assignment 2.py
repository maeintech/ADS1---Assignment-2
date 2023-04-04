#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 15:08:48 2023

@author: Mae
"""
import numpy as np
import pandas as pd

def skew(dist):
    """ Calculates the centralised and normalised skewness of dist. """
    
    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)
    
    # now calculate the skewness
    value = np.sum(((dist-aver) / std)**3) / len(dist-1)
    
    return value


def kurtosis(dist):
    """ Calculates the centralised and normalised excess kurtosis of dist. """
    
    # calculates average and std, dev for centralising and normalising
    aver = np.mean(dist)
    std = np.std(dist)
    
    # now calculate the kurtosis
    value = np.sum(((dist-aver) / std)**4) / len(dist-1) - 3.0
    
    return value


# import data from excel file
def read(x, y):
    """
    Reads and imports files from excel spreadsheet to a python DataFrame

    Arguments:
    x: string, The name of the excel file which is to be read
    y: integer, indicates the number of rows on the excel file to be
    skipped
    z: string, The name of the column header for the column to be used as the index

    Returns:
    data: A pandas dataframe with all values from the excel file
    data_t: The transposed pandas dataframe
    """
    data = pd.read_excel(x, skiprows= y)
    data_transpose = data.transpose()
    return data, data_transpose

x = 'API_19_DS2_en_excel_v2_5252304.xls'
y = 3


data, data_transpose = read(x, y)