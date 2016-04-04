# Machine Learning for Public Policy
# Assignment 1
# 3/30/16
# ALDEN GOLAB

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def load_file(filename):
    '''
    Reads csv file with column index of 'ID' and returns a Pandas dataframe.
    '''
    return pd.read_csv(filename, index_col = 'ID')

def get_summary_stats(dataframe):
    '''
    Takes dataframe and returns summary statistics for each numerical column 
    as a pandas dataframe.
    '''
    summaries = []
    mode_nums = []
    for col in dataframe.columns:
        if dataframe[col].dtype != object and col != 'ID':
            summary = dataframe[col].describe()
            # Get mode and append
            mode = pd.Series(dataframe[col].mode())
            mode = mode.rename({i: 'mode' + str(i) for i in range(len(mode))})
            summary = summary.append(mode)
            # Get median and append
            median = pd.Series(dataframe[col].median())
            median = median.rename({median.index[0]: 'median'})
            summary = summary.append(median)
            # Count missing values and append
            missing = pd.Series(dataframe[col].isnull().sum())
            missing = missing.rename({missing.index[0]: 'missing'})
            summary = summary.append(missing)
            # Append to list
            summaries.append(('float64', col, summary))
        else:
            summary = dataframe[col].describe()
            # Count missing values and append
            missing = pd.Series(dataframe[col].isnull().sum())
            missing = missing.rename({missing.index[0]: 'missing'})
            summary = summary.append(missing)
            # Append to list
            summaries.append(('object', col, summary))            

    return summaries

def plot_distributions(dataframe):
    '''
    Plots distributions for each column and saves to file.
    '''
    for col in dataframe.columns:
        if dtype(col) == object:
            bar = dataframe[col].value_counts().plot(kind = 'bar')
        else:
            fig = dataframe[col].hist()
            fig.suptitle('Histogram for {}'.format(col), fontsize = 14)
    pass

def run_assignment(filename):
    '''
    '''


