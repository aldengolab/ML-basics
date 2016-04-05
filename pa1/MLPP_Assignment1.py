# Machine Learning for Public Policy
# Assignment 1: Problem A
# 3/30/16
# ALDEN GOLAB

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import urllib2
import json
import copy

FIGURE_DIR = 'Figures'

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

    for summ in summaries:
        summ[2].to_csv('{}/{}_summary.csv'.format(FIGURE_DIR, summ[1]))

def plot_distributions(dataframe):
    '''
    Plots distributions for each column and saves to file.
    '''
    for col in dataframe.columns:
        if type(col) == str:
            dataframe[col].value_counts().plot(kind = 'bar')
            plt.suptitle('Bar Graph for {}'.format(col), fontsize = 14)
            plt.savefig('{}/{}.png'.format(FIGURE_DIR, col))
            print('Saved figure as {}.png'.format(col))
        else:
            dataframe[col].hist()
            plt.suptitle('Histogram for {}'.format(col), fontsize = 14)
            plt.savefig('{}/{}.png'.format(FIGURE_DIR, col))
            print('Saved figure as {}.png'.format(col))

def get_gender(dataframe):
    '''
    Uses genderize.io API to get genders based on first name.
    '''
    base_url = 'https://api.genderize.io/?'
    urls = []
    names = set([])
    to_add = []
    count = 0
    genders = None
    
    for row in dataframe[dataframe['Gender'].isnull()].iterrows():
        names.add('{}'.format(row[1]['First_name']))
    for name in names: 
        count += 1
        to_add.append('name[{}]={}'.format(count, name))
        if count == 9:
            urls.append(base_url + '&'.join(to_add))
            count = 0
            to_add = []
    if len(to_add) > 1:
        urls.append(base_url + '&'.join(to_add))
    if len(to_add) == 1:
        urls.append(base_url + 'name=' + to_add[0][8:])

    print('Running names through genderize API...')

    for url in urls:
        json_file = urllib2.urlopen(url)
        if type(genders) == type(None):
            genders = pd.DataFrame(json.load(json_file))
        else: 
            df = pd.DataFrame(json.load(json_file))
            genders = pd.concat([genders, df])

    genders.replace('female', 'Female')
    genders.replace('male', 'Male')
    genders.index = list(range(len(genders)))

    print('Replacing missing gender...')
    
    for row in dataframe[dataframe['Gender'].isnull()].iterrows():
        gender_index = genders[genders['name'] == row[1]['First_name']].index[0]
        value = genders.loc[gender_index, 'gender']
        dataframe.loc[row[0], 'Gender'] = value

    print('Missing gender replaced.')

def run_missing_A(data, filename):
    '''
    Fills in missing values for Age, GPA, and Days_missed with mean and 
    writes file to csv. 
    '''
    dataframe = copy.deepcopy(data)

    for row in dataframe[dataframe['Age'].isnull()].iterrows():
        dataframe.loc[row[0], 'Age'] = dataframe['Age'].mean()

    for row in dataframe[dataframe['GPA'].isnull()].iterrows():
        dataframe.loc[row[0], 'GPA'] = dataframe['GPA'].mean()

    for row in dataframe[dataframe['Days_missed'].isnull()].iterrows():
        dataframe.loc[row[0], 'Days_missed'] = dataframe['Days_missed'].mean()

    dataframe.to_csv(filename[:-4] + '_methodA.csv')
    print('Wrote Method A to file')

def run_missing_B(data, filename):
    '''
    Fills in missing values for Age, GPA, and Days_missed with class-
    conditional mean and writes file to csv. 
    '''
    dataframe = copy.deepcopy(data)
    
    for row in dataframe[dataframe['Age'].isnull()].iterrows():
        dataframe.loc[row[0], 'Age'] = dataframe['Age'][dataframe['Graduated']\
         == row[1]['Graduated']].mean()

    for row in dataframe[dataframe['GPA'].isnull()].iterrows():
        dataframe.loc[row[0], 'GPA'] = dataframe['GPA'][dataframe['Graduated']\
         == row[1]['Graduated']].mean()

    for row in dataframe[dataframe['Days_missed'].isnull()].iterrows():
        dataframe.loc[row[0], 'Days_missed'] = dataframe['Days_missed']\
        [dataframe['Graduated'] == row[1]['Graduated']].mean()

    dataframe.to_csv(filename[:-4] + '_methodB.csv')
    print('Wrote Method B to file')

def run_missing_C(data, filename):
    '''
    Imputes missing data using simple normal probabilistic imputation. Assumes
    a normal distribution for ease of implementation; Bayesian would be a 
    significant improvement.
    '''
    dataframe = copy.deepcopy(data)

    for row in dataframe[dataframe['Age'].isnull()].iterrows():
        dataframe.loc[row[0], 'Age'] = np.random.normal(dataframe['Age'].mean(), 
            dataframe['Age'].std())

    for row in dataframe[dataframe['GPA'].isnull()].iterrows():
        dataframe.loc[row[0], 'GPA'] = np.random.normal(dataframe['GPA'].mean(), 
            dataframe['GPA'].std())

    for row in dataframe[dataframe['Days_missed'].isnull()].iterrows():
        dataframe.loc[row[0], 'Days_missed'] = np.random.normal(dataframe[\
            'Days_missed'].mean(), dataframe['Days_missed'].std())

    dataframe.to_csv(filename[:-4] + '_methodC.csv')
    print('Wrote Method C to file')

def run_assignment(filename):
    '''
    Runs all of Problem 1 from the assignment. 
    '''
    data = load_file(filename)
    get_summary_stats(data)
    plot_distributions(data)
    get_gender(data)
    run_missing_A(data, filename)
    run_missing_B(data, filename)
    run_missing_C(data, filename)



