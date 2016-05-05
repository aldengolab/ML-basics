# CAPP 30254: Machine Learning for Public Policy
# ALDEN GOLAB
# ML Pipeline
# 
# Data processing functions.

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

def impute(data, column, method = 'mean', classification = None, 
    distribution = None, write = False):
    '''
    Runs imputation for data, given a particular method and column. Default
    will run mean imputation. If distribution is not selected, will run
    probabilistic with normal distribution. Requires classification for 
    conditional mean imputation. Writes imputed dataframe to csv.
    '''
    if method == 'mean':
        data = impute_mean(data, column)
    elif method == 'conditional':
        if classification == None: 
            raise ValueError('Classification needed for conditional imputation.')
        else:
            data = impute_cond(data, column, classification)
    elif method == 'probabilistic':
        if distribution == None:
            data = impute_prob(data, column, dist = 'normal')
        else: 
            data = impute_prob(data, column, dist = distribution)

    if write == True:
        new_file = filename[:-4] + '_imputed.csv'
        data.to_csv(new_file)
        print('Wrote data with imputation to {}'.format(new_file))
    
    return data

def impute_mean(data, column):
    '''
    Generalized mean imputation.

    Inputs: pandas dataframe, column to impute into
    '''
    dataframe = copy.deepcopy(data)

    for row in dataframe[dataframe[column].isnull()].iterrows():
        dataframe.loc[row[0], column] = dataframe[column].mean()

    return dataframe

def impute_cond(data, column, classification):
    '''
    Genderalized conditional mean imputation.

    Inputs: pandas dataframe, column to impute into, classification to impute on
    '''
    dataframe = copy.deepcopy(data)
    
    for row in dataframe[dataframe[column].isnull()].iterrows():
        dataframe.loc[row[0], column] = dataframe[column][dataframe[classification]\
         == row[1][classification]].mean()

    return dataframe

def impute_prob(data, column, dist = 'normal'):
    '''
    Imputes missing data using probabilistic imputation. Default is normal.

    Inputs: pandas dataframe, column to impute, distribution
    '''
    dataframe = copy.deepcopy(data)

    if dist == 'Normal':
        for row in dataframe[dataframe[column].isnull()].iterrows():
            dataframe.loc[row[0], column] = np.random.normal(dataframe[column].mean(), 
                dataframe[column].std())

    return dataframe

def discretize(data, column, bins = 5, bin_size = None, labels = None, max_val = None, 
    min_val = None):
    '''
    Makes continuous column values discrete in a new column. Accepts a total 
    number of bins to separate values into or a bin size. If given both, will 
    prioritize bin size over total number of bins. Performs outer join with 
    existing dataset. If no labels are given, default values are integers. Will 
    use range of current data. Ranges will include right-most value and exclude
    left-most value.

    Optional: use max_val and min_val to specify a range of values for which 
    the bin_size to bin over; will then apply to the data. min_val will not
    be included in the range: (min_val, next_val]. If zero is selected, min value
    will be set to -.0001 so that 0 values are included.

    Output: pandas dataframe with new column
    '''
    assert data[column].dtype != object

    if bin_size != None: 
        max_val = max(data[column].values)
        min_val = min(data[column].values)
        if max_val == None or min_val == None:
            bins = int((max_val - min_val) / bin_size)
        else:  
            if min_val == 0:
                bins = [-.0001]
            else:
                bins = [min_val]
            n = min_val
            while n <= max_val:
                n += bin_size
                bins.append(n)

        print('Splitting {} by {} from {} to {}'.format(column, bin_size, min_val, max_val))

    if labels != None:
        assert len(labels) == bins
        new_column = pd.cut(data[column], bins = bins, labels = labels)
    else:
        new_column = pd.cut(data[column], bins = bins)

    new_column.name = str(column) + '_disc'
    rv = pd.concat([data, new_column], axis=1, join='outer')

    return rv

def dichotomize(data, column):
    '''
    Takes a categorical column and makes new, binary columns for each value.

    Output: pandas dataframe with new columns
    '''
    concat = []
    for value in data[column].values:
        # Add ones to attributes that match 
        set_1 = data[data[column] == value]
        add = pd.DataFrame({value: [1] * len(set_1)})
        add.index = set_1.index
        set_to_add1 = pd.concat([set_1, add], axis = 1, join = 'inner')
        # Add zeroes to attributes that do not match
        set_0 = data[data[column] != value]
        add = pd.DataFrame({value: [0] * len(set_0)})
        add.index = set_0.index
        set_to_add0 = pd.concat([set_0, add], axis = 1, join = 'inner')
        # Merge the two back together
        to_add = set_to_add0.merge(set_to_add1, how = 'outer')
        # Place in concat list
        concat.append(data.merge(to_add, how = 'left'))

    # Put everything together
    dataframe = concat[0]
    for df in concat[1:]: 
        dataframe = dataframe.merge(df)

    return dataframe

def log_scale(dataframe, col):
    '''
    Converts given column into a log scale, then appends to the end of the 
    dataframe.
    '''
    data = dataframe[col]
    new = []
    for x in data:
        new.append(math.log(x)