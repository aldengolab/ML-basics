# CAPP 30254: Machine Learning for Public Policy
# PA2 ML Pipeline
# ALDEN GOLAB
# Primary code file

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy

def load_file(filename, index = None):
    '''
    Reads file with column index  and returns a Pandas dataframe. If index
    name is missing for UID, use [0] to refer to the first column even if 
    it is unnamed.

    Currently only has options for csv. More to come.
    '''
    if 'csv' in filename:
        if index != None: 
            return pd.read_csv(filename, index_col = index)
        else: 
            return pd.read_csv(filename)
    else: 
        print ('Input currently not built for this filetype')

def summarize(dataframe, column = None, plots = False, write = False):
    '''
    Takes dataframe and returns summary statistics for each numerical column 
    as a pandas dataframe. Optional print plots of all columns; to do so, 
    input 'Y' for plots. Optional write summaries to file, input 'Y'.
    '''
    summaries = []

    if column == None:
        for col in dataframe.columns:
            summaries.append(get_summary(dataframe, col))        
    else:
        summaries.append(get_summary(dataframe, column))

    if write == True:
        for summ in summaries:
            summ[2].to_csv('{}_summary.csv'.format(summ[1]))
        print('Wrote descriptive statistics to file.')
    if plots == True:
        plot(dataframe)

    return summaries

def get_summary(dataframe, col):
    '''
    Produces column summary.
    '''
    if dataframe[col].dtype != object and 'disc' not in col:
        summary = ('float64', col, get_cont_summary(dataframe, col))
    else:
        summary = ('object', col, get_cat_summary(dataframe, col)) 

    return summary

def get_cont_summary(dataframe, col):
    '''
    Produces summary of continuous data.
    '''
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

    return summary

def get_cat_summary(dataframe, col):
    '''
    Produces summary of categorical data.
    '''
    summary = dataframe[col].describe()
    # Count missing values and append
    missing = pd.Series(dataframe[col].isnull().sum())
    missing = missing.rename({missing.index[0]: 'missing'})
    summary = summary.append(missing)

    return summary

def plot(dataframe):
    '''
    Plots histograms or bar charts for each column and saves to file within
    the same directory.
    '''
    histogram = ['object', 'category']
    for col in dataframe.columns:
        if dataframe[col].dtype.name in histogram:
            dataframe[col].value_counts().plot(kind = 'bar')
            plt.suptitle('Bar Graph for {}'.format(col), fontsize = 14)
            plt.savefig('{}.png'.format(col))
            print('Saved figure as {}.png'.format(col))
            plt.close()
        else:
            dataframe[col].hist()
            plt.suptitle('Histogram for {}'.format(col), fontsize = 14)
            plt.savefig('{}.png'.format(col))
            print('Saved figure as {}.png'.format(col))
            plt.close()

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

def write_csv(data, filename):
    '''
    Writes a pandas dictionary to a filename.
    '''
    data.to_csv(filename)
    print('Wrote data to file {}'.format(filename))




