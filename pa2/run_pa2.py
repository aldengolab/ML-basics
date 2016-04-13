# CAPP 30254: Machine Learning for Public Policy
# PA2 ML Pipeline
# ALDEN GOLAB
# Code used for assignment (broadly; descriptive stats calls not shown)

import pipeline as pl
import pandas as pd
import numpy as np
import Classifier

X_VARS = ['RevolvingUtilizationOfUnsecuredLines', 'age',
 'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 'MonthlyIncome', 
 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
 'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse',
 'NumberOfDependents']
Y_VAR = 'SeriousDlqin2yrs'

def clean_data(filename):
    '''
    Runs all of the pre-processing work for pa2. 
    '''
    data = pl.load_file(filename, index=[0])
    print('File loaded.')
    data = pl.impute(data, 'MonthlyIncome')
    print('MonthlyIncome imputed.')
    data = pl.discretize(data, 'age', bin_size = 10, max_val = 150, min_val=0)
    data = pl.discretize(data, 'MonthlyIncome', bins = 20)
    print('Data discretized.')
    data = pl.impute(data, 'NumberOfDependents', classification = 'age_disc')
    print('NumberOfDependents imputed.')
    return data

def train_model(data):
    '''
    Trains logistic model with given training data. 
    '''
    return Classifier.Classifier(data, 'logistic', x = X_VARS, y = Y_VAR)

def predict_test(logit, test_data_filename):
    '''
    Runs a trained logit model on the test data.
    '''
    test_data = pl.clean_data(test_data_filename)
    prediction = logit.test_model(test_data)
    return prediction

def go(filename, test_data_filename):
    data = clean_data(filename)
    summarize(data, write = True, plots = True)
    logit = train_model(data)
    prediction = predict_test(logit, test_data_filename)
    write_csv(prediction, 'prediction.csv')
    summarize(prediction)

