# CAPP 30254: Machine Learning for Public Policy
# PA2 ML Pipeline
# ALDEN GOLAB
# Code used for assignment

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
	Runs all of the work for pa2. 
	'''
	data = pl.load_file('cs-training.csv', index=[0])
	data = pl.impute(data, 'MonthlyIncome')
	data = pl.discretize(data, 'age', bin_size = 10, max_val = 150, min_val=0)
	data = pl.discretize(data, 'MonthlyIncome', bins = 20)
	data = impute(data, 'NumberOfDependents', classification = 'age_disc')
	return data



