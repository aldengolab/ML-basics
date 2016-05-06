# CAPP 30254: Machine Learning for Public Policy
# ALDEN GOLAB
# ML Pipeline
# 
# Model loop. 

## CODE STRUCTURE LIBERALLY BORROWED FROM RAYID GHANI, WITH EXTENSIVE EDITS ##
## https://github.com/rayidghani/magicloops/blob/master/magicloops.py ##
## Accessed: 5/5/2016 ##

from __future__ import division
import sys
import read
import matplotlib.cm as cm
import copy
import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm, metrics, tree, decomposition, svm
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import ParameterGrid
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
import random
import pylab as pl
import matplotlib.pyplot as plt
from scipy import optimize
import time
import process

THRESHOLD = .75

def define_project_params():
    '''
    Parameters specific to the project being run.
    '''
    y_variable = 'SeriousDlqin2yrs'
    imp_cols = ['MonthlyIncome', 'NumberOfDependents', 'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfTime30-59DaysPastDueNotWorse']
    robustscale_cols = ['MonthlyIncome', 'NumberOfTime30-59DaysPastDueNotWorse',
    'DebtRatio', 'NumberOfOpenCreditLinesAndLoans', 'NumberOfTimes90DaysLate', 
    'NumberRealEstateLoansOrLines', 'NumberOfTime60-89DaysPastDueNotWorse', 
    'NumberOfDependents', 'RevolvingUtilizationOfUnsecuredLines']
    models_to_run = ['KNN','RF','LR','AB','NB','DT']
    scale_columns = ['RevolvingUtilizationOfUnsecuredLines', 'age', 
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    X_variables = ['RevolvingUtilizationOfUnsecuredLines', 'age', 
    'NumberOfTime30-59DaysPastDueNotWorse', 'DebtRatio', 
    'MonthlyIncome', 'NumberOfOpenCreditLinesAndLoans', 
    'NumberOfTimes90DaysLate', 'NumberRealEstateLoansOrLines',
    'NumberOfTime60-89DaysPastDueNotWorse', 'NumberOfDependents']
    return (y_variable, imp_cols, models_to_run, robustscale_cols, 
        scale_columns, X_variables)

def define_clfs_params():
    '''
    Defines all relevant parameters and classes for classfier objects.
    '''
    clfs = {
        'RF': RandomForestClassifier(n_estimators = 50, n_jobs = -1),
        'ET': ExtraTreesClassifier(n_estimators = 10, n_jobs = -1, criterion = 'entropy'),
        'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth = 1), algorithm = "SAMME", n_estimators = 200),
        'LR': LogisticRegression(penalty = 'l1', C = 1e5),
        'SVM': svm.SVC(kernel = 'linear', probability = True, random_state = 0),
        'GB': GradientBoostingClassifier(learning_rate = 0.05, subsample = 0.5, max_depth = 6, n_estimators = 10),
        'NB': GaussianNB(),
        'DT': DecisionTreeClassifier(),
        'SGD': SGDClassifier(loss = "hinge", penalty = "l2"),
        'KNN': KNeighborsClassifier(n_neighbors = 3) 
        }
    params = { 
        'RF':{'n_estimators': [1,10,100,1000,10000], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10]},
        'SGD': {'loss': ['hinge','log','perceptron'], 'penalty': ['l2','l1','elasticnet']},
        'ET': {'n_estimators': [1,10,100,1000,10000], 'criterion' : ['gini', 'entropy'] ,'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000,10000]},
        'GB': {'n_estimators': [1,10,100,1000,10000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100]},
        'NB' : {},
        'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,5,10,20,50,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10]},
        'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear']},
        'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']}
        }
    
    return clfs, params

def clf_loop(dataframe, clfs, models_to_run, y_variable, X_variables, 
 imp_cols = [], addl_runs = 0, evalution = ['AUC'], stat_k = .05, plot = True, 
 robustscale_cols = [], scale_columns = []):
    '''
    Runs through each model specified by models_to_run once with each possible
    setting in params.
    '''
    for n in range(1 + addl_runs):
        print('Sampling new test/train split...')
        X_train, X_test, y_train, y_test = process.test_train_split(dataframe, 
            y_variable, test_size=0.1)
        print('Imputing data for new split...')
        for col in imp_cols:
            X_train, mean = process.impute(X_train, col, keep = True)
            X_test = process.impute_specific(X_test, col, mean)
        print('XTRAIN:', X_train)
        print('XTEST:', X_test)
        print('Finished imputing, transforming data...')
        for col in robustscale_cols:
            X_train = process.robust_transform(X_train, col)
            X_test = process.robust_transform(X_test, col)
        X_train = process.scale_columns(X_train, scale_columns)
        X_test = process.scale_columns(X_test, scale_columns)
        print('XTRAIN:', X_train)
        print('XTEST:', X_test)
        print('Training model...')
        for index, clf in enumerate([clfs[x] for x in models_to_run]):
            print(models_to_run[index])
            parameter_values = grid[models_to_run[index]]
            for p in ParameterGrid(parameter_values):
                try:
                    clf.set_params(**p)
                    print(clf)
                    y_pred_probs = clf.fit(X_train, y_train).predict_proba(
                        X_test)[:,1]
                    if 'precision' in evalution:
                        print(precision_at_k(y_test, y_pred_probs, stat_k))
                    if plot:
                        plot_precision_recall_n(y_test, y_pred_probs, clf)
                    if 'AUC' in evalution:
                        print(auc_at_k(y_test, y_pred_probs, stat_k))
                    if 'recall' in evalution:
                        print(recall_at_k(y_test, y_pred_probs, stat_k))
                except IndexError as e:
                    print('Error: {0}'.format(e))
                    continue

def plot_precision_recall_n(y_true, y_prob, model_name):
    '''
    Plots the precision recall curve.
    '''
    from sklearn.metrics import precision_recall_curve

    y_score = y_prob
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_true, y_score)
    precision_curve = precision_curve[:-1]
    recall_curve = recall_curve[:-1]
    pct_above_per_thresh = []
    number_scored = len(y_score)
    
    for value in pr_thresholds:
        num_above_thresh = len(y_score[y_score>=value])
        pct_above_thresh = num_above_thresh / float(number_scored)
        pct_above_per_thresh.append(pct_above_thresh)
    pct_above_per_thresh = np.array(pct_above_per_thresh)
    
    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(pct_above_per_thresh, precision_curve, 'b')
    ax1.set_xlabel('percent of population')
    ax1.set_ylabel('precision', color='b')
    ax2 = ax1.twinx()
    ax2.plot(pct_above_per_thresh, recall_curve, 'r')
    ax2.set_ylabel('recall', color='r')
    
    name = model_name
    plt.title(name)
    plt.savefig(name)
    #plt.show()

def auc_at_k(y_true, y_scores, k = None):
    '''
    Dyanamic k-threshold AUC. Defines threshold for Positive at the 
    value that returns the k*n top values where k is within [0-1]. If k is not
    specified, threshold will default to THRESHOLD.
    '''
    if k != None:
        threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))] 
    else: 
        threshold = THRESHOLD
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.roc_auc_score(y_true, y_scores), threshold)

def precision_at_k(y_true, y_scores, k = None):
    '''
    Dyanamic k-threshold precision. Defines threshold for Positive at the 
    value that returns the k*n top values where k is within [0-1]. If k is not
    specified, threshold will default to THRESHOLD.
    '''
    if k != None:
        threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))] 
    else: 
        threshold = THRESHOLD
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.precision_score(y_true, y_pred), threshold)

def recall_at_k(y_true, y_scores, k = None):
    '''
    Dyanamic k-threshold recall. Defines threshold for Positive at the 
    value that returns the k*n top values where k is within [0-1]. If k is not
    specified, threshold will default to THRESHOLD.
    '''
    if k != None:
        threshold = np.sort(y_scores)[::-1][int(k*len(y_scores))] 
    else: 
        threshold = THRESHOLD
    y_pred = np.asarray([1 if i >= threshold else 0 for i in y_scores])
    return (metrics.recall_score(y_true, y_pred), threshold)

def main(filename): 
    '''
    Runs the loop.
    '''
    dataframe = read.load_file(filename, index = 0)
    # Replace 98s with missing values
    dataframe = process.replace_value_with_nan(dataframe, 
        'NumberOfTime30-59DaysPastDueNotWorse', 98)
    dataframe = process.replace_value_with_nan(dataframe, 
        'NumberOfTimes90DaysLate', 98)
    dataframe = process.replace_value_with_nan(dataframe, 
        'NumberOfTime30-59DaysPastDueNotWorse', 98)

    clfs, params = define_clfs_params()
    y_variable, imp_cols, models_to_run, robustscale_cols, scale_columns, \
     X_variables = define_project_params()
    clf_loop(dataframe, clfs, models_to_run, y_variable, X_variables, imp_cols = imp_cols,
        scale_columns = scale_columns)

'''
if __name__ == '__main__':
    if sys.argv[0] !=  []:
        data = read.load_file(sys.argv[0])
        main(sys.argv[0])
    else:
        print('Usage: model.py <datafilename>')
'''
