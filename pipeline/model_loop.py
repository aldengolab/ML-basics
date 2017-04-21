'''
Loops through ML models for classification.

Basic code borrowed from RAYID GHANI, with extensive edits
https://github.com/rayidghani/magicloops/blob/master/magicloops.py
'''

from __future__ import division
import numpy as np
import random
import os
import traceback
import pickle
from scipy.sparse import isspmatrix_csc, csc_matrix
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, Perceptron, SGDClassifier, OrthogonalMatchingPursuit, RandomizedLogisticRegression
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from model import Model

class ModelLoop():

    def __init__(self, X_train, X_test, y_train, y_test, models, iterations, run_name,
                 thresholds, label, comparison_threshold, project_folder=None):
        '''
        Constructor for the ModelLoop.
        '''
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.models_to_run = models
        self.iterations_max = iterations
        self.run_name = run_name
        self.params_iter_max = 50
        self.thresholds = thresholds
        self.clfs = None
        self.params = None
        self.define_clfs_params()
        self.label = label
        self.best_fpr = None
        self.closest_fpr = None
        self.comparison_threshold = comparison_threshold
        if project_folder == None:
            self.project_folder = self.run_name
        else:
            self.project_folder = project_folder

    def define_clfs_params(self):
        '''
        Defines all relevant parameters and classes for classfier objects.
        Edit these if you wish to change parameters.
        '''
        # These are the classifiers; add new classifiers here
        self.clfs = {
            'RF': RandomForestClassifier(n_estimators = 50, n_jobs = -1),
            'ET': ExtraTreesClassifier(n_estimators = 10, n_jobs = -1, criterion = 'entropy'),
            'AB': AdaBoostClassifier(DecisionTreeClassifier(max_depth = [1, 5, 10, 15]), algorithm = "SAMME", n_estimators = 200),
            'LR': LogisticRegression(penalty = 'l1', C = 1e5),
            'SVM': svm.SVC(kernel = 'linear', probability = True, random_state = 0),
            'GB': GradientBoostingClassifier(learning_rate = 0.05, subsample = 0.5, max_depth = 6, n_estimators = 10),
            'NB': GaussianNB(),
            'DT': DecisionTreeClassifier(),
            'SGD': SGDClassifier(loss = 'log', penalty = 'l2'),
            'KNN': KNeighborsClassifier(n_neighbors = 3),
            'NN': MLPClassifier()
            }
        # These are the parameters which will be run through
        self.params = {
             'RF':{'n_estimators': [1,10,100,1000], 'max_depth': [10, 15,20,30,40,50,60,70,100], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'random_state': [1]},
             'LR': {'penalty': ['l1','l2'], 'C': [0.00001,0.0001,0.001,0.01,0.1,1,10], 'random_state': [1]},
             'SGD': {'loss': ['log'], 'penalty': ['l2','l1','elasticnet'], 'random_state': [1]},
             'ET': {'n_estimators': [1,10,100,1000], 'criterion' : ['gini', 'entropy'], 'max_depth': [1,3,5,10,15], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'random_state': [1]},
             'AB': {'algorithm': ['SAMME', 'SAMME.R'], 'n_estimators': [1,10,100,1000], 'random_state': [1]},
             'GB': {'n_estimators': [1,10,100,1000], 'learning_rate' : [0.001,0.01,0.05,0.1,0.5],'subsample' : [0.1,0.5,1.0], 'max_depth': [1,3,5,10,20,50,100], 'random_state': [1]},
             'NB': {},
             'DT': {'criterion': ['gini', 'entropy'], 'max_depth': [1,2,15,20,30,40,50], 'max_features': ['sqrt','log2'],'min_samples_split': [2,5,10], 'random_state': [1]},
             'SVM' :{'C' :[0.00001,0.0001,0.001,0.01,0.1,1,10],'kernel':['linear'], 'random_state': [1]},
             'KNN' :{'n_neighbors': [1,5,10,25,50,100],'weights': ['uniform','distance'],'algorithm': ['auto','ball_tree','kd_tree']},
             'NN': {'hidden_layer_sizes':[(100,), (50,50), (30,30,30), (25,25,25,25)], 'max_iter': [200],'activation':['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'learning_rate':['constant', 'invscaling', 'adaptive'], 'random_state': [1]}
             }

    def run(self):
        '''
        Runs through each model specified by models_to_run once with each possible
        setting in params.
        '''
        N = 0
        self.prepare_report()
        for index, clf in enumerate([self.clfs[x] for x in self.models_to_run]):
            iteration = 0
            print 'Running {}.'.format(self.models_to_run[index])
            parameter_values = self.params[self.models_to_run[index]]
            grid = ParameterGrid(parameter_values)
            while iteration < self.iterations_max and iteration < len(grid):
                print '    Running Iteration {} of {}...'.format(iteration + 1,
                      self.iterations_max)
                if len(grid) > self.iterations_max:
                    p = random.choice(list(grid))
                else:
                    p = list(grid)[iteration]
                try:
                    m = Model(clf, self.X_train, self.y_train, self.X_test,
                                self.y_test, p, N, self.models_to_run[index],
                                iteration, self.run_name, self.label,
                                self.thresholds, self.outfile)
                    m.run()
                    self.check_model_performance(m, self.comparison_threshold)
                    m.performance_to_file()
                    self.pickle_model(m)
                except IndexError as e:
                    print p
                    print N
                    print 'IndexError: {}'.format(e)
                    print traceback.format_exc()
                    continue
                except RuntimeError as e:
                    print p
                    print N
                    print 'RuntimeError: {}'.format(e)
                    print traceback.format_exc()
                    continue
                except AttributeError as e:
                    print p
                    print N
                    print 'AttributeError: {}'.format(e)
                    print traceback.format_exc()
                    continue
                except ValueError as e:
                    print p
                    print N
                    print 'Unexpected ValueError: {}'.format(e)
                    print traceback.format_exc()
                    continue
                iteration += 1
            N += 1

    def pickle_model(self, m):
        '''
        Pickles the models held within this loop.
        '''
        pickle_path = self.project_folder + '/' +'pickle_jar/{}_{}-{}.pkl'.format(self.run_name, m.N, m.iteration)
            if not os.path.isdir(self.project_folder + '/' +'pickle_jar'):
                os.makedirs(self.project_folder + '/' +'pickle_jar')
            with open(pickle_path, 'wb') as p:
                pickle.dump(model, p)

    def prepare_report(self):
        '''
        Prepares the output file(s).
        '''
        if not os.path.isdir(self.project_folder):
            os.makedirs(self.project_folder)
        self.outfile = self.project_folder + '/' + self.run_name + '_modelingresults.csv'
        with open(self.outfile, 'w') as f:
            f.write('model_id, run, label, model_type, iteration, roc-auc, threshold, precision, recall, accuracy, fpr, pickle_path, params_json\n')
