# CAPP 30254: Machine Learning for Public Policy
# PA2 ML Pipeline
# ALDEN GOLAB
# A class for Classifier ML models

import pandas as pd
import numpy as np
import sklearn.linear_model as sklinear

class Classifier: 
    '''
    An encapsulation for classifier construction.

    Inputs: pandas dataframe, classification model type, x variables in 
    list, y variable

    class_type can take: 'logistic'
    '''

    def __init__(self, training_data, class_type, x = [], y = None):
        self._type = class_type
        self._train_data = training_data
        self._x = x
        self._y = y
        self.model = None
        self._coefficients = None
        self.train_model()
        self.determine_accuracy()

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def type(self):
        return self._type

    @property
    def accuracy(self):
        return self._accuracy

    @property
    def coefficients(self):
        return self._coefficients
    
    
    def train_model(self):
        '''
        Trains model.
        '''
        if self._type == 'logistic':
            self.model_logit()
        else: 
            print('Model type not yet encapsulated.')

    def model_logit(self):
        '''
        Trains logit model on training data.
        '''
        self.model = sklinear.LogisticRegression()
        self.model.fit(self._train_data[self._x], self._train_data[self._y])
        self._coefficients = pd.DataFrame(self.model.coef_.transpose())

    def determine_accuracy(self):
        '''
        Predicts values of y using trained model on training data. 
        '''
        prediction = pd.DataFrame({'predict': self.model.predict(
            self._train_data[self._x])})
        prediction.index = self._train_data.index
        test_set = pd.concat([self._train_data, prediction], axis=1)
        self._accuracy = float(len(test_set[test_set[self._y] == test_set[
            'predict']])) / float(len(test_set))

    def test_model(self, test_data):
        '''
        Takes pandas dataframe and predicts y based on x's using trained 
        model.
        '''
        prediction = pd.DataFrame({'prediction': self.model.predict(test_data[
            self._x])})
        prediction.index = test_data.index
        test_data = pd.concat([test_data, prediction], axis=1)
        return test_data

    def coefficients(self):
        '''
        Gets labeled coefficients as a pandas dataframe.
        '''
        x_labels = pd.DataFrame({'Variable': self._x})
        rv = pd.concat([x_labels, self.coefficients], axis = 1)
        rv.index = ['Variable', 'Coefficient']
        return rv


        


        
