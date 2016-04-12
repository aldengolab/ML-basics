# CAPP 30254: Machine Learning for Public Policy
# PA2 ML Pipeline
# ALDEN GOLAB
# A class for Classifier ML models

import pandas as pd
import numpy as np
import sklearn.linear_model as sklinear

Class Classifier: 
	'''
	An encapsulation for classifier construction.

	Inputs: pandas dataframe, classification model type
	'''

	def __init__(self, training_data, class_type, x = [], y = None):
		self._type = class_type
		self._train_data = training_data
		self._x = x
		self._y = y
		self.model = None
		self.run_model(self)

	def run_model(self):
		'''
		Runs model.
		'''
		if self._trpe = 'logistic':
			self.model = self.model_logit()
		else: 
			print('Model not yet encapsulated.')

	def model_logit(self):
		'''
		Runs logit model on training data.
		'''
		self.model = sklinear.LogisticRegression()
		
		self.model.fit()



		
