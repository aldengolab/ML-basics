'''
A class to hold models and their related methods.

Contains methods for doing top-k based metrics, but these are not currently
enabled, since CivicScape does not require them.
'''

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import metrics

class Model():
    '''
    A class to hold model information and methods.
    '''
    def __init__(self, clf, X_train, y_train, X_test, y_test, p, N, model_type,
                 iteration, run_name, label, thresholds, outfile):
        '''
        Constructor.
        '''
        self.clf = clf
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.params = p
        self.N = N
        self.model_type = model_type
        self.iteration = iteration
        self.thresholds = thresholds
        self.run_name = run_name
        self.label = label
        self.threshold = None
        self.y_pred_probs = None
        self.roc_auc = None
        self.precision = None
        self.recall = None
        self.accuracy = None
        self.fpr = None
        self.pipeline = None
        self.outfile = outfile
        self.pickle_path = None

    def run(self):
        '''
        Runs a model with params p.
        '''
        self.clf.set_params(**self.params)
        self.pipeline = Pipeline([
            ('clf', self.clf),
        ])
        self.y_pred_probs = self.pipeline.fit(self.X_train.todense(), self.y_train).predict_proba(self.X_test.todense())[:, 1]

    def calc_performance(self, measure):
        '''
        Stores performance given a threshold for prediction.
        '''
        self.roc_auc = self.auc_roc()
        self.threshold = measure
        self.fpr = self.fpr_at_threshold(self.threshold)
        self.precision = self.precision_at_threshold(self.threshold)
        self.recall = self.recall_at_threshold(self.threshold)
        self.accuracy = self.accuracy_at_threshold(self.threshold)
        self.fpr = self.fpr_at_threshold(self.threshold)

    def performance_to_file(self, pickle_path='N/A'):
        '''
        Write results to file.
        '''
        self.pickle_path = pickle_path
        filename = self.outfile
        for measure in self.thresholds:
            self.calc_performance(measure)
            self.model_performance_to_file(measure=measure, filename=filename)

    def model_performance_to_file(self, measure, filename, method='a'):
        '''
        Writes standard performance metrics (AUC, precision, etc.) to file.

        COLUMNS:
        model_id, run, label, model_type, iteration, roc-auc, threshold, precision, recall, accuracy, pickle_path, params_json
        '''
        with open(filename, method) as f:
            result = '"{0}-{1}", "{2}", "{3}", "{4}", "{5}", "{6}", "{7}", "{8}", "{9}", "{10}", "{11}", "{12}", "[{13}]"\n'.format(
                self.N, self.iteration, self.run_name, self.label,
                self.model_type, self.iteration, self.roc_auc, measure,
                self.precision, self.recall, self.accuracy, self.fpr,
                self.pickle_path, self.params)
            f.write(result)

    def auc_roc(self):
        '''
        Computes the Area-Under-the-Curve for the ROC curve.
        '''
        return metrics.roc_auc_score(self.y_test, self.y_pred_probs)

    def accuracy_at_threshold(self, threshold):
        '''
        Dyanamic threshold accuracy.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in self.y_pred_probs])
        return metrics.accuracy_score(self.y_test, y_pred)

    def precision_at_threshold(self, threshold):
        '''
        Dyanamic threshold precision.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in self.y_pred_probs])
        return metrics.precision_score(self.y_test, y_pred)

    def recall_at_threshold(self, threshold):
        '''
        Dyanamic threshold recall.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in self.y_pred_probs])
        return metrics.recall_score(self.y_test, y_pred)

    def fpr_at_threshold(self, threshold):
        '''
        Calculates the FPR at a specific threshold.
        '''
        y_pred = np.asarray([1 if i >= threshold else 0 for i in self.y_pred_probs])
        matrix = metrics.confusion_matrix(self.y_test, y_pred)
        fp = matrix[0][1]
        neg = matrix[0].sum()
        return float(fp)/float(neg)

    def recall_at_k(self, y_true, y_scores, k):
        '''
        Dynamic k recall, where 0<k<1.
        '''
        y_pred = self.k_predictions(y_scores, k)
        return metrics.recall_score(y_true, y_pred)

    def precision_at_k(self, y_true, y_scores, k):
        '''
        Dynamic k precision, where 0<k<1.
        '''
        y_pred = self.k_predictions(y_scores, k)
        return metrics.precision_score(y_true, y_pred)

    def accuracy_at_k(self, y_true, y_scores, k):
        '''
        Dynamic k accuracy, where 0<k<1.
        '''
        y_pred = self.k_predictions(y_scores, k)
        return metrics.accuracy_score(y_true, y_pred)

    def k_predictions(self, y_scores, k):
        '''
        Returns the y_pred vector as a numpy array using 0<k<1 as the prediction
        threshold.
        '''
        y_scores = list(enumerate(y_scores))
        y_scores.sort(key=lambda x: x[1])
        cut = np.floor(len(y_scores) * (1-k))
        y_pred = [0] * len(y_scores)
        for i in range(len(y_scores)):
            if i >= cut:
                y_pred[i] = (y_scores[i][0], 1)
            else:
                y_pred[i] = (y_scores[i][0], 0)
        y_pred.sort()
        rv = []
        for y in y_pred:
            rv.append(y[1])
        return np.asarray(rv)
