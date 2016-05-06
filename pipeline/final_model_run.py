# CAPP 30254: Machine Learning for Public Policy
# ALDEN GOLAB
# ML Pipeline

# Code for final model

from model import *

def run_model(train_filename, test_filename, result_filename):
    '''
    Runs final RF model from loop. Prints to csv.
    '''
    train = read.load_file(train_filename, index = 0)
    # Replace 98s with missing values
    train = process.replace_value_with_nan(train, 
        'NumberOfTime30-59DaysPastDueNotWorse', 98)
    train = process.replace_value_with_nan(train, 
        'NumberOfTimes90DaysLate', 98)
    train = process.replace_value_with_nan(train, 
        'NumberOfTime60-89DaysPastDueNotWorse', 98)
    
    test = read.load_file(test_filename, index = 0)
        # Replace 98s with missing values
    test = process.replace_value_with_nan(test, 
        'NumberOfTime30-59DaysPastDueNotWorse', 98)
    test = process.replace_value_with_nan(test, 
        'NumberOfTimes90DaysLate', 98)
    test = process.replace_value_with_nan(test, 
        'NumberOfTime60-89DaysPastDueNotWorse', 98)

    y_variable, imp_cols, models_to_run, robustscale_cols, scale_columns, \
     X_variables = define_project_params()
    parameter_values = {'n_estimators': [1000], 'max_depth': [20], 'max_features': ['log2'],'min_samples_split': [10]}

    y_train = train[y_variable]
    X_train = train[X_variables]
    X_test  = test[X_variables]

    for col in imp_cols:
        X_train, mean = process.impute_mean(X_train, column = col)
        X_test, mean = process.impute_mean(X_test, column = col)
    for col in robustscale_cols:
        X_train = process.robust_transform(X_train, column = col)
        X_test = process.robust_transform(X_test, column = col)
    for col in scale_columns:
        X_train = process.normalize_scale(X_train, col = col)
        X_test = process.normalize_scale(X_test, col = col)

    clf = RandomForestClassifier(n_estimators = 50, n_jobs = -1)
    p = ParameterGrid(parameter_values)[0]
    clf.set_params(**p)
    print(clf)
    y_prediction = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
    y_prediction = pd.DataFrame(y_prediction)
    final = pd.concat([test, y_prediction], axis = 1)
    
    final.to_csv('{}.csv'.format(result_filename))
    print('Printed results to file {}'.format(result_filename))

