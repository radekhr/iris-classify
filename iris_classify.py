#!/usr/bin/python3
# -*- coding: utf-8 -*-

import warnings
import pandas
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_array_from_csv():
    """Return dataset values from url"""
    url = "https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/d546eaee765268bf2f487608c537c05e22e4b221/iris.csv"
    dataset = pandas.read_csv(url)
    return dataset.values


def prepare_train_validation_arrays(array, seed, validation_size):
    """Return validation and training datasets from base dataset"""
    X, Y = array[:, :4], array[:, 4]
    return train_test_split(X, Y, test_size=validation_size, random_state=seed)


def create_algorithm_models_list():
    """Return list with most popular algorithm models"""
    models = list()
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVC', SVC()))
    return models


def evaluate_models_with_kfold_validation(seed, models, X_train, Y_train, scoring, num_splits):
    """Return algorithms results and names from kfold cross evaluation"""
    results = list()
    names = list()
    mean = int()
    for name, model in models:
        # rskf = model_selection.RepeatedStratifiedKFold(n_splits=num_kfold_splits, n_repeats=3, random_state=seed)
        skf = KFold(n_splits=num_splits, random_state=seed)
        cv_results = cross_val_score(model, X_train, Y_train, cv=skf, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        print("Algorithm {0} - Mean: {1:.3f}, Std.deviation: {2:.3f}".format(name, cv_results.mean(), cv_results.std()))
        if cv_results.mean() >= mean:
            mean = cv_results.mean()
            best_algorithm = name
    return best_algorithm, results, names


def display_compared_algorithms(results, names):
    """Display algorithms results in boxplot form"""
    fig = plt.figure()
    fig.suptitle('Compare algorithms')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()


def make_predicts_for_best_algorithm(best_algorithm, X_train, Y_train, X_validation):
    """Make predicts depends on best fitted algorithm"""
    algo = object()
    if best_algorithm == 'LR':
        algo = LogisticRegression()
    elif best_algorithm == 'LDA':
        algo = LinearDiscriminantAnalysis()
    elif best_algorithm == 'KNN':
        algo = KNeighborsClassifier()
    elif best_algorithm == 'CART':
        algo = DecisionTreeClassifier()
    elif best_algorithm == 'NB':
        algo = GaussianNB()
    elif best_algorithm == 'SVC':
        algo = SVC()
    algo.fit(X_train, Y_train)
    return algo.predict(X_validation)


def display_detailed_predicts_report(best_algorithm, Y_validation, predicts):
    """Show accuracy score, confusion matrix and classification report for predicted array"""
    print('\nBest matched algorithm for given params: ', best_algorithm)
    print('\nAccuracy score: {:.3f}'.format(accuracy_score(Y_validation, predicts)))
    print('\nConfusion matrix (elements that are not on the main diagonal indicates the number of errors made): ')
    print(confusion_matrix(Y_validation, predicts, labels=['setosa', 'versicolor', 'virginica']))
    print('\nClassification report: ')
    print(classification_report(Y_validation, predicts))


if __name__ == '__main__':
    seed = 11
    validation_size = 0.2
    num_splits = 5
    scoring = 'accuracy'

    array = get_array_from_csv()
    X_train, X_validation, Y_train, Y_validation = prepare_train_validation_arrays(array, seed, validation_size)
    models = create_algorithm_models_list()
    best_algorithm, results, names = evaluate_models_with_kfold_validation(seed, models, X_train, Y_train, scoring, num_splits)
    display_compared_algorithms(results, names)
    predicts = make_predicts_for_best_algorithm(best_algorithm, X_train, Y_train, X_validation)
    display_detailed_predicts_report(best_algorithm, Y_validation, predicts)


