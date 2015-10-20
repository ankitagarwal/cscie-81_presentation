# Notes:-
# This code uses CART instead of c4.5. They both are pretty close to each other. Refer documentation for difference.
# This code uses SAMME.R for Adaboost instea of ADaboost.MI used for boosting in the paper.
#
# Links to documentations
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit

# TODO
# Re-verify all parameters to ensembles methods and make sure we are not diverging from the paper.
# use 10 fold cross validation to determine accuracy of the ensemble methods or something simpler ?
# Do fancy charts with the accuracy data.
# Find out if there is a way in extratreeclassifier to make it use only positive information gain splits.
# Are we going to use pruning?



# Import stuff
import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier


CSV_SEP = ';'

# get list of datasets of UCI rep to use
def get_dataset_list():
    datasets = ['https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv']
    return datasets

# Do bagging.
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_bagging(X, y):
    bagging = BaggingClassifier(DecisionTreeClassifier(), 200, 0.67, 1.0, True, True)
    bagging.fit(X, y)
    return bagging

# Do boosting
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_boosting(X, y):
    boosting = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100)
    boosting.fit(X, y)
    return boosting

# Do Randomization
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_randomization(X, y):
    random = ExtraTreesClassifier(200)
    random.fit(X, y)
    return random

# Do plain vanilla CART
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_cart(X, y):
    cart = DecisionTreeClassifier()
    cart.fit(X, y)
    return cart


# Main
for dataset_url in get_dataset_list():
    df = pd.read_csv(dataset_url, CSV_SEP)
    X = df.as_matrix()
    print(dataset_url)
    df.head()