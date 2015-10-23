# Notes:-
# This code uses CART instead of c4.5. They both are pretty close to each other. Refer documentation for difference.
# This code uses SAMME.R for Adaboost instead of ADaboost.MI used for boosting in the paper.
#
# Links to documentations
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier
# http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.ensemble
# http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier.fit

# CLEANING
# Lots of cleaning needed. Convert everything to a constant format to parse. Tab separate, comma separated, colon separated, etc
# Add missing headers
# Prep missing values using imputing.
# Categorical data needs to be mapped to dummy numerical values for trees to work properly.



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
from sklearn.cross_validation import cross_val_score
from sklearn import cross_validation
from sklearn.preprocessing import Imputer
import timeit
import time

CSV_SEP = ','

# get list of datasets of UCI rep to use
def get_dataset_list():
    datasets = [
                'data/anneal.data.txt',
                'data/segment.dat.txt',
                'data/krkopt.data.txt',
                'data/house-votes-84.data.txt',
                'data/soybean-large.data.txt',
                'data/sat.trn.txt',
                'data/sonar.all-data.txt',
                'data/letter-recognition.data.txt',
                'data/winequality-white.csv',
                'data/winequality-red.csv'

                # 'splice.data.txt' - Need to enter comma after every char ,rest cleaning is done.
                #'data/glass.data.txt', - This is continuous
                #'data/heart.dat.txt',- This is continuous
                ]

    return datasets

# Do bagging.
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_bagging(X, y):
    bagging = BaggingClassifier(DecisionTreeClassifier(), 200, 0.67, 1.0, True, True)
    return cross_val_score(bagging, X, y, cv=10)

# Do boosting
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_boosting(X, y):
    boosting = AdaBoostClassifier(DecisionTreeClassifier(), n_estimators=100)
    return cross_val_score(boosting, X, y, cv=10)

# Do Randomization
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_randomization(X, y):
    random = ExtraTreesClassifier(200)
    return cross_val_score(random, X, y, cv=10)

# Do plain vanilla CART
# X : {array-like, sparse matrix} of shape = [n_samples, n_features]
# Y : array-like, shape = [n_samples]
def do_cart(X, y):
    cart = DecisionTreeClassifier()
    return cross_val_score(cart, X, y, cv=10)

# Convert accuracy scores to error rates.
def convert_to_error_rate(score):
    error = []
    for foldscore in score:
        errorscore = 1 - foldscore
        error.append(errorscore)
    return error

# Remove missing values from data and replace then with NaN for imputer to work.
def remove_missing_values(arr):
    X = arr
    for row_ind, row in enumerate(arr):
        for col_ind, element in enumerate(row):
            if (element == '?'):
                X[row_ind][col_ind] = float('NaN') #Fixed number to identify missing values.
    return X

# Convert categorical feature data to dummy values to CART can work.
def convert_categorical_data(arr):
    X = []

    for row_ind, row in enumerate(arr):
        # create a map from your variable names to unique integers:
        intmap = dict([(val, i) for i, val in enumerate(set(row))])
        # make the new array hold corresponding integers instead of strings:
        new_vals = [intmap[val] for val in row]
        X.append(new_vals)
    return X

def time_fn( fn, *args, **kwargs ):
    start = time.clock()
    results = fn( *args, **kwargs )
    end = time.clock()
    fn_name = fn.__module__ + "." + fn.__name__
    print(fn_name + ": " + str(end-start) + "s")
    return [results, end-start]

# Main
scores = []
error_rates = []
mean_error_rates = []
mean_scores = []
meanTimes = []
for dataset_url in get_dataset_list():
    print("Testing: "+dataset_url)
    datascore = []
    dataerrorrate = []

    # Read data.
    df = pd.read_csv(dataset_url, CSV_SEP)

    # Remove non attribute columns.
    Y = df['class'].values
    if ('instance' in df):
        df.drop('instance', 1)
    df = df.drop('class', 1)

    # Prep data.
    df.fillna(df.mode().iloc[0])
    X = df.as_matrix()
    X = remove_missing_values(X)
    X = convert_categorical_data(X)
    imp = Imputer(strategy='most_frequent', axis=0)
    imp.fit(X, Y)
    X = imp.transform(X)

    # Get cross validation scores.
    cart_score, cartTime = time_fn(do_cart, X, Y)
    bagging_score, bagTime = time_fn(do_bagging, X, Y)
    boosting_score, boostTime = time_fn(do_boosting, X, Y)
    random_score, randomTime = time_fn(do_randomization, X, Y)

    print("Cart: {0} bag: {1} boost: {2} random: {3}".format(cartTime, bagTime, boostTime, randomTime))
    meanTimes.append([cartTime, bagTime, boostTime, randomTime])

    # Store scores.
    datascore.append(cart_score)
    datascore.append(bagging_score)
    datascore.append(boosting_score)
    datascore.append(random_score)
    scores.append(datascore)

    # Get error rates.
    cart_error = convert_to_error_rate(cart_score)
    bagging_error = convert_to_error_rate(bagging_score)
    boosting_error = convert_to_error_rate(boosting_score)
    random_error = convert_to_error_rate(random_score)

    # Store error rates.
    dataerrorrate.append(cart_error)
    dataerrorrate.append(bagging_error)
    dataerrorrate.append(boosting_error)
    dataerrorrate.append(random_error)
    error_rates.append(dataerrorrate)

    # Calculate mean error rates.

    mean_error_rates.append([np.mean(cart_error), np.mean(bagging_error), np.mean(boosting_error), np.mean(random_error)])
    mean_scores.append([np.mean(cart_score), np.mean(bagging_score), np.mean(boosting_score), np.mean(random_score)])

print("Average Times: ")
cartTime = np.mean([x[0] for x in meanTimes])
print("Cart: {0}".format(cartTime))
bagTime = np.mean([x[1] for x in meanTimes])
print("Bag: {0}".format(bagTime))
boostTime = np.mean([x[2] for x in meanTimes])
print("Boost: {0}".format(boostTime))
randomTime = np.mean([x[3] for x in meanTimes])
print("Random: {0}".format(randomTime))

names = dict()
names[0] = "Annealing (9)"
names[1] = "Voting (15)"
names[2] = "King Rook King Pawn (31)"
names[3] = "Soybean (6)"
names[4] = "Segment (4)"
names[5] = "Satellite (8)"
names[6] = "Sonar (1)"
names[7] = "Letter Recognition (2)"
names[8] = "Wine quality - white"
names[9] = "Wine quality - red"

# Plot box plot - scores.
for idx, score in enumerate(scores):
    plt.boxplot(score, labels=['CART', 'Bagging', 'Boosting', 'Randomization'])
    plt.ylabel('Score')
    plt.xlabel('Classification Method')
    plt.title('Classification Score - ' + names[idx])
    plt.show()

# Plot box plot - Error rates.
for idx, errorrate in enumerate(error_rates):
    plt.boxplot(errorrate, labels=['CART', 'Bagging', 'Boosting', 'Randomization'])
    plt.ylabel('Error rates')
    plt.xlabel('Classification Method')
    plt.title('Error Rates -' + names[idx])
    plt.show()

# Plot error rates of various algorithms
mean_error_rates = np.array(mean_error_rates)
plt.plot(mean_error_rates.T[0], label='CART')
plt.plot(mean_error_rates.T[1], label='Bagging')
plt.plot(mean_error_rates.T[2], label='Boosting')
plt.plot(mean_error_rates.T[3], label='Randomization')
plt.legend(loc='best')
plt.ylabel('Error rates')
plt.xlabel('Data Set')
plt.title('Line plot of error rates for various datasets using various algorithms')
plt.show()

mean_scores = np.array(mean_scores)
plt.plot(mean_scores.T[0], label='CART')
plt.plot(mean_scores.T[1], label='Bagging')
plt.plot(mean_scores.T[2], label='Boosting')
plt.plot(mean_scores.T[3], label='Randomization')
plt.legend(loc='best')
plt.ylabel('Classification scores')
plt.xlabel('Data Set')
plt.title('Line plot of classication scores for various datasets using various algorithms')
plt.show()

mean_times = np.array(meanTimes)
plt.plot(mean_times.T[0], label='CART')
plt.plot(mean_times.T[1], label='Bagging')
plt.plot(mean_times.T[2], label='Boosting')
plt.plot(mean_times.T[3], label='Randomization')
plt.legend(loc='best')
plt.ylabel('Run times')
plt.xlabel('Data Set')
plt.title('Line plot of run times for various datasets using various algorithms')
plt.show()


