# Thesis Topic
# Create a Decision Support System (DSS) which will support datasets regarding breast cancer.
# All required to be done, is import all necessary modules, required for the system designed.
# To begin with, the built-in CSV module will be imported.

import csv

# Then the "rest" of the modules required, are imported (and explained) one-by-one.
# The numpy module may be useful for experiments with the scikit-learn module,
# which will be imported later on.

import numpy as np

# Another "useful" package (in case of emergency) ought to be the math package
# just as the CSV module, it is built-in in Python.

import math

# And then, the pandas module will be imported.
# The pandas module will be of need for the Decision Support System designed and implemented,
# as well as the CSV datasets the program will fetch to be read.

import pandas as pd

# Most experiments, will base on machine learning and data science.
# The scikit-learn module will be called and imported using the "sklearn" abbreviation
# and by importing it, all features will be granted access.

import sklearn

# The "random" package is imported.
# Occasionally, some sequences may feature random numbers and/or time.
# Therefore, it is considered useful.

import random

# For more complicated algorithms, classifiers and distributions, the "scipy" package might be of need;
# therefore, it will be imported, and there will be high chance of use in the model.

# the scipy package will be imported with all its features.

import scipy

# Lastly, for the plot part of the model,
# the "pyplot" and "seaborn" packages are imported.
# Both may consider useful for the model, because later they may be used to generate and display figures.

# First, importing the "pyplot" package for basic plots and histograms.

import matplotlib.pyplot as plt

# For advanced plotting and heatmaps, it is useful to import the "seaborn" package.

import seaborn as sns

# Since the scikit-learn package has been imported (see above, on the first cell)
# some extra features, coming from this package, will optionally be imported, 
# in order for them to be used, and will be used in the current model.

# For instance, the train and test split, will be used, 
# as long as each dataset will be trained and tested.

from sklearn.model_selection import train_test_split

# Same thing with cross-validation models and metrics.
# Such as the K-Fold Cross Validation process, which will be used later on the model,

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate

# and one for K-Fold process
# in this case, a Stratified K-Fold method will be used.

from sklearn.model_selection import StratifiedKFold

# feature selection techniques,
# as well as classification techniques, 
# used for later contributions to the model

from sklearn import tree
from sklearn import svm

# additional module for the decision tree classifier
# (in case it is done separately)

from sklearn.tree import DecisionTreeClassifier

# import the Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

# will be added

# as well as extras such as matrices

import sklearn.metrics

# the command above, will import necessary metrics on the model
# optionally, the confusion matrix metric will be imported (to avoid errors)
# and so the same with classification report metric

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# will be added

# Lastly, the XGBoost classifier will be imported, due to data classification
# in later contributions to the model.

import xgboost as xgb

# an attempt for reading the second dataset will each be made.
# In the below line of code, the dataR2.csv file will be stored into a dataframe, using a method from the pandas package.
# The dataframe for the second dataset, will be named into df_d2.
# It will be used for the second dataset given.

df_d2 = pd.read_csv('C:\\Users\\user\\Documents\\thesis\\files\\dataR2.csv',
                       sep=",",
                        decimal=".")
# Before getting info for the second dataset, the first five rows of the dataset will be read.
# It is an optional part; however, it will be useful to get necessary information for the dataset
# and its later procedures.

df_d2.head()

# Before printing the second dataframe, as long as the first one has been shown and printed
# it is optional to show info of the second dataframe.

# The second dataframe, follows as it is.
# The procedure, will remain as the first.

# It will display necessary information and content
# related to all of the second dataframe, 
# first five rows, having been imported just above.

df_d2.info()

# Later, the dataset will be printed on the screen
# regarding the content received above.

print (df_d2)

# Doing so, for the second dataset and its dataframe.

# It is also optional to display it as a statistical distribution.
# So that, before applying training and testing methods, 
# it would be essential to have a look at it much more detailed.

df_d2.describe()

# Before the train and test set procedure takes place,
# the 'Classification' column will be checked and distributed.

# First, count how many values are there on the 'Classification' column.

# can be performed, either, using the value_counts() method
# under: class_distr = df_d1['Classification'].value_counts()

# or declare a separate variable for the total amount of values.
# (and perform the total() method for the final check)

# total_classification = df_d2['Classification'].count()

# In this case, a separate variable for the total values of the 'Classification' column will be declared.

total_classification = df_d2['Classification'].count()

# Then, count how many values are there for the classification value set to '1'
# For this case, it will be detected on which rank the values may belong.
# Hypotheticly, let's pretend the values set to '1', stand for "breast cancer negative" diagnosis,
# which means, the patient does not have breast cancer.

neg_count = df_d2[df_d2['Classification'] == 1].shape[0]

# Next step, is to count how many values are there for the classification value set to '2'
# For this case, it will be detected on which rank the values may belong.
# Hypotheticly, let's pretend the values set to '2', stand for "breast cancer positive" diagnosis,
# which means, the patient does have breast cancer.

pos_count = df_d2[df_d2['Classification'] == 2].shape[0]
# Once all values have been counted, the current objective is to print them,
# and see the results of each value registered.

# Beginning with all values in total 
# (positive and negative, 1 and 2, altogether)

print("Total values are: ", total_classification)

# then carry on with each value individually

# negative values (or values set to 1)

print("Negative values are: ", neg_count)

# positive values (or values set to 2)

print("Positive values are: ", pos_count)

# The second (and current) dataset has been read.

# Information and contents of the second dataset have been fetched.
# An attempt to pre-process it, will be made.

# It will be split into a train and test set,
# in order for the values to be trained (and tested, each)
# for the train-test process to take place.

# It will be split into labels and features.
# Features, are represented under the x variable. (in this case, x2, as it stands for the second dataset variable)
# Labels (aka the target variable) are represented under the y variable. (in this case, y2, as it stands for the second dataset y variable)

# On the command below, the x2 and y2 variables, will determine each, features and labels.

# solution without converting each value of the 'Classification' column into numerical values

x2 = df_d2.drop('Classification', axis=1) # Features' variable
y2 = df_d2['Classification'] # Target variable

# (for the y1 variable, we can also declare y1 = df_d1.diagnosis without putting the column name in brackets)
# The next step, is to split the data into training and testing set.
# Features and labels have been represented and declared in two variables each;
# x for the features, and y for the labels/target variable.

# in the above case, x2 for the features
# and y2 for the labels of the df_d1 dataframe.

# Once split into labels and features, its logic will be implemented 
# on a scale of 90-10; meant by, 90% for training, and 10% for testing.
# Its test size, will be set to 0.1.

# Out of 116 rows and 10 columns, the random state is set to 42 by default.
# There will not be any further change to its training value.

# In this process, the data will have to be split into training and testing variables
# under x2_train, x2_test, for the x axis
# and y2_train, y2_test, for the y axis.

# the train_test_split function, has been imported above,
# and before reading the datasets in CSV format.

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y2, test_size=0.1, random_state=25)
# Optionally, the train and test values could be printed.
# Either as a set (each for the x1 and y1 variables),
# or individually

# In this case, set values will be examined.
# Examining the first dataset, therefore the train and test variables, 
# have been declared as x2 and y2.

print("x2 train shape is: ", x2_train.shape)
print("x2 test shape is: ", x2_test.shape)

print("y2 train shape is: ", y2_train.shape)
print("y2 test shape is: ", y2_test.shape)

# After the decision tree has been made, an attempt on performing a stratified k-fold cross-validation process, will be made.
# Cross validation, however, requires support vector machines module, which may be used due to probabilistic values
# It doesn't only apply to the usual cross-validation process (without folding) 
# as well as the K-fold cross-validation (pure) and stratified K-fold cross-validation processes.

# In this case, a stratified ten-fold cross-validation process is being examined.
# Which means, the K variable will be set to 10 as its value.
# (the K variable stands for folding)

# Optionally, the shapes for the x2 and y2 variables, will be printed.

# x2.shape
# y2.shape

print(x2.shape)
print(y2.shape)

# Stratified K-Fold process held for the second dataset.
# Splits will be set to 10 initially.

skf2 = StratifiedKFold(n_splits=10, shuffle=True)

# Get the number of splits for the process.

skf2.get_n_splits(x2, y2)

# Print the status of the stratified K-Fold process 
# before the actual process takes place.

print(skf2)

# initialize values used for true positives, true negatives
# false positives and false negatives

# each value will be set to 0

# begin with true positives

tp2_total = 0

# false positives

fp2_total = 0

# true negatives

tn2_total = 0

# false negatives

fn2_total = 0
# After training, testing and performing a k-fold cross validation process on the second dataframe's contents,
# an attempt on creating decision trees for each fold generated, will be made.

# Beginning with (and examining) Decision Trees.
# As long as the necessary modules have been imported, the model will now be defined for the experiment to take place.

# This case will be featured inside a loop, 
# so that the tree will be plotted right after the stratified K-fold cross-validation module
# having already taken place.

# Since none of the regression methods worked, the next attempt will be made on creating decision tree(s) for each output.
# (catches an error without converting any value of the 'diagnosis' column into numerical values)
# this case will examine the trained data

# define the decision tree classifier
# and initialize BEFORE the loop

dtc2 = tree.DecisionTreeClassifier()
# The actual process held for the second dataset, after printing all necessary stats.

for i, (train_index, test_index) in enumerate(skf2.split(x2, y2)):

    # first, print the amount of folds
    print(f"Fold {i+1}:")

    # print the train indices
    print(f"  Train: index={train_index}")

    # print the test indices
    print(f"  Test:  index={test_index}")

    # fit the classifier (after initialization)
    # assign a new variable named dtc2_train as it appeals to the trained data
    # on both x2 and y2 train sets

    dtc2_train = dtc2.fit(x2_train, y2_train)

    # next step is to make predictions on the test data

    y2_pred = dtc2.predict(x2_test)

    # next up, create a confusion matrix

    cm2 = confusion_matrix(y2_test, y2_pred)

    # optionally use classification report (much more detailed)

    clr2 = classification_report(y2_test, y2_pred)

    print (clr2, "\n")
    
    # extract all variables set for true positives, true negatives, false positives and false negatives
    # (generic use)

    tp2, fp2, tn2, fn2 = cm2.ravel()

    #print for each fold

    print(f"Confusion matrix for Fold {i+1}: \n", cm2)

    # calculate metrics for each fold
    # use a new variable

    tp2_total += tp2
    fp2_total += fp2
    tn2_total += tn2
    fn2_total += fn2

    # next up, calculate metrics for accuracy, precision, recall and F-score
    # on what was calculated during the first set of the ten-fold cross-validation process.

    #accuracy
    acc2 = accuracy_score(y2_test, y2_pred)
    print(f"Accuracy for fold {i+1}: ", acc2)

    # precision

    pr2 = precision_score(y2_test, y2_pred, average=None)
    print(f"Precision for fold {i+1}: ", pr2)

    # recall

    rec2 = recall_score(y2_test, y2_pred, average=None)
    print(f"Recall for fold {i+1}: ", rec2)

    # F1-score

    fsc2 = f1_score(y2_test, y2_pred, average=None)
    print(f"F1-score for fold {i+1}: ", fsc2)