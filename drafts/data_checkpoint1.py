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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# will be added

# Lastly, the XGBoost classifier will be imported, due to data classification
# in later contributions to the model.

import xgboost as xgb
# After importing all features and extras, the program will read and fetch the datasets,
# required for the experimental model in progress.

# In this part, the read() module from the pandas package will be used, under "pd.read()"
# An attempt for reading the first dataset (data.csv) will be made.
# In the below line of code, the data.csv file will be stored into a dataframe, using a method from the pandas package.

# The dataframe for the first dataset, will be named into df_d1.
# It will be used for the first dataset.

# For the first dataframe, the file used to be fetched and read, was slightly altered
# the "ID" column was removed from the file, so that the prompt used later, 
# will not read the patients' ID number.

df_d1 = pd.read_csv('C:\\Users\\user\\Documents\\thesis\\files\\data_upd.csv',
                        sep=",",
                        decimal=".")

# Before getting info for the first dataset, the first five rows of the dataset will be read.
# It is an optional part; however, it will be useful to get necessary information for the dataset
# and its later procedures.

df_d1.head()

# The output above, shows the first five rows of the first dataframe.

# Before printing the first (and later, the second) dataframe, 
# it is optional to show information and contents of each dataframe.

# Starting with the first dataframe.

# The command below, will display necessary information and content
# related to the first dataframe, 
# first five rows, having been imported just above.

df_d1.info()

# Later, the dataset will be printed on the screen
# regarding the content received above.

print (df_d1)

# It is also optional to display the dataframe as a statistical distribution.
# So that, before applying training and testing methods, 
# it would be essential to have a look at it much more detailed.

df_d1.describe()

# Before the train and test set procedure takes place,
# the 'diagnosis' column will be checked and distributed.

# First, count how many values are there on the 'diagnosis' column.

# can be performed, either, using the value_counts() method
# under: diagnosis_distr = df_d1['diagnosis'].value_counts()

# or declare a separate variable for the total amount of values.
# (and perform the total() method for the final check)

# total_diagnosis = df_d1['diagnosis'].count()

# In this case, a separate variable for the total values of the 'diagnosis' column will be declared.

total_diagnosis = df_d1['diagnosis'].count()

# Then, count how many values are there for the 'M' (malignant) label.

mal_count = (df_d1['diagnosis'] == 'M').sum()

# And at the end, count the benign values.

ben_count = (df_d1['diagnosis'] == 'B').sum()
# Once all values have been counted, the current objective is to print them,
# and see the results of each value registered.

# Beginning with all values in total 
# (malignant and benign, altogether)

print("Total values are: ", total_diagnosis)

# then carry on with each value individually

# malignant values

print("Malignant values are: ", mal_count)

# benign values

print("Benign values are: ", ben_count)

# As long as the first datasets has been read,
# information and contents have been fetched,
# an attempt to pre-process it, will be made.

# It will be split into train and test sets,
# in order for the values to be trained. (and tested, each)
# for the train-test process to take place.

# It will be split into labels and features.
# Features, are represented under the x variable. (in this case, x1, as it stands for the first dataset)
# Labels (aka the target variable) are represented under the y variable. (in this case, y1, as it stands for the first dataset)

# On the command below, the x1 and y1 variables, will determine each, features and labels.

# solution without converting each value of the 'diagnosis' column into numerical values

x1 = df_d1.drop('diagnosis', axis=1) # Features' variable
y1 = df_d1['diagnosis'] # Target variable

# (for the y1 variable, we can also declare y1 = df_d1.diagnosis without putting the column name in brackets)
# The next step, is to split the data into training and testing set.
# Features and labels have been represented and declared in two variables each;
# x for the features, and y for the labels/target variable.

# in the above case, x1 for the features
# and y1 for the labels of the df_d1 dataframe.

# Once split into labels and features, its logic will be implemented 
# on a scale of 90-10; meant by, 90% for training, and 10% for testing.
# Its test size, will be set to 0.1.

# Out of 569 rows and 31 columns, the random state is set to 42 by default.
# There will not be any further change to its training value.

# In this process, the data will have to be split into training and testing variables
# under x1_train, x1_test, for the x axis
# and y1_train, y1_test, for the y axis.

# the train_test_split function, has been imported above,
# and before reading the datasets in CSV format.

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.1, random_state=0)
# Optionally, the train and test values could be printed.
# Either as a set (each for the x1 and y1 variables),
# or individually

# In this case, set values will be examined.
# Examining the first dataset, therefore the train and test variables, 
# have been declared as x1 and y1.

print("x1 train shape is: ", x1_train.shape)
print("x1 test shape is: ", x1_test.shape)

print("y1 train shape is: ", y1_train.shape)
print("y1 test shape is: ", y1_test.shape)

# An attempt on performing a stratified k-fold cross-validation process, will be made.
# Cross validation, however, requires support vector machines module, which may be used due to probabilistic values
# It doesn't only apply to the usual cross-validation process (without folding) 
# as well as the K-fold cross-validation (pure) and stratified K-fold cross-validation processes.

# In this case, a stratified ten-fold cross-validation process is being examined.
# Which means, the K variable will be set to 10 as its value.
# (the K variable stands for folding)

# Optionally, the shapes for the x1 and y1 variables, will be printed.

# x1.shape
# y1.shape

print(x1.shape)
print(y1.shape)

# Stratified K-Fold process held for the first dataset.
# Splits will be set to 10 initially.

skf1 = StratifiedKFold(n_splits=10, shuffle=True)

# Get the number of splits for the process.

skf1.get_n_splits(x1, y1)

# Print the status of the stratified K-Fold process 
# before the actual process takes place.

print(skf1)

# initialize values used for true positives, true negatives
# false positives and false negatives

# each value will be set to 0

# begin with true positives

tp1_total = 0

# false positives

fp1_total = 0

# true negatives

tn1_total = 0

# false negatives

fn1_total = 0
# After training, testing and performing a k-fold cross validation process on the first dataframe's contents,
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

dtc1 = tree.DecisionTreeClassifier()
# The actual process held for the first dataset, after printing all necessary stats.

for i, (train_index, test_index) in enumerate(skf1.split(x1, y1)):

    # first, print the amount of folds
    print(f"Fold {i+1}:")

    # print the train indices
    print(f"  Train: index={train_index}\n")

    # print the test indices
    print(f"  Test:  index={test_index}\n")

    # fit the classifier (after initialization)
    # assign a new variable named dtc_train as it appeals to the trained data
    # on both x1 and y1 train sets

    dtc_train = dtc1.fit(x1_train, y1_train)

    # next step is to make predictions on the test data

    y1_pred = dtc1.predict(x1_test)

    # next up, create a confusion matrix

    cm1 = confusion_matrix(y1_test, y1_pred)

    # optionally use classification report (much more detailed)

    clr1 = classification_report(y1_test, y1_pred)

    print(clr1, "\n")
    
    # extract all variables set for true positives, true negatives, false positives and false negatives
    # (generic use)

    tp1, fp1, tn1, fn1 = cm1.ravel()

    # print for each fold

    print(f"Confusion matrix for Fold {i+1}: \n", cm1)

    # calculate metrics for each fold
    # use a new variable
    
    tp1_total += tp1
    fp1_total += fp1
    tn1_total += tn1
    fn1_total += fn1
    
    # next up, calculate metrics for accuracy, precision, recall and F-score
    # on what was calculated during the first set of the ten-fold cross-validation process.

    # accuracy

    acc1 = accuracy_score(y1_test, y1_pred)
    print(f"Accuracy for fold {i+1}: ", acc1)
    
    # precision

    pr1 = precision_score(y1_test, y1_pred, average=None)
    print(f"Precision for fold {i+1}: ", pr1)

    # recall

    rec1 = recall_score(y1_test, y1_pred, average=None)
    print(f"Recall for fold {i+1}: ", rec1)

    # F1-score

    fsc1 = f1_score(y1_test, y1_pred, average=None)
    print(f"F1-score for fold {i+1}: ", fsc1)
