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
from sklearn.model_selection import cross_val_score

# as well as classification techniques, 
# used for later contributions to the model

from sklearn import tree
from sklearn import svm

# will be added

# as well as extras such as matrices

import sklearn.metrics

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

df_d1 = pd.read_csv('C:\\Users\\DI-LAB\\Documents\\thesis-main\\files\\data_upd.csv',
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

# Next up, an attempt for reading the second dataset will each be made.
# In the below line of code, the dataR2.csv file will be stored into a dataframe, using a method from the pandas package.
# The dataframe for the second dataset, will be named into df_d2.
# It will be used for the second dataset given.

df_d2 = pd.read_csv('C:\\Users\\DI-LAB\\Documents\\thesis-main\\files\\dataR2.csv',
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

# Doing so, for the second dataset and its dataframe.

# It is also optional to display it as a statistical distribution.
# So that, before applying training and testing methods, 
# it would be essential to have a look at it much more detailed.

df_d2.describe()

# Before the train and test set procedure takes place,
# the 'diagnosis' column will be checked and distributed.

# First, count how many values are there on the 'diagnosis' column.

# can be performed, either, using the value_counts() method
# under: diagnosis_distr = df_d1['diagnosis'].value_counts()

# or declare a separate variable for the total amount of values.
# (and perform the total() method for the final check)

# total_diagnosis = df_d1['diagnosis'].count()

# In this case, a separate variable for the total values of the diagnosis column will be declared.

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

# As long as the first (and second) datasets have been read,
# information and contents have been fetched,
# an attempt to pre-process them, will be made.

# Beginning with the first dataset.
# The first dataset, will be split into train and test sets,
# in order for the values to be trained. (and tested, each)

# Both datasets, will be split into labels and features.
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
# on a scale of 80-20; meant by, 80% for training, and 20% for testing.
# Its test size, will be set to 0.2.

# Out of 569 rows and 31 columns, the random state is set to 42 by default.
# There will not be any further change to its training value.

# In this process, the data will have to be split into training and testing variables
# under x1_train, x1_test, for the x axis
# and y1_train, y1_test, for the y axis.

# the train_test_split function, has been imported above,
# and before reading the datasets in CSV format.

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y1, test_size=0.2, random_state=0)

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

# After training and testing the first dataframe's contents, 
# an attempt on performing algorithms on classification, will be made.

# Beginning with Decision Trees.
# As long as the necessary modules have been imported,
# the model will now be defined for the experiment to take place.

# Since none of the regression methods worked, the next attempt will be made on creating decision tree for each output.
# (catches an error without converting any value of the 'diagnosis' column into numerical values)

clf = tree.DecisionTreeClassifier()
clf = clf.fit(x1, y1)

# Once trained, next step on this cell is to create a decision tree
# and then, optionally, show a plot of it.

tree.plot_tree(clf)

# After the decision tree has been made, an attempt on performing a cross-validation process, will be made.
# Cross validation, however, requires support vector machines module, which will not be used due to probabilistic values

# Optionally, the shapes for the x1 and y1 variables, will be printed.

# x1.shape
# y1.shape

print(x1.shape)
print(y1.shape)

# The train and test process has already taken place.
# Next step within this cell is to perform cross-validation,
# and calculate the final score.

crv = svm.SVC(kernel='linear', C=1,).fit(x1_train, y1_train)

# and then, the score will be calculated (and printed on the screen)

crv.score(x1_test, y1_test)

# Optionally, calculate the cross-validation score.

crv_score = cross_val_score(crv, x1, y1, cv=5) # cross_val_score(clf, X, y, cv=5)

# (it took that long, and let's be honest, without a single crashing or error)

# Optionally, print it...

print('Cross-validation score value, is: ', crv_score)

# ... as well as print any possible mean, standard deviation or variance values.

print('Cross-validation mean value, is: ', crv_score.mean())

print('Cross-validation standard deviation value, is: ', crv_score.std())

print('Cross-validation variance value, is: ', crv_score.var())

# optionally, perform any possible precision, recall, F-score, or confusion matrix values