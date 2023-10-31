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
# The pandas module will use of need for the Decision Support System I am designing and implementing,
# as well as the CSV datasets the program will fetch to be read.

import pandas as pd

# Most experiments, will base on machine learning and data science.
# The scikit-learn module will be called and imported using the "sklearn" abbreviation
# and by importing it, all features will be granted access.

import sklearn

# Lastly, the "random" package is imported.
# Occasionally, some sequences may feature random numbers and/or time.
# Therefore, it is considered useful.

import random

# After importing all features, the program will fetch the datasets required
# In this part, the read() module from the pandas package will be used, under "pd.read()"
# An attempt for reading the first dataset (data.csv) will be made.
# In the below line of code, the data.csv file will be stored into a dataframe, using a method from the pandas package.

# The dataframe for the first dataset, will be named into df_d1.
# It will be used for the first dataset.

df_d1 = pd.read_csv("data.csv")

# Next up, an attempt for reading the second dataset will each be made.
# In the below line of code, the dataR2.csv file will be stored into a dataframe, using a method from the pandas package.
# The dataframe for the second dataset, will be named into df_d2.
# It will be used for the second dataset given.

df_d2 = pd.read_csv("dataR2.csv")