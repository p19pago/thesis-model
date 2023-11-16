## thesis-dss

A repository where I will share my progress out of my experiment, as part of my thesis for the academic year of 2023-2024.

# About

The topic I have chosen for my thesis, is develop a **Decision Support System (DSS)** and **mobile application** on Breast Cancer diagnoses.

# Progress

**Tuesday, October 24th, 2023:** Introduction to thesis. Download and use two (2) datasets

## This is the first checkpoint.

For the first check-point of my thesis experiment, an introduction was made on Decision Support Systems. 

Two (2) datasets, for the model, were downloaded and used.

  * one, from the [Kaggle official website](https://kaggle.com)
  * one, obtained externally.

**Wednesday, November 1st, 2023:** Further work on the model and implement its logic.

## This is the second checkpoint.

Having worked on the first steps of the model, the next task is implement the logic of the model.
<br>
In brief, find how many columns are used on each dataset *[on both their CSV and XLS format(s)]* .

Then, assign which classification method will be applied, for each column of both datasets.
<br>
*(or on each dataset, depending on its effectiveness)*

Describing and taking notes on each dataset's contents, I realized each dataset, ought to be analyzed, one by one.

### Progress on the second checkpoint

**Sunday, November 12th, 2023:** Select which datasets should be split into training and testing sets.
<br>
Assign train and test set(s) each.

For the first dataset, I assigned a x and y variable set *(each x1 and y1, representing the first dataset)*, in order for me to perform the final training and testing sets.

As long as the procedure had taken place, to make sure it has worked well and with important success, it was optional to print the output of the dataset (in total).
<br>
Specificly, how many rows and columns are there.

**Monday, November 13th, 2023:** Add new cells or rework specific cells of the model, import necessary modules, perform classification techniques.

Before the third checkpoint, I gave in a more analytic look of the dataframe(s) included in the model.
Which means, more detailed description and first look on the model's features; 

   * **what data it provides**
   * **what is useful for me to experiment with**,

hands-on.

### Distribution and Value Count

After describing the dataframe, before performing the train and test procedure, I performed a distribution of the `diagnosis` column on the first dataset;
calculated how many values are in total, malignant *(represented by 'M')* and benign *(represented by 'B')* as individuals.

The distribution did calculate how many values there are for both options *('M' and 'B')*, using a sum.
<br>
Same procedure for the total size of the column.

In this case, no sum was used.
Instead, I went by using a count() built-in method in Python.

Let alone it was effective enough to have a detailed image on the amount of diagnoses on our dataframe.

### Modules & Extras Import

For the procedures of training-testing, classification, feature elimination and subset selection, 


# Where can I find files?

Each
