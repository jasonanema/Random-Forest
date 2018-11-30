# Random-Forest

This repository contains python code for a decision tree and a random forest classifier which I wrote from scratch as a project in a Data Mining course I took at UIUC in Fall 2017 (CS 412). I wrote these to work on the type of data sets that are shared in this file (note that there are both testing and training data sets for each).

Used Python version 2.7.6 and only used built-in libraries (datetime, random, sys).

Read report.pdf for details, such as: Brief introduction to classification methods and classification framework, Model Evaluation Measures Parameters chosen during implementation and why, & Conclusion on whether ensemble method improves performance of the basic classification method I chose, why or why no

You can run either decision tree or random forest on the various sets of data with one of:

python RandomForest.py train_file test_file OR python DecisionTree.py train_file test_file

The output (to stdout) is k by k numbers (k numbers per line, separated by a space, k lines in total) representing the confusion matrix of each classier on testing data, where k is the count of class labels. Various parameters have been toned for each method for each of the four training data sets, as described in report.pdf.

The data sets are all JSON format where each line contains an instance and is ended by a '\n' character. <label> is an integer indicating the class label. The pair <index>: <value> gives a feature (attribute) value: <index> is a non­negative integer and <value> is a number (we only consider categorical attributes). Note that one attribute may have more than 2 possible values, meaning it is a multi-value categorical attribute.
