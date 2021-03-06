# SNLP
All of my speech and natural language processing code
This repository will be holding the project for identifying the nationality of someone based on the names

# build_model.py
This script takes in a set of training data with lines of the form

    surname,nationality

as in surnames-dev.csv.
It outputs a vector of weights produced by multiple linear regression to a .npy file.
This file of weights can then be read into another script for use in classification of surnames.

# predict.py
usage:

    python predict.py <test data> <weights file> <threshold> <output file>
	
The test data should be the _surnames-test.csv_ file. 
The weights file is a .npy file generated by `build_model.py`.
Don't rename the file because it gets the nationality from there.
Threshold: float [0-1]
output file: comma separated values, so probably good to call it something .csv

Right now the output file looks like,

    surname,actual nationality,Russian/Not Russian,confidence value
	
where surname, actual nationality, and russian/not russian are strings,
and the confidence value is a float.

Currently it uses the threshold passed from the command line to make a prediction.
After making a prediction, it evaluates if the prediction is tp, fp, tn, or fn.

After making predictions for the whole test set, it calculates precision and recall.
