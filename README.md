# SNLP
All of my speech and natural language processing code
This repository will be holding the project for identifying the nationality of someone based on the names

# build-model.py
This script takes in a set of training data with lines of the form

    surname,nationality

as in surnames-dev.csv.
It outputs a vector of weights produced by multiple linear regression to a .npy file.
This file of weights can then be read into another script for use in classification of surnames.
