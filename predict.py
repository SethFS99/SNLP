#!/usr/bin/env python3

# predict.py
# Using weights from build-model.py and a confidence threshold
# predict the class of surnames.
# September 2020
# Seth Flores and Jake Lasley

import numpy as np
import sys
import os
sys.path.append(os.path.relpath("./"))
from build_model import getUnigramVector

def precision(tp, fp, tn, fn):
    return tp/(tp + fp)


def recall(tp, fp, tn, fn):
    return tp/(tp + fn)


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python predict.py " +
              "<test data> <weights file> <threshold> <output file>" )
        sys.exit()

    ### handle arguments
    threshold = float(sys.argv[3])
    # read in weights and get target nationality
    weights = np.load(sys.argv[2])
    target = sys.argv[2].split('-')[0] # get target surname from weights file name
    # precision and recall for <Target> surnames
    # positive class -> surname is <target>
    # negative class -> surname is NOT <target>
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    with open(sys.argv[1], mode="r", encoding="utf-8") as input_file, \
         open(sys.argv[4], mode="w", encoding="utf-8") as output_file:

        output_file.write("threshold: ")
        output_file.write(str(threshold))
        output_file.write("\n\n")
        
        # Create 2d list of surname unigram vectors
        # and 2d list of classifications for those vectors
        for line in input_file:
            temp = line.strip().split(",")
            surname = temp[0]
            nationality = temp[1]
            s_vec = getUnigramVector(surname)
            confidence = np.matmul(s_vec, weights)
            prediction = None
            if confidence > threshold:
                prediction = target
            else:
                prediction = "Not " + target

            if nationality == target: 
                if prediction == target: 
                    tp += 1
                else:                       
                    fn += 1
            else:
                if prediction == target:
                    fp += 1
                else:
                    tn += 1
            
            output_file.write(surname)
            output_file.write(",")
            output_file.write(nationality)
            output_file.write(",")
            output_file.write(str(prediction))
            output_file.write(",")
            output_file.write(str(confidence))
            output_file.write("\n")

    print("tp: {} fp: {} tn: {} fn: {}".format(tp, fp, tn, fn))
    print("precision: {}".format(precision(tp, fp, tn, fn)))
    print("recall: {}".format(recall(tp, fp, tn, fn)))
