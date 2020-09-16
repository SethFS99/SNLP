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
    try:
        return tp/(tp + fp)
    except ZeroDivisionError:
        return 0
        

def recall(tp, fp, tn, fn):
    try:
        return tp/(tp + fn)
    except ZeroDivisionError:
        return 0

def predict(test_data, thresholds, target):
    """
    test_data: n x 3 array like where
               td[n] is an iterable of the form 
               [ surname, nationality, confidence ]
    thresholds: iterable with floats on [0,1]
    target: string of nationality for which the
            confidence was computed
    """
    # precision and recall for <Target> surnames
    # positive class -> surname is <target>
    # negative class -> surname is NOT <target>
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    results = []
    for threshold in thresholds:
        for temp in test_data:
            surname = temp[0]
            nationality = temp[1]
            confidence = temp[2]
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
        # per threshold
        p = precision(tp, fp, tn, fn)
        r = recall(tp, fp, tn, fn)
        results.append([threshold, tp, fp, tn, fn, p, r])
    return results



if __name__ == "__main__":

    # pre-process only -> python predict.py <input file> <output(confidence)> <weights> [--pre]
    # one-shot         -> python predict.py <test data(input)> <output(confidence)> <weights> <threshold>
    # multi threshold  -> python predict.py <test data(input)> <output(pre/recall)> <target>  <threshold> [,<threshold> ...] --multi
    if len(sys.argv) < 4:
        print("Usage: python predict.py " +
              "<input file> <output(confidence)> <weights> --pre\n"+
              "or\n" +
              "<test data(input)> <output(confidence)> <weights> <threshold>\n"+
              "or\n" +
              "<test data(input)> <output(pre/recall)> <target>  <threshold> [,<threshold> ...] --multi\n")
        sys.exit()

    if "--multi" in sys.argv: # multiple thresholds
        with open(sys.argv[1], mode="r", encoding="utf-8") as input_file, \
             open(sys.argv[2], mode="w", encoding="utf-8") as output_file:
            target = sys.argv[3]
            thresholds = [ float(x) for x in sys.argv[4:len(sys.argv)-1] ]
            test_data = []
            for line in input_file:
                temp = line.strip().split(",")
                test_data.append([ temp[0], temp[1], float(temp[2]) ])
            results = predict(test_data, thresholds, target)
            output_file.write("threshold,tp,fp,tn,fn,precision,recall\n")
            for line in results:
                output_file.write("{},{},{},{},{},{},{}".format(
                    line[0],line[1],line[2],line[3],
                    line[4],line[5], line[6]))
                output_file.write("\n")
            sys.exit()

    with open(sys.argv[1], mode="r", encoding="utf-8") as input_file, \
            open(sys.argv[2], mode="w", encoding="utf-8") as output_file:
        weights = np.load(sys.argv[3])
        target = sys.argv[3].split('-')[0] # get target surname from weights file name
        test_data = []
        for line in input_file:
            temp = line.strip().split(",")
            surname = temp[0]
            nationality = temp[1]
            s_vec = getUnigramVector(surname)
            confidence = np.matmul(s_vec, weights)
            test_data.append([temp[0], temp[1], confidence ])

        if "--pre" in sys.argv:
            for line in test_data:
                output_file.write("{},{},{}".format(line[0],line[1],line[2]))
                output_file.write("\n")
            sys.exit()

        results = predict(test_data, [threshold], target)
        for line in results:
            for value in line:
                print("{},".format(value))
            print("\n")
