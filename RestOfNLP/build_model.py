
# build-model.py
# Using multiple linear regression to
# generate weights for classifying surnames
# September 2020
# Seth Flores and Jake Lasley

import numpy as np
from string import ascii_lowercase
import sys

def getUnigramVector(word):
    word = word.lower()
    uv = { s:0 for s in ascii_lowercase }
    for letter in word:
        try:
            uv[letter] += 1
        except KeyError:
            print('KeyError when processing: ', letter, ' ... ignoring', file=sys.stderr)
            continue
    normalized =  [ count/len(word) for count in list(uv.values()) ] 
    return normalized

if __name__ == "__main__":
#    if len(sys.argv) != 3:
#        print("Usage: python build-model.py " +
#              "<input file>  <Nationality>" )
#        sys.exit()


    # Turning names into vectors
    surname_vectors = []
    class_ = []
    with open("surnames-dev.csv", mode="r", encoding="utf-8") as input_file:
        target = "Output_3"
        # Create 2d list of surname unigram vectors
        # and 2d list of classifications for those vectors
        for line in input_file:
            temp = line.strip().split(",")
            surname = temp[0]
            nationality = temp[1]
            surname_vectors.append(getUnigramVector(surname))
            if nationality == target:
                class_.append(1)         # 1 ==> Russian
            else:
                class_.append(0)

    # Linear Regression
    # Convert lists to np.matrix like
    surname_mat = np.array(surname_vectors)
    class_mat = np.array(class_)

    weights, residuals, rank, s = np.linalg.lstsq(surname_mat, class_mat, rcond=None)
    print(np.shape(weights))
    fname = target + "-weights"
    np.save(fname, weights)
    print("Weights saved to ", fname, ".npy")
                
