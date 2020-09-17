#!/usr/bin/env python3

# build-model.py
# Using multiple linear regression to
# generate weights for classifying surnames
# September 2020
# Seth Flores and Jake Lasley

import numpy as np
from string import ascii_lowercase
import sys
import itertools # for itertools.product()

# code fragments from class

def generate_bigrams(word):
    lower = word[:-1]
    upper = word[:1]
    bigram_gen = map(lambda l,u: l+u, lower, upper)
    for bigram in bigram_gen:
        yield bigram


def getBigramVector(word):
    word = word.lower()
    bv = {a+b:0 for (a,b) in itertools.product(
        ascii_lowercase, ascii_lowercase)}
    for bigram in generate_bigrams(word):
        try: 
            bv[bigram] += 1
        except KeyError:
            print('KeyError when processing: ', letter, ' ... ignoring', file=sys.stderr)
            continue
    normalized =  [ count/len(word) for count in list(bv.values()) ] 
    return normalized


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
    if len(sys.argv) != 3:
        print("Usage: python build-model.py " +
              "<input file>  <Nationality>" )
        sys.exit()

    target = sys.argv[2]

    # Turning names into vectors
    b_vectors = [] # normalized bigram frequency vectors
    u_vectors = [] # normalized unigram frequency vectors
    class_ = []
    with open(sys.argv[1], mode="r", encoding="utf-8") as input_file:
        # Create 2d list of surname unigram vectors
        # and 2d list of classifications for those vectors
        for line in input_file:
            temp = line.strip().split(",")
            surname = temp[0]
            nationality = temp[1]
            b_vectors.append(getBigramVector(surname))
            u_vectors.append(getUnigramVector(surname))
            if nationality == target:
                class_.append(1)         
            else:
                class_.append(0)

    # Linear Regression
    # Convert lists to np.matrix like
    b_mat = np.array(b_vectors)
    u_mat = np.array(u_vectors)
    class_mat = np.array(class_)

    b_weights, residuals, rank, s = np.linalg.lstsq(b_mat, class_mat, rcond=None)
    fname = target + "-bigram-weights"
    np.save(fname, b_weights)
    print("Weights saved to ", fname, ".npy")
                
    u_weights, residuals, rank, s = np.linalg.lstsq(u_mat, class_mat, rcond=None)
    fname = target + "-unigram-weights"
    np.save(fname, u_weights)
    print("Weights saved to ", fname, ".npy")
            

