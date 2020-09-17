#!/bin/sh

LANG="English"
TRAIN_FILE="surnames-dev.csv"

# make the model
python build_model.py $TRAIN_FILE $LANG

# pre process
python predict.py $TRAIN_FILE $LANG-bigram-conf.csv $LANG-bigram-weights.npy --pre
python predict.py $TRAIN_FILE $LANG-unigram-conf.csv $LANG-unigram-weights.npy --pre
