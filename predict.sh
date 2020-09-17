#!/bin/sh

LANG="English"

python predict.py $LANG-bigram-conf.csv $LANG-bigram-stats.csv $LANG 0.{1..9} --multi
python predict.py $LANG-unigram-conf.csv $LANG-unigram-stats.csv $LANG 0.{1..9} --multi
