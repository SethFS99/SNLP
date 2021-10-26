# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 11:35:44 2020

@author: sethf
"""

import numpy as np
import math

def loadGloveModel(gloveFile):   # from Karishma Malkan on stackoverflow 
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print( "Done.",len(model)," words loaded!")
    return model

def distance(vec1, vec2):
  return math.sqrt(sum([(x-y)*(x-y) for x,y in zip( vec1, vec2)]))
   
def mostSimilar(query):
  distances = [distance(model[x], model[query]) for x in model.keys()[:200000]]
#  distances = [distance(model[x], model[query]) for x in model.keys()]
  minval = min(distances)
  print( 'for %s: min distance of %.2f found for %s' % (query, minval, model.keys()[distances.index(minval)]))

#---------------------------------
model = loadGloveModel('glove.6B.50d.txt')  # 17 sec on my desktop
#print model['texas-el']

mostSimilar('texas')
