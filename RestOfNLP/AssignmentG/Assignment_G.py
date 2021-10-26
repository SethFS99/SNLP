# Nigel Ward, September 2018
# Code Skeleton for SLP  Assignment D: Sentiment Analysis
import spacy
import numpy as np    # might instead use scipy or scikit-learn
import re, sys,os

interestingWords = ["great", "good", "bad", "still", "awful", "amazing", "fantastic","worst", "terrible"]
interestingPairs = ["not bad", "not good"]
interestingClasses = ["JJ", "NN", "8L"]
def getLexicon():
    negLexicon =[]
    posLexicon =[]
    lexiconFile = open(os.getcwd()+"/MPAQ_Lexicon.txt",  mode="r", encoding="utf-8")
    for line in lexiconFile.readlines():
        ls = line.split(" ")
        wordRating = []
        wordRating.append((ls[2])[6:])#word stored here
        wordRating.append((ls[-1])[14:-1])#rating stored here
        if wordRating[1] == "positive":
            posLexicon.append(wordRating[0])
        if wordRating[1] == "negative":
            negLexicon.append(wordRating[0])

    return negLexicon,posLexicon
def wordC(reviewString,negLex,posLex):
    x ={}
    for i in negLex:
        x[i] =0
    for j in posLex:
        x[j] = 0
    words = re.split("[\|/|;|,|.|!|?"+ '|"|'+" |-|:|\n|\s|\t|'|&|(|)]\s*",reviewString)
    for w in words:#more tokenizing
          if len(w) == 0:
              continue#blank words
          if '-' in w:#hyphens aren't always tokenized
              nw = w.split('-')      
              for i in nw:#back to normal tokenizing
                  if len(nw) == 0:
                      continue
                  if i in x:
                      x[i]+=1
          else:
              if w in x:
                  x[w]+=1

    wc = [k for k in x.values()]
    return wc
    
def featurizeReview(reviewString,negLex,posLex):#Will be modified to become naieve bayes/use lexicons
    #tokenize the string
    feats = np.zeros(3)
    feats[0] = 1# a constant term, for lstsq
    #feats 1 is pos, 2 neg, 3 neutral
    words = re.split("[\|/|;|,|.|!|?"+ '|"|'+" |-|:|\n|\s|\t|'|&|(|)]\s*",reviewString)
    for w in words:#more tokenizing
          if len(w) == 0:
              continue#blank words
          if '-' in w:#hyphens aren't always tokenized
              nw = w.split('-')      
              for i in nw:#back to normal tokenizing
                  if len(nw) == 0:
                      continue
                  if i in posLex:
                      feats[1]+=1
                  if i in negLex:
                      feats[2] +=1
          else:
              if w in posLex:
                  feats[1]+=1
              if w in negLex:
                  feats[2] +=1

    

    return feats

def readReviewSet(directory,negLex,posLex):
    reviewsFileName = "%s/subj.%s" % (directory, directory)
    ratingFileName = "%s/rating.%s" % (directory, directory)  
    reviewsFp = open(reviewsFileName,  mode="r", encoding="utf-8")
    ratingsFp = open(ratingFileName,  mode="r", encoding="utf-8")
    featureMatrix = []
    wc=[]
    for review in reviewsFp.readlines():
        feats = featurizeReview(review,negLex,posLex)
        wc.append(wordC(review,negLex,posLex))
        featureMatrix.append(feats)
    ratingVec = []
    for label in ratingsFp.readlines():
      ratingVec.append(float(label))
    return featureMatrix, ratingVec,wc

def buildModel(features, targets):  # least squares
    f = np.array(features)
    t = np.array(targets)
    model, residuals, rank, svs = np.linalg.lstsq(f, t)
    print("The model is: ")
    print(model)
    return model

def applyModel(model, features):
   predictions = np.matmul(features, model)
   return predictions

def printInfo(index, label, pred, data):
   print("For story #", index, ":", "label=", label, "pred=", pred)
   print("Features: ")
   print(data)

def evalPredictions(predictions, targets, data, descriptor):
   deltas = predictions - targets
   mse = np.matmul(deltas, deltas) / len(predictions)
   print("For ", descriptor, "mse is ", mse)
   mae = sum(abs(deltas)) / len(predictions)
   print("and the mae is ", mae)

def showWorst(predictions, targets, data):
   deltas = predictions - targets
   worstUnder = np.argmin(deltas)
   worstOver = np.argmax(deltas)
   print("Worst underestimate and worst overestimate: ")
   printInfo(worstUnder, targets[worstUnder], predictions[worstUnder], data[worstUnder])
   printInfo(worstOver, targets[worstOver], predictions[worstOver], data[worstOver])
def correctLabels(labels):
    for i in range(len(labels)):
        if labels[i] < 0.1 :
            labels[i] = 0.0
        elif labels[i] < 0.2 :
            labels[i] = 0.1
        elif labels[i] < 0.3 :
            labels[i] = 0.2
        elif labels[i] < 0.4 :
            labels[i] = 0.3
        elif labels[i] < 0.5 :
            labels[i] = 0.4
        elif labels[i] < 0.6 :
            labels[i] = 0.5
        elif labels[i] < 0.7 :
            labels[i] = 0.6
        elif labels[i] < 0.8 :
            labels[i] = 0.7
        elif labels[i] < 0.9 :
            labels[i] = 0.8
        elif labels[i] < 1 :
            labels[i] = 0.9
        else:
            labels[i] = 1
    return labels
#=== main ========================================================
negLex,posLex = getLexicon()
trData, trLabels,wcData = readReviewSet("James+Berardinelli",negLex,posLex) # lazily omit Dennis 
#trData, trLabels = readReviewSet("Dennis+Schwartz") # lazily omit Dennis 
#trData, trLabels = readReviewSet("Scott+Renshaw") # lazily omit Dennis 
#trData, trLabels = readReviewSet("Steve+Rhodes") # lazily omit Dennis
trLabels = correctLabels(trLabels) 
model = buildModel(trData, trLabels)
model2=buildModel(wcData,trLabels)
if len(sys.argv) >= 2 and (sys.argv[1] == "yesThisReallyIsTheFinalRun"):
  testData, testLabels, testWc = readReviewSet("Steve+Rhodes",negLex,posLex)
else:
  testData, testLabels,testWc = readReviewSet("Scott+Renshaw",negLex,posLex)  # devtest
testLabels = correctLabels(testLabels)
predictions = applyModel(model, testData)
p = applyModel(model2, testWc)
evalPredictions(predictions, testLabels, testData, "dummy model")
evalPredictions(p, testLabels, testData, "BOW model")
showWorst(predictions, testLabels, testData)

baselinePredictions = np.ones(len(testLabels)) * np.average(testLabels)
evalPredictions(baselinePredictions, testLabels, testData, "baseline (average)")


# output

#the model is
#[ 0.60081999 -0.01981207 -0.08698933 -0.00183384  0.07853322  0.00191875]
#
#for dummy model, mse is 0.042
#  (and the mae is 0.164)
#worst underestimate and worst overestimate:
#   for story #272: label= 0.50, pred= -0.13
#      features:  [  1.   1.   9.   1.   0.  36.]
# understandable since the name of the movie is "very bad things"
#   for story #2: label= 0.00, pred= 0.66
#      features:  [  1.   1.   0.   0.   0.  40.]
# understandable since the word "good" appears, and not much else that's informative 

#for baseline (average), mse is 0.038
#  (and the mae is 0.157)