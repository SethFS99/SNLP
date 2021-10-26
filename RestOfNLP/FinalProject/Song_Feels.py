# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 19:21:44 2020

@author: sethf
"""
#Reused code from assignment G
import lyricsgenius as genius
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_absolute_error
import numpy as np
from LyricReader import readLyrics
import re, spacy,os,math

#code from Dr. Nigel Ward, University of Texas at El Paso
def loadGloveModel():   # from Karishma Malkan on stackoverflow 
    print("Loading Glove Model")
    f = open('glove.6B.50d.txt','r', encoding = 'utf-8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

def makeStr(review):
    s=""
    for i in review:
        s+=" "+i
    return s
def reverseEmbed(embedding):
    # print(embedding)
    for i in range(len(embedding)):
        embedding[i]=embedding[i]*-1#invert embeddings
    # print(embedding,"REVERSED")
    return list(embedding)
       
def reviewsToEmbeddings1D(Lyrics, sentiment, testLyrics, testSentiment):
    model = loadGloveModel()
    nlp = spacy.load('en_core_web_sm')
    #read songs to use as train data
    
    Lyrics = np.array(Lyrics)
    testLyrics = np.array(testLyrics)
    sentiment = np.array(sentiment)#cast lyrics and sentiment to np arrays to be able to do nice tricks with them
    testSentiment = np.array(testSentiment)
    Tsongs=""
    TrSongs=""

    for i in Lyrics:
        Tsongs+=i+"\n"
    for i in TestLyrics:
        TrSongs+=i+"\n"#append appropriate items to the songs
    f1 = open(os.getcwd()+"/TrainSongs.txt",mode="w")
    f1.write(Tsongs)
    f1.close()
    f2 = open(os.getcwd()+"/TestSongs.txt",mode="w")
    f2.write(TrSongs)
    f2.close()
    with open(os.getcwd()+"/TrainSongs.txt") as f1, open(os.getcwd()+"/TestSongs.txt") as f2:
        train_X_strings = f1.read().splitlines()
        test_X_strings = f2.read().splitlines()
        #tokenize the songs
        
    
    for i in range(len(train_X_strings)):
        train_X_strings[i] = re.findall(r'(\b[bcdfghj-np-tv-z]*[aeiou]+[bcdfghj-np-tv-z]*)\b', makeStr(train_X_strings[i]))
    for i in range(len(test_X_strings)):
        test_X_strings[i] = re.findall(r'(\b[bcdfghj-np-tv-z]*[aeiou]+[bcdfghj-np-tv-z]*)\b', makeStr(test_X_strings[i]))
    train_X=[]
    for song in train_X_strings:
        song_txt = makeStr(song)
        current_doc = nlp(song_txt)
        reviewEmbeddings=[]
        for word in range(len(song)):
                #find next adj or verb to modify
            embedding = model.get(song[word])
            if embedding is not None:
                reviewEmbeddings += list(embedding)
        #average all word embedding values into a single mean per review, making this a linear regression model with one parameter.
        train_X.append(np.mean(reviewEmbeddings))
    
    train_X = np.array(train_X)
    
    
    
    #--------------------------------------
    
    
    test_X=[]
    for song in test_X_strings:
        song_txt = makeStr(song)
        current_doc = nlp(song_txt)
        reviewEmbeddings=[]
        for word in range(len(song)):
                #find next adj or verb to modify
            embedding = model.get(song[word])
            if embedding is not None:
                reviewEmbeddings += list(embedding)
        #average all word embedding values into a single mean per review, making this a linear regression model with one parameter.
        test_X.append(np.mean(reviewEmbeddings))
    test_X = np.array(test_X)

    train_y = sentiment
    test_y = testSentiment
    
    
    return train_X, train_y, test_X, test_y,model


def unigramFeatures(Lyrics,sentiment):
    vectorizer = CountVectorizer()
    ind = np.random.permutation(len(Lyrics))#my random lyrics to train/test with
    Lyrics = np.array(Lyrics)
    sentiment = np.array(sentiment)#cast lyrics and sentiment to np arrays to be able to do nice tricks with them
    pTrain= int(len(Lyrics)*0.8)#80% of the lyrics will be used to train and 20 to test
    Tsongs=""
    TrSongs=""
    
    trainSet = ind[:pTrain]
    testSet= ind[pTrain:]#indicies for train and test sets
    for i in Lyrics[trainSet]:
        Tsongs+=i+"\n"
    for i in Lyrics[testSet]:
        TrSongs+=i+"\n"#append appropriate items to the songs
    f1 = open(os.getcwd()+"/TrainSongs.txt",mode="w")
    f1.write(Tsongs)
    f1.close()
    f2 = open(os.getcwd()+"/TestSongs.txt",mode="w")
    f2.write(TrSongs)
    f2.close()
    with open(os.getcwd()+"/TrainSongs.txt") as f1, open(os.getcwd()+"/TestSongs.txt") as f2:
        train_X_strings = f1.read().splitlines()
        test_X_strings = f2.read().splitlines()
        train_X = vectorizer.fit_transform(train_X_strings)
        test_X = vectorizer.transform(test_X_strings)

    train_y = sentiment[trainSet]
    test_y = sentiment[testSet]

    return train_X, train_y, test_X, test_y
def getSong(songName, Artist, glove):
    file = open(os.getcwd()+"/UserSong.txt","w")
    try:
        song = genius.search_song(songName, Artist)
        try:
            file.write(song.lyrics)#occasionally songs aren't compatable, not sure why as they should be
        except:
            print("Song could not be written continuing...")
            return [-1000]

    except:
        print("some exception occured")
        return [-1000]
    file.close()
    with open(os.getcwd()+"/UserSong.txt","r") as file:
        test_X_strings = file.read().splitlines()
    for i in range(len(test_X_strings)):
        test_X_strings[i] = re.findall(r'(\b[bcdfghj-np-tv-z]*[aeiou]+[bcdfghj-np-tv-z]*)\b', makeStr(test_X_strings[i]))
    test_X=[]
    for song in test_X_strings:
        reviewEmbeddings=[]
        for word in range(len(song)):
                #find next adj or verb to modify
            embedding = glove.get(song[word])
            if embedding is not None:
                reviewEmbeddings += list(embedding)
        #average all word embedding values into a single mean per review, making this a linear regression model with one parameter.
        if len(reviewEmbeddings) > 0:
            test_X.append(np.mean(reviewEmbeddings))


    test_X = np.array(test_X)
    return test_X
    
    
if __name__ == "__main__":

    
    TrLyrics,TrSentiment,TestLyrics, TestSentiment = readLyrics()
    #Splits the values of positive and Negative to  our binary classes -1 and 1
    for val in range(len(TrSentiment)):
        if TrSentiment[val] == "Negative":
            TrSentiment[val] = -1.0
        elif TrSentiment[val] == "Positive":
            TrSentiment[val] = 1
        else:
            TrSentiment[val] = -1
    for val in range(len(TestSentiment)):
        if TestSentiment[val] == "Negative":
            TestSentiment[val] = -1.0
        elif TestSentiment[val] == "Positive":
            TestSentiment[val] = 1
        else:
            TestSentiment[val] = -1
    # Linear regression model with bag of words as features.
    regressor = LinearRegression()
    train_X, train_y, test_X, test_y = unigramFeatures(TrLyrics,TrSentiment)    
    regressor.fit(train_X, train_y)

    # Evaluate using the test set.
    predict_y = regressor.predict(test_X)
    for pred in range(len(predict_y)):#creating a very bad classifier here
        if predict_y[pred] < 0:
            predict_y[pred] = -1
        elif predict_y[pred] > 0:
            predict_y[pred] = 1
        else:
            predict_y[pred]=0
                
    acc = np.sum(predict_y==test_y)/len(test_y)
    print(f"Acc(test) for unigram features = {acc:.3f}")  
    

    # Linear regression model with 1 dimensional word embedding per review as its sole feature.
    regressor = LinearRegression()
    train_X, train_y, test_X, test_y,glove = reviewsToEmbeddings1D(TrLyrics,TrSentiment, TestLyrics, TestSentiment)
    regressor.fit(train_X.reshape(-1, 1), train_y)
    originaly = []
    predict_y = regressor.predict(test_X.reshape(-1,1))
    for i in predict_y:
        originaly.append(i)
    for pred in range(len(predict_y)):#creating a very bad classifier here
        if predict_y[pred] < 0:
            predict_y[pred] = -1
        elif predict_y[pred] > 0:
            predict_y[pred] = 1
        else:
            predict_y[pred]=0
                
    acc = np.sum(predict_y==test_y)/len(test_y)
    print(f"Acc(test) for embedding features = {acc:.3f}") 
    #make our genius object to look up songs
    genius = genius.Genius('eOo8fgIk3HyopTIZ6NeAnj_M24xL3ms_N7PKaRXeYhTmxR0M3rFsr0yrszQ3a95g', skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"], remove_section_headers=True)
    
    while(True):#now present user the ability to determine their songs
        songName = input("What's the song called?\n>>> ")
        Artist = input("Who is the artist?\n>>> ")
        testSong = getSong(songName, Artist, glove)
        if testSong[0] == -1000:
            continue
        p = regressor.predict(testSong.reshape(-1,1))
        if p[0] > 0:
            print("That song is overall positive :)")
        elif p[0] < 0:
            print("That song is overall negative :(")
        