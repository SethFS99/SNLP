# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:44:43 2020

@author: sethf
"""
import os,re
def readLyrics(checkRaw=False):#in case you want to debug at all you can get the full raw list
    Lyrics =[]
    TLyrics = []
    sentiment = []
    Tsentiment = []
    file = open(os.getcwd()+"/Lyrics.txt", "r")#open file to read
    file2 = open(os.getcwd()+"/TestLyrics.txt","r")
    
    rawFile = file.read().replace('\n',' ')#get all the data at once
    testFile = file2.read().replace("\n", " ")
    file2.close()
    file.close()
    rawFile = re.split("[<<|>>]", rawFile)
    testFile = re.split("[<<|>>]", testFile)
    newRaw =[]
    newTest = []
    for i in range(len(rawFile)):
        if len(rawFile[i])>1:
            newRaw.append(rawFile[i])
    for i in range(len(testFile)):
        if len(testFile[i]) > 1:
            newTest.append(testFile[i])
    #^^Filter out the spaces from the newly split rawFile
    for i in range(len(newRaw)):
        if i%2==0:#even adresses will have lyrics the following will have it's sentiment
            Lyrics.append(newRaw[i])
        else:
            if(testFile[i] == "Sentimental"):
                sentiment.append("Negative")
                continue
            sentiment.append(newRaw[i])
    # if(checkRaw):#for debugging early on
    #     return Lyrics,sentiment,newRaw
    for i in range(len(newTest)):
        
        if i%2==0:

            TLyrics.append(newTest[i])
        else:

            Tsentiment.append(newTest[i])

    
    return Lyrics, sentiment, TLyrics,Tsentiment#lyrics and sentiment are mapped 1-1
if __name__ == "__main__":
    Lyrics, sentiment,TLyrics,Tsentiment = readLyrics()