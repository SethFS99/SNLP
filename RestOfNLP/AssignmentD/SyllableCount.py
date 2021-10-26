# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 12:04:26 2020

@author: sethf
"""
import sys, re
def trySplit(word):
    vowels = "[a|e|i|o|u]"
    word = re.split(vowels,word)
    if ('' in word):
        cEmpty = 0
        for x in word: 
            if x == '':
                cEmpty +=1
        for x in range(cEmpty):
            word.remove('')
    if len(word) <= 2:
        # print(word)
        return 1
    if len(word) == 1:
        word = word[0]
    return 0

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python WordCount.py " + "<input file> <output file>" )#error message
        sys.exit()
    NOS = 0
    with open(sys.argv[1], mode="r", encoding="utf-8") as input_file, \
            open(sys.argv[2], mode="w", encoding="utf-8") as output_file:
                file = input_file.read()#reads the entire file
                file = re.split("[;|,|.|!|?"+ '|"|'+" |-|:|(|)|\n|\s|\t|']",file)#splits file by all spaces and punctuation plus some extra
                vowels = "aeiou"
                for word in file:
                    if len(word) == 0:
                        continue#skips empty words
                    if "-" in word:#some words with the - character don't parse correctly with the above Regular Expression so they are parsed again here
                        newWords = word.split("-")
                        for x in newWords:#parsing out empty characters
                            if len(x) ==0:
                                continue
                            newX = x.lower()
                            #this is inefficient but works right now
                            NOS += trySplit(newX)

                        continue# if worth with '-' character found and entered continue to next word
                    newWord = word.lower()#converts word to lowercase
                    NOS += trySplit(newWord)
                output_file.write("Num Words: %d" % NOS)#writes the word, number of appearances and starts new line
                
                    