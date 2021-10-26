# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 23:30:06 2020

@author: sethf
"""
import sys,re
NumCorrect=0
JapenseR =0
Fp = 0
Tp = 0
Fn = 0
with open(sys.argv[1], mode="r", encoding="utf-8") as input_file, \
          open(sys.argv[2], mode="w", encoding="utf-8") as output_file:
#        x = len(input_file)
        for surname in input_file:
            nationality = surname.strip(",")
            nationality = re.split(",|\n",nationality)
            nationality.remove('')
#            print(nationality)
            if nationality[1].lower() == nationality[2].lower():
#                print(nationality[1], "  ",nationality[2])
                NumCorrect+=1
                if nationality[1].lower() == "japanese":
                    Tp+=1
#            print(nationality[1].lower(), " : ", nationality[2].lower())
            if nationality[1].lower() == "japanese" and nationality[2].lower() != nationality[1].lower():
#                    print("got a FN")
                    Fn+=1
            elif nationality[1].lower() != "japanese" and nationality [2].lower() == "japanese":
#                print("got a FP")
                Fp+=1
        print("Accuracy: ", NumCorrect/3003 )
        print("precision: ", Tp/(Tp+Fp))
        print("Recall: ", Tp/(Tp+Fn))