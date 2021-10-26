# Speech and Language Processing
# Assignment B1: Introduction to Prediction
'''
this code was edited entirely by Seth Flores for the course Speech and Language Proccessing
Assignment B1
'''
from sklearn.linear_model import LinearRegression
import numpy as np
import sys, re


def reset_dict(letterCount):#resets all values in the dictionary back to 0 for the next word
    for key in letterCount:
        letterCount[key] = 0
        
def analyze_word(word, letterCount):#find the num of each letter in the word to assign values later
    word = word.lower()#ensures the word is lowercase
    specials = "áéíóúüñ"
    for l in word:
        try:
            letterCount[l]+=1
        except:
            continue
    #^find num letters in each word and store in a dictionary
    #now find the weights of each letter
    w = []
    for v in letterCount.values():
        w.append(v)#add the values to a list that i can then modify using numpy
    tempw = np.zeros((len(w),1),dtype = int)
    for i in range(len(w)):
        tempw[i][0]= w[i]
    weights = np.linalg.lstsq(tempw, np.arange(len(w)))
    print(weights[0], ": ", len(weights[0]))
def is_viet(word):
    if len(word) <=4:
        return True
    return False
def is_scottish(word):#
    wordL = re.split("mac|am|son",word)
    return len(wordL) >1

def is_korean(word):
    if is_chinese(word) and len(word) <4:
        return True
    return False

def is_russian(word):#
    wordL = re.split("ov|sky|ka|ba|ko",word)
    return len(wordL) >1

def is_irish(word):#
    wordL = re.split("mc|'|al|o'",word)
    return len(wordL) >1
    
def is_german(word):
    wordL = re.split("dt|erg|sch|nz",word)
    return len(wordL) >1

def is_polish(word):#
    wordL = re.split("grad",word)
    return len(wordL) >1

def is_port(word):
    wordL = re.split("ro|cruz", word)
    return len(wordL) >1

def is_french(word):#
    #honhonhon
    wordL = re.split("eux|air|du",word)
    return len(wordL) > 1
def is_greek(word):
    wordL = re.split("is|olus|os",word)
    if len(wordL) >1:
        return True
    else:
        return False
def is_dutch(word):#do this last
    if word.count("e") >2:
        return True
    
def is_chinese(word):
    wordL = re.split("oo|ao|in|an|ei|xi",word)
    if len(wordL) > 1:
        return True
    else:
        return False
def is_czech(word):
    wordL = re.split("ak|ek", word)
    if len(wordL) > 1:
        return True
    else:
        return False
    
def is_arabic(word):
    wordL = re.split("ail|im|na|afa|ar", word)
    if len(wordL) > 1:
        return True
    else:
        return False
    
def is_spanish(word):
    """Naive Spanish Surname Identification"""
#    word = word.lower()
#    keys = "áéíóúüñ"
    wordL = re.split("qu|rr|re|ra|es|ana",word)
    if len(wordL) > 1:
        return True
    
#    for letter in word:
#        if letter in keys:
#            return True
    return False


def is_italian(word):
    """Naive Italian Surname Identification"""
    wordL = re.split("ni|ri|ci|gg|ti",word)
    if len(wordL) > 1:
        return True
    return False

def is_japanese(word):#
    """Naive Japanese Surname Identification"""
    wordL = re.split("naka|tsu|kawa|ki|ku|ama|awa|ish|shi|wat|uji", word)
    if len(wordL) > 1:
        return True
    else:
        return False
def is_eng(word):
    wordL = re.split("tru|ell|ow|al|rd|ley|ton|ins",word)
    if len(wordL)>1:
        return True
    else: 
        return False
def check_nationality(word):
    """Naive Nationality Identification

    Returns "Unknown" for nationalities that are detected as 
    other than french, japanese, ductch, russian,spanish,korean,chinese,italian,vitenamese, english,greek,irish,scottish,czech, or arabic
    """
    word = word.lower()#don't worry about case
    if is_japanese(word):
        return "Japanese"
    if is_irish(word):
        return "Irish"
    if is_scottish(word):
        return "Scottish"
    if is_polish(word):
        return "Polish"
    if is_french(word):
        return "French"
    if is_russian(word):
        return "Russian"
    if is_korean(word):
        return "Korean"
    if is_port(word):
        return "Portugese"
    if is_italian(word):
        return "Italian"
    if is_greek(word):
        return "Greek"
    if is_czech(word):
        return "Czech"
    if is_eng(word):
        return "English"
    if  is_chinese(word):
        return "Chinese"
    if is_arabic(word):
        return "Arabic"
    if is_spanish(word):
        return "Spanish"
    if is_dutch(word):
        return "Dutch"
    if is_viet(word):
        return "Vietnamese"
    return "Unknown"

if __name__ == "__main__":
    letterCount = {'a':0,'b':0,'c':0,'d':0,'e':0,'f':0,'g':0,'h':0,'i':0,'j':0,'k':0,'l':0,'m':0,'n':0,'o':0,'p':0,'q':0,'r':0,'s':0,'t':0,'u':0,'v':0,'w':0,'x':0,'y':0,'z':0}#dictionary for our list of letters
#    if len(sys.argv) != 3:
#        print("Usage: python b1.py " +
#              "<input file> <output file>" )
#        sys.exit()
#    Nationality = ["Japenese", "English", "Russian", "Chinese", "Czech", "Greek", "Italian", "Portugese","Vietnamese","Spanish","Arabic","Dutch","French","Korean","Polish","Scottish","Irish"]
    with open("musician_surnames.csv", mode="r", encoding="utf-8") as input_file, \
          open("musician_out.csv", mode="w", encoding="utf-8") as output_file:
        for surname in input_file:
            surname = surname.strip()
#            print(surname.split(",")[0])
            output_file.write(surname)
            output_file.write(",")
            analyze_word(surname.split(",")[0], letterCount)
            reset_dict(letterCount)
            output_file.write(check_nationality(surname.split(",")[0]))
            output_file.write("\n")