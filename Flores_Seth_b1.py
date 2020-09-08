# Speech and Language Processing
# Assignment B1: Introduction to Prediction
'''
this code was edited entirely by Seth Flores for the course Speech and Language Proccessing
Assignment B1
'''
import sys, re

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
    if len(sys.argv) != 3:
        print("Usage: python b1.py " +
              "<input file> <output file>" )
        sys.exit()
#    Nationality = ["Japenese", "English", "Russian", "Chinese", "Czech", "Greek", "Italian", "Portugese","Vietnamese","Spanish","Arabic","Dutch","French","Korean","Polish","Scottish","Irish"]
    with open(sys.argv[1], mode="r", encoding="utf-8") as input_file, \
          open(sys.argv[2], mode="w", encoding="utf-8") as output_file:
        for surname in input_file:
            surname = surname.strip()
#            print(surname.split(",")[0])
            output_file.write(surname)
            output_file.write(",")
            output_file.write(check_nationality(surname.split(",")[0]))
            output_file.write("\n")