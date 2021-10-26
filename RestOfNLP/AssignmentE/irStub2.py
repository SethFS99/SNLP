# Nigel Ward, UTEP, October 2018
# Updated by Angel Garcia, UTEP, July 2020
# Speech and Language Processing
# Assignment E: Information Retrieval

# This is just a skeleton that needs to be fleshed out.
# It is not intended as an example of good Python style

import numpy as np
import sys,re

import nltk 
from nltk.corpus import stopwords
from nltk import PorterStemmer



def takeOutStopWords(unigram_words):
    tokens = []
    stopWords = set(stopwords.words('english'))
    # print(stopWords)
    stemmer = PorterStemmer()
    for word in (unigram_words):
        if word == "":
            continue
            
        if word.lower() in stopWords:
            continue
        tokens.append(word)
    for word in range(len(tokens)):
        tokens[word] = stemmer.stem(tokens[word])
    # Stemming text

    # Remove stop words
    # print(tokens)
    return tokens

# print(takeOutStopWords(test))


def parseAlternatingLinesFile(file):
    """ read a sequence of pairs of lines
    e.g. text of webpage(s), name/URL
    """
    sequenceA = []
    sequenceB = []

    with open(file, mode="r", encoding="utf-8") as f:
        for i,line in enumerate(f):
            if i % 2:
                sequenceB.append(line.strip())
            else:
                sequenceA.append(line.strip())

    return sequenceA, sequenceB

def generateCharTrigrams(text):
    """Generate Character Trigrams from Text"""
    for i in range(len(text)-3+1):
        yield text[i:i+3]
        
def generateWordUnigrams(text):
    ''' Generate the word unigrams from the text  '''
    #Step 1 parse the words out then pass them back
    words = re.split("[\|/|;|,|.|!|?"+ '|"|'+" |-|:|\n|\s|\t|'|&|(|)]\s*",text)#splits file by all spaces and punctuation plus some extra
    for w in words:
        if len(w) == 0:
            continue
        if '-' in w:
            nw = w.split('-')
            for x in nw:
                if len(x) ==0:
                    continue
                yield x
            continue
        yield w
        
def computeFeatures(text, trigramInventory):        
    """Computes the count of trigrams.
    Trigrams can catch some similarities
    (e.g. between  "social" and "societal" etc.)
    
    But really should be replaced with something better
    """
    counts = {}
    for trigram in generateWordUnigrams(text):
        # print(trigram)
        if trigram in trigramInventory:
            counts[trigram] +=1
        else:
            counts[trigram]=1
    return counts
   

def computeSimilarity(dict1, dict2):
    """Compute the similarity between 2 dictionaries of trigtrams

    Ad-hoc and inefficient.
    """
    
    keys_d1 = set(dict1.keys())#query

    keys_d2 = set(dict2.keys())#my dict for prof
    matches = []
    for i in keys_d1:
        for k in keys_d2:
            # print("if %s in %s"% (i,k))
            if i in k:
                matches.append(i)

    similarity = len(matches) / len(dict2)

    return similarity
def removeBlanks(words):
    newWords = []
    for i in range(len(words)):
        if words[i] != "" or '-' not in words[i]:
            newList = words[i].split(" ")
            newWords += newList
            continue
        if '-' in words[i]:
            newList = re.split("[-]\s",words[i])
            newWords+=newList
    return newWords
def recombineQuery(words):
    q =""
    for i in words:
        q = q + i + " "
    return q
def retrieve(queries, trigramInventory, archive):     
    """returns an array: for each query, the top 3 results found"""
    top3sets = []
    for query in queries:
        #print(f"query is {query}")
        #------------need to remove stopwords from query and stem them
        # words = re.split("[(|)|,|/]\s*",query)#split query into a list of words
        # words = removeBlanks(words)
        # words = takeOutStopWords(words)
        # query = recombineQuery(words)
        # print(query,"new query")
        q = computeFeatures(query, trigramInventory)
        #print(f"query features are \n{q}")
        similarities = [computeSimilarity(q, d) for d in archive] 
        #testing stuff################################################
        # x = np.argsort(similarities)[-2:]
        # print(x)
        # print("top similarities for q: " ,query)
        # for i in x:
        #     print(similarities[i])
        # print(similarities)#############################################
        top3indices = np.argsort(similarities)[-3:]
        #print(f"top three indices are {top3indices}")
        
        top3sets.append(top3indices)  
    return top3sets

def valueOfSuggestion(result, position, targets):
    weight = [1.0, .5, .25]
    if result in targets:
        return weight[max(position, targets.index(result))]
    else:
        return 0


def scoreResults(results, targets):   #-----------------------------
    merits = [valueOfSuggestion(results[i], i, targets) 
            for i in range(3)]
    return sum(merits)


def scoreAllResults(queries, results, targets, descriptor):   
    print()
    print(f"Scores for {descriptor}")
    scores = [(q, r, t, scoreResults(r, t)) 
            for q, r, t in zip(queries, results, targets)]
    for q, r, t, s in scores:
        print(f"for query: {q}")
        print(f"  results = \n{r}")
        print(f"  targets = \n{t}")
        print(f"  score = {s:.3f}")

    all_scores = [s for _,_,_,s in scores]
    print("num queries", len(all_scores))
    overallScore = np.mean(all_scores)
    print(f"All Scores:\n{all_scores}")
    print(f"Overall Score: {overallScore:.3f}")

    return overallScore

def pruneUniqueNgrams(ngrams):
    twoOrMore = {} 
    print("Before pruning: " +
            f"{len(ngrams)} ngrams across all documents")

    twoOrMore = {k:v for k,v in ngrams.items() if ngrams[k] > 1}

    print("After pruning: " +
            f"{len(twoOrMore)} ngrams across all documents")
    
    return twoOrMore

def findAllNgrams(contents):
    unigram_table = {}
    for token in contents:
        if token in unigram_table:
            unigram_table[token] += 1
        else:
            unigram_table[token] = 1
    return unigram_table

def targetNumbers(targets, nameInventory):
    """targets is a list of strings, each a sequence of names"""
    targetIDs = []
    for target in targets:
      threeNumbers = [] 
      for name in target.split():
          try:
              threeNumbers.append(nameInventory.index(name))
          except ValueError: 
              print("some name did not appear")
      targetIDs.append(threeNumbers)
    return targetIDs
          

if __name__ == "__main__":
    
    
    if len(sys.argv) != 3:
        print("Usage: python irStub.py " +
              "<document file>" +
              "<queries file>")
        sys.exit()

    print("......... irStub .........")
    
    contents, names =  parseAlternatingLinesFile(sys.argv[1]) 
    
    print(f"read in pages for {names}")
    
    trigramInventory = findAllNgrams(contents)
    archive =[computeFeatures(line, trigramInventory) for line in contents]

    print("[--------archive made---------------]")
    queries, targets = parseAlternatingLinesFile(sys.argv[2])
    targetIDs = targetNumbers(targets, names)
    results = retrieve(queries, trigramInventory, archive)
    modelName = "silly word unigram model"
    
    Overall = scoreAllResults(queries, results, targetIDs, 
            f"{modelName} on {sys.argv[1]}")
