from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_absolute_error
import numpy as np
import re, spacy

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

#Average embeddings into one average word embedding vector per review
def reviewsToEmbeddingsAVG(sch_subj, sch_rating, ren_subj, ren_rating, rho_subj, rho_rating):
    model = loadGloveModel()
    # Read Schwartz and Berardinelli subjects to use as train data.
    with open(sch_subj) as f1, open(ber_subj) as f2:
        train_X_strings = f1.read().splitlines() + f2.read().splitlines()
        for i in range(len(train_X_strings)):
            #tokenize the reviews
            train_X_strings[i] = re.findall(r'(\b[bcdfghj-np-tv-z]*[aeiou]+[bcdfghj-np-tv-z]*)\b', train_X_strings[i])
    #Train data will be a matrix of samples, 50, rows for number of samples and columns for word embedding size.
    train_X = np.zeros((len(train_X_strings), 50))
    i = 0
    for review in train_X_strings:
        mean = np.zeros(50)
        count = 0
        for word in review:
            if model.get(word) is not None:
                mean = mean + model[word]
                count += 1
        #averaged all word embedding vectors in review to one average word embedding vector for dimensionality reduction.
        mean = (1/count) * mean
        train_X[i] = mean
        i += 1

    # Read Rhodes subjects and use to create test data
    with open(rho_subj) as f:
        test_X_strings = f.read().splitlines()
        for i in range(len(test_X_strings)):
            #tokenize the reviews
            test_X_strings[i] = re.findall(r'(\b[bcdfghj-np-tv-z]*[aeiou]+[bcdfghj-np-tv-z]*)\b', test_X_strings[i])
    #test data will be a matrix of samples, 50, rows for number of samples and columns for word embedding size.
    test_X = np.zeros((len(test_X_strings), 50))
    i = 0
    for review in test_X_strings:
        mean = np.zeros(50)
        count = 0
        for word in review:
            if model.get(word) is not None:
                mean = mean + model[word]
                count += 1
        #averaged all word embedding vectors in review to one average word embedding vector for dimensionality reduction.
        mean = (1/count) * mean
        test_X[i] = mean
        i += 1

    # Read Schwartz and Berardinelli ratings directly into train_y.
    train_y = np.concatenate((
        np.fromfile(sch_rating, sep="\n"),
        np.fromfile(ber_rating, sep="\n")
    ))

    # Read Rhodes ratings directly into test_y.
    test_y = np.fromfile(rho_rating, sep="\n")
    
    return train_X, train_y, test_X, test_y
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
        
def reviewsToEmbeddings1D(sch_subj, sch_rating, ren_subj, ren_rating, rho_subj, rho_rating):
    model = loadGloveModel()
    nlp = spacy.load('en_core_web_sm')
    
    # Read Schwartz and Berardinelli subjects to use as train data.
    with open(sch_subj) as f1, open(ber_subj) as f2:
        train_X_strings = f1.read().splitlines() + f2.read().splitlines()
        for i in range(len(train_X_strings)):
            #tokenize the reviews
            train_X_strings[i] = re.findall(r'(\b[bcdfghj-np-tv-z]*[aeiou]+[bcdfghj-np-tv-z]*)\b', train_X_strings[i])
    train_X = []
    for review in train_X_strings:
        mod_next_JJ = False
        review_txt = makeStr(review)
        current_doc = nlp(review_txt)
        # print(len(review),"<-review|", len(current_doc), "<- doc")
        # print(review)
        reviewEmbeddings = []
        for word in range(len(review)):#words able to do under this form of tokenizing: not, never, doesn't, isn't, shouldn't, don't, aren't{adding more words will sometimes result in worse results}
            if review[word] == "not" or review[word] == "never"or review[word] == "doesn"or review[word]=="isn" or review[word]=="shouldn" or review[word]=="don" or review[word]=="aren":#if the word not has been seen, remember to modify the next adjective/adverb we see
                mod_next_JJ=True
                #find next adj or verb to modify
            embedding = model.get(review[word])
            if embedding is not None:
                if(mod_next_JJ and (current_doc[word].tag_ == "JJ" or current_doc[word].tag_ == "RB")):
                    mod_next_JJ=False
                    reviewEmbeddings+= reverseEmbed(embedding)#invert the value of the embeddings to make them the reverse of what they would be normally
                else:reviewEmbeddings += list(embedding)
        #average all word embedding values into a single mean per review, making this a linear regression model with one parameter. y = mx + b
        

        train_X.append(np.mean(reviewEmbeddings))
    train_X = np.array(train_X)

    # Read Rhodes subjects and use to create test data
    with open(rho_subj) as f:
        test_X_strings = f.read().splitlines()
        for i in range(len(test_X_strings)):
            #tokenize the reviews
            test_X_strings[i] = re.findall(r'(\b[bcdfghj-np-tv-z]*[aeiou]+[bcdfghj-np-tv-z]*)\b', test_X_strings[i])
    test_X = []
    for review in test_X_strings:
        mod_next_JJ = False
        review_txt = makeStr(review)
        current_doc = nlp(review_txt)
        reviewEmbeddings = []
        for word in range(len(review)):
            if review[word] == "not" :#remember to modify the next adverb/adjective
                mod_next_JJ=True
            embedding = model.get(review[word])
            if embedding is not None:
                if(mod_next_JJ and (current_doc[word].tag_ == "JJ" or current_doc[word].tag_ == "RB")):
                    mod_next_JJ=False
                    reviewEmbeddings+= reverseEmbed(embedding)#invert the values of the next adverb/adject after the word not
                
                else:reviewEmbeddings += list(embedding)
        #average all word embedding values into a single mean per review, making this a linear regression model with one parameter. y = mx + b
        test_X.append(np.mean(reviewEmbeddings))
    test_X = np.array(test_X)

    # Read Schwartz and Berardinelli ratings directly into train_y.
    train_y = np.concatenate((
        np.fromfile(sch_rating, sep="\n"),
        np.fromfile(ber_rating, sep="\n")
    ))

    # Read Rhodes ratings directly into test_y.
    test_y = np.fromfile(rho_rating, sep="\n")
    
    return train_X, train_y, test_X, test_y

def unigramFeatures(sch_subj, sch_rating, ren_subj, ren_rating, rho_subj, rho_rating):
    vectorizer = CountVectorizer()
    # Read Schwartz and Berardinelli subjects and use their vocabulary to 
    # transform them to a document-term matrix, train_X.
    with open(sch_subj) as f1, open(ber_subj) as f2:
        train_X_strings = f1.read().splitlines() + f2.read().splitlines()
        train_X = vectorizer.fit_transform(train_X_strings)

    # Read Rhodes subjects and use transform them to a document-term matrix,
    # test_X, using the same vocabulary as train_X.
    with open(rho_subj) as f:
        test_X_strings = f.read().splitlines()
        test_X = vectorizer.transform(test_X_strings)

    # Read Schwartz and Berardinelli ratings directly into train_y.
    train_y = np.concatenate((
        np.fromfile(sch_rating, sep="\n"),
        np.fromfile(ber_rating, sep="\n")
    ))


    # Read Rhodes ratings directly into test_y.
    test_y = np.fromfile(rho_rating, sep="\n")
    
    return train_X, train_y, test_X, test_y

def tfidfFeatures(sch_subj, sch_rating, ren_subj, ren_rating, rho_subj, rho_rating):
    vectorizer = TfidfVectorizer()
    # Read Schwartz and Berardinelli subjects and use their vocabulary to 
    # transform them to a document-term matrix, train_X.
    with open(sch_subj) as f1, open(ber_subj) as f2:
        train_X_strings = f1.read().splitlines() + f2.read().splitlines()
        train_X = vectorizer.fit_transform(train_X_strings)

    # Read Rhodes subjects and use transform them to a document-term matrix,
    # test_X, using the same vocabulary as train_X.
    with open(rho_subj) as f:
        test_X_strings = f.read().splitlines()
        test_X = vectorizer.transform(test_X_strings)

    # Read Schwartz and Berardinelli ratings directly into train_y.
    train_y = np.concatenate((
        np.fromfile(sch_rating, sep="\n"),
        np.fromfile(ber_rating, sep="\n")
    ))

    # Read Rhodes ratings directly into test_y.
    test_y = np.fromfile(rho_rating, sep="\n")
    
    return train_X, train_y, test_X, test_y

if __name__ == "__main__":
    sch_rating = "scaledata/Dennis+Schwartz/rating.Dennis+Schwartz"
    ber_rating = "scaledata/James+Berardinelli/rating.James+Berardinelli"
    ren_rating = "scaledata/Scott+Renshaw/rating.Scott+Renshaw"
    rho_rating = "scaledata/Steve+Rhodes/rating.Steve+Rhodes"
    sch_subj = "scaledata/Dennis+Schwartz/subj.Dennis+Schwartz"
    ber_subj = "scaledata/James+Berardinelli/subj.James+Berardinelli"
    ren_subj = "scaledata/Scott+Renshaw/subj.Scott+Renshaw"
    rho_subj = "scaledata/Steve+Rhodes/subj.Steve+Rhodes"
    
    # Linear regression model with bag of words as features.
    regressor = LinearRegression()
    train_X, train_y, test_X, test_y = unigramFeatures(sch_subj, sch_rating, ren_subj, ren_rating, rho_subj, rho_rating)
    regressor.fit(train_X, train_y)

    # Evaluate using the test set.
    predict_y = regressor.predict(test_X)
    mae = mean_absolute_error(test_y, predict_y)
    print(f"MAE(test) for unigram features = {mae:.3f}")  
    
    # Linear regression model with tf-idf vectors as features.
    regressor = LinearRegression()
    train_X, train_y, test_X, test_y = tfidfFeatures(sch_subj, sch_rating, ren_subj, ren_rating, rho_subj, rho_rating)
    regressor.fit(train_X, train_y)
    
    # Evaluate using the test set.
    predict_y = regressor.predict(test_X)
    mae = mean_absolute_error(test_y, predict_y)
    print(f"MAE(test) for tfidf features = {mae:.3f}")  
    
    # Linear regression model with 1 dimensional word embedding per review as its sole feature.
    regressor = LinearRegression()
    train_X, train_y, test_X, test_y = reviewsToEmbeddings1D(sch_subj, sch_rating, ren_subj, ren_rating, rho_subj, rho_rating)
    regressor.fit(train_X.reshape(-1, 1), train_y)

    # Evaluate using the test set.
    predict_y = regressor.predict(test_X.reshape(-1, 1))
    mae = mean_absolute_error(test_y, predict_y)
    print(f"MAE(test) for 1 word embedding as sole feature = {mae:.3f}")  
    
    # Linear regression model with averaged word embeddings vectors as features.
    regressor = LinearRegression()
    train_X, train_y, test_X, test_y = reviewsToEmbeddingsAVG(sch_subj, sch_rating, ren_subj, ren_rating, rho_subj, rho_rating)
    regressor.fit(train_X, train_y)

    # Evaluate using the test set.
    predict_y = regressor.predict(test_X)
    mae = mean_absolute_error(test_y, predict_y)
    print(f"MAE(test) for averaged word embeddings features ={mae:.3f}")  
    
    
    
