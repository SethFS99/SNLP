# Angel F. Garcia Contreras, UTEP, July 2020
# Speech and Language Processing
# Assignment C3: Introduction to Sequence Modeling

import itertools
import math
import string
import sys

def read_data(fname):
    """Read names from file (one name per line)"""
    data = []
    with open(fname, mode="r", encoding="utf-8") as input_file:
        data = [line.lower().strip() for line in input_file]
    return data

def generate_bigrams(word):
    """Generator for bigrams in word
    Includes starting (@) and ending (#) symbols
    """
    lower = "@" + word
    upper = word + "#"
    
    bigram_gen = map(lambda l,u: l+u, lower, upper)

    for bigram in bigram_gen:
        yield bigram

def compute_frequencies(data):
    """data is an iterable.
    Frequency counts for both Bigrams and Unigrams.
    """
    all_chars = "@" + string.ascii_lowercase + " #"
    bigram_freqs = {a+b:0 for (a,b) in itertools.product(
            all_chars[:-1], all_chars[1:])}
    unigram_freqs = {a:0 for a in all_chars}

    for name in data:
        for letter in "@" + name + "#":

            if letter in unigram_freqs:
                unigram_freqs[letter] += 1
            else: 
                unigram_freqs[letter] = 1
        for bigram in generate_bigrams(name):
            if bigram in bigram_freqs:
                bigram_freqs[bigram] += 1
            else:
                bigram_freqs[bigram] = 1

    return bigram_freqs, unigram_freqs

def compute_probabilities_add_k(bigram_freq, letter_freq, k=1.0):
    """Compute bigram conditional probabilities,
    using add-k smoothing.

    Basic formula:
        P(b | a) = C(ab) / C(a)
    With add-k:
        P(b | a) = (C(ab) + k) / (C(a) + k*N)
    """
    probs = {a+b:0 for (a,b) in itertools.product(
            string.ascii_lowercase + " @", 
            string.ascii_lowercase + " #")}

    for bigram in bigram_freq.keys():
        a = bigram[0]
        b = bigram[1]
        C_ab = bigram_freq[bigram]
        C_a = letter_freq[a]
        probs[a+b] = (C_ab + k)/(C_a  + k * len(letter_freq))

    return probs

def get_name_probabilities(model_probs, test_data):
    """Compute the probabilities for all names in test_data,
    based on the bigram-based model model_probs
    """
    # It is possible that test_data has unknown bigrams.
    # This sets a default value for unknown bigrams 
    min_p = min(model_probs.values())

    for name in test_data:
        p = 1
        for bigram in generate_bigrams(name):
            p *= model_probs.get(bigram, min_p)
        
        yield name, p

# Create auto-completed names for each name in test_data
def get_name_predictions(model_probs, test_data):
    """Auto-complete predictor for names.
    Find the next letters for each name in test_data,
    based on the higest-probability bigrams
    """
    # It is possible that test_data has unknown bigrams.
    # This sets a default value for unknown bigrams 
    min_p = min(model_probs.values())

    char_list = string.ascii_lowercase + " #"
    for name in test_data:
        p = 1
        for bigram in generate_bigrams(name):
            p *= model_probs[bigram] if bigram in model_probs else min_p
        
        new_name = name
        while new_name[-1] != "#":
            possible_bigrams = (new_name[-1] + n for n in char_list)
            
            max_prob = min_p
            best_bigram = ""

            # Generator of (bigram, prob) tuples
            all_bigrams = ((b, model_probs.get(b, min_p))
                    for b in possible_bigrams)

            (best_bigram, max_prob) = max(all_bigrams, 
                    key=lambda t: t[1]) # Compare by probability

            # Update name, probability
            p *= max_prob
            new_name += best_bigram[-1]
        
        yield name, new_name[:-1], p
    
def get_cross_entropy(p_probs, m_probs):
    """Compute cross-entropy between a predicted & actual models
    """
    return -sum((p*math.log2(m) for (p, m)
            in zip(p_probs.values(), m_probs.values())))/len(p_probs)

if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print("Usage: python c3.py " +
              "<train file>" )
        sys.exit()

    # Load files
    train_data = read_data(sys.argv[1])
    test_data = read_data("surnames-rus-dev.csv")
    m_bigram_freq, m_letter_freq = compute_frequencies(train_data)
    q_bigram_freq, q_letter_freq = compute_frequencies(test_data)
    # add-k: optimizing value of k based on minimizing cross-entropy
    curr_k = 0.5
    k_step = 0.25
    m_probs = compute_probabilities_add_k(
            m_bigram_freq, m_letter_freq, k=0.5)
    q_probs = compute_probabilities_add_k(q_bigram_freq,q_letter_freq, k=0.5)
    best_ce = get_cross_entropy(m_probs, q_probs)
    prev_ce = 1.0

    i = 0

    while prev_ce > best_ce and (prev_ce - best_ce) > 1.0e-4:
        if i > 20:
            break
        m_probs_top = compute_probabilities_add_k(
                m_bigram_freq, m_letter_freq, k=(curr_k + k_step))
        q_probs_top = compute_probabilities_add_k(q_bigram_freq,q_letter_freq, k=(curr_k+k_step))

        m_probs_bot = compute_probabilities_add_k(
                m_bigram_freq, m_letter_freq, k=(curr_k - k_step))
        q_probs_bot = compute_probabilities_add_k(q_bigram_freq,q_letter_freq, k=(curr_k - k_step))
        ce_top = get_cross_entropy(m_probs_top, q_probs_top)
        print(ce_top," cross entropy")
        ce_bot = get_cross_entropy(m_probs_bot, q_probs_bot)
        print(ce_top, " cross entropy_bottom")
        if ce_top < ce_bot and ce_top < best_ce:
            m_probs = m_probs_top
            curr_k = curr_k + k_step
            k_step = k_step/2
            prev_ce = best_ce
            best_ce = ce_top
        elif ce_bot < ce_top and ce_bot < best_ce:
            m_probs = m_probs_bot
            curr_k = curr_k - k_step
            k_step = k_step/2
            prev_ce = best_ce
            best_ce = ce_bot
        i += 1
    
    print(best_ce, "best entropy")
