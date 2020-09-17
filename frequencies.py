#!/usr/bin/env python3

# frequencies.py
# Calculate bigram frequencies for surnames
# September 2020
# Seth Flores and Jake Lasley

from string import ascii_lowercase
import sys
import itertools # for itertools.product()


def generate_bigrams(word):
    lower = word[:-1]
    upper = word[1:]
    bigram_gen = map(lambda l,u: l+u, lower, upper)
    for bigram in bigram_gen:
        yield bigram


def compute_frequencies(data, language):
    """
    data: iterable
    language: string; first letter capitalized
    Return frequency counts for both bigrams
    and unigrams, on a specific language
    """
    bigram_freqs = {a+b:0 for (a,b) in itertools.product(
        ascii_lowercase, ascii_lowercase)}
    letter_freqs = { a:0 for a in ascii_lowercase }

    filtered = filter(lambda x: x[1] == language, data)

    for (name, _) in filtered:
        for letter in name.lower():
            letter_freqs[letter] += 1
        for bigram in generate_bigrams(name.lower()):
            bigram_freqs[bigram] += 1

    return bigram_freqs, letter_freqs

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python frequencies.py " +
              "<input file>  <Language>" )
        sys.exit()

    target = sys.argv[2]

    with open(sys.argv[1], mode="r", encoding="utf-8") as input_file:
        # Create 2d list of surname unigram vectors
        # and 2d list of classifications for those vectors
        data = []
        for line in input_file:
            temp = line.strip().split(",")
            data.append(temp)
        b_freqs, u_freqs = compute_frequencies(data, target)
        for bigram in b_freqs:
            print('{}: {}'.format(bigram, b_freqs[bigram]))
        print()
        for letter in u_freqs:
            print('{}: {}'.format(letter, u_freqs[letter]))
        print()
        
