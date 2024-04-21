import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def tokenize(sentence):
    """
    Tokenize a sentence into an array of words/tokens.
    """
    return nltk.word_tokenize(sentence)

def stem(word):
    """
    Stem a word to find its root form.
    """
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, words):
    """
    Create a bag of words array:
    - For each known word that exists in the sentence, set the corresponding index in the bag of words array to 1,
    - Otherwise, set it to 0.
    """
    # Stem each word in the tokenized sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    # Set corresponding index to 1 for each known word in the sentence
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1
    return bag
