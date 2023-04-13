#! /usr/bin/python3

"""
Author: Genevieve LaLonde

A text preprocessor.
Created as part of an interview coding challenge.
"""

import nltk
# nltk.download('punkt') # already downloaded
from nltk.tokenize import word_tokenize
from nltk import pos_tag
# nltk.download('stopwords') # already downloaded
from nltk.corpus import stopwords
# nltk.download('wordnet') # already downloaded
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def tokenize(text):
    # Tokenization
    return word_tokenize(text)


def posTagging(tokens):
    # Part of Speech tagging
    # The WordNet lemmatizer requires POS tags.
    return pos_tag(tokens)


def removeStopWords(tagged):
    # English stop word removal
    return [(word, pos) for (word, pos) in tagged if word.lower() not in stopwords.words('english')]


def lemmatize(nonstop):
    # Lemmatization
    # The WordNet lemmatizer only accepts single char POS so we'll have to convert. 
    # Only handle these 4, or don't lemmatize (keep original).
    pos_dict = {'J':wordnet.ADJ, 'V':wordnet.VERB, 'N':wordnet.NOUN, 'R':wordnet.ADV}
    wordnet_lemmatizer = WordNetLemmatizer()

    lemma = []
    for element, tag in nonstop:
        tag = pos_dict.get(tag[0])
        if not tag:
            lemma.append(element)
        else:
            lemma.append(wordnet_lemmatizer.lemmatize(element, tag))

    # Convert back to whitespace separated sequence
    return (" ").join(word for word in lemma)


def preprocessText(text):
    # TODO: Also carefully clean out punctuation.

    # These are broken out to functions to allow their usage as a library if desired.
    tokens = tokenize(text)
    tagged = posTagging(tokens)
    nonstop = removeStopWords(tagged)
    sequence = lemmatize(nonstop)

    return sequence
