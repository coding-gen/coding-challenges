#! /usr/bin/python3

"""
Author: Genevieve LaLonde

Perform sentiment analysis on a multiclass classification problem.
Classes: ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
Created as part of an interview coding challenge.

Statistical methods modified from: https://www.analyticsvidhya.com/blog/2021/06/rule-based-sentiment-analysis-in-python/

ML method modified from: https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment
@inproceedings{barbieri-etal-2020-tweeteval,
    title = "{T}weet{E}val: Unified Benchmark and Comparative Evaluation for Tweet Classification",
    author = "Barbieri, Francesco  and
      Camacho-Collados, Jose  and
      Espinosa Anke, Luis  and
      Neves, Leonardo",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2020",
    month = nov,
    year = "2020",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2020.findings-emnlp.148",
    doi = "10.18653/v1/2020.findings-emnlp.148",
    pages = "1644--1650"
}
"""


import database_connection as db
import argparse
import pandas as pd 
from os import path
import text_preprocess
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.stem import WordNetLemmatizer
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
import numpy as np
from scipy.special import softmax
from evaluator import evaluate
from datetime import datetime


def getArgs():
    parser = argparse.ArgumentParser(description='An evaluator for the sentiment analysis task.')
    parser.add_argument('-f',
        '--input-file',
        default = '',
        dest='in_file',
        type=str,
        help='Name of file to input instead of from database.')
    parser.add_argument('-m',
        '--method',
        default = 1,
        dest='method',
        type=int,
        help='Choose the method of the sentiment analyzer. Valid: 0-3')
    return parser.parse_args()


def getDataFromFile(in_file):
    df = pd.read_csv(in_file, sep=',', keep_default_na=False)
    return df   


def getDataFromDB():
    data = db.getDataDump()
    df = pd.DataFrame.from_dict(data)
    return df


def getPolarity(text):
    # In text blob polarity, 1 is more positive, -1 is more negative, and around 0 is neutral
    return TextBlob(text).sentiment.polarity


def lowThresholdAnalysis(score):
    if score < -0.34:
        return 'NEGATIVE'
    elif score > 0.34:
        return 'POSITIVE'
    else:
        return 'NEUTRAL'


def highThresholdAnalysis(score):
    if score < -0.5:
        return 'NEGATIVE'
    elif score > 0.5:
        return 'POSITIVE'
    else:
        return 'NEUTRAL'


def getVaderSentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    return scores['compound']


def preprocess(text):
    # Preprocess text (username and link placeholders)
    new_text = []
 
    for t in text.strip().split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def getRobertaScore(text, model, tokenizer, classifier):
    encoded_input = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
    classifier.save_pretrained(model)
    output = classifier(**encoded_input)

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    return scores


def robertaScoresAnalysis(scores):
    labels = ['NEGATIVE', 'NEUTRAL', 'POSITIVE']
    ranking = np.argsort(scores)
    ranking = ranking[::-1]
    return labels[ranking[0]]


if __name__ == "__main__":
    start_time = datetime.now()
    # Get the data
    args = getArgs()

    data = pd.DataFrame()
    if args.in_file != '':
        # pull data from file
        data = getDataFromFile(args.in_file)
    else:
        # pull data from db instead.
        data = getDataFromDB()

    # PreProcess the text
    concatenation = data['title'] + ' ' + data['entry']

    # Get Predictions with various methods
    if args.method == 1: 
        # Method: TextBlob, concatenating title and entry
        # PreProcess the text
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Preprocessing the data.')
        processed = concatenation.apply(text_preprocess.preprocessText)
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Finished preprocessing the data.')
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Calculating predictions with the TextBlob statistical method.')        
        polarity = processed.apply(getPolarity)
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Finished calculating predictions.')
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Analyzing the scores.')
        sentiment_output = polarity.apply(lowThresholdAnalysis)
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Finished analyzing the scores.')


    elif args.method == 2:
        # Vader: Valence Aware Dictionary and Sentiment Reasoner
        processed = concatenation.apply(text_preprocess.preprocessText)
        scores = processed.apply(getVaderSentiment)
        sentiment_output = scores.apply(highThresholdAnalysis)

        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Preprocessing the data.')
        processed = concatenation.apply(text_preprocess.preprocessText)
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Finished preprocessing the data.')
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Calculating predictions with the Vader statistical method.')  
        scores = processed.apply(getVaderSentiment)
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Finished calculating predictions.')
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Analyzing the scores.')
        sentiment_output = scores.apply(highThresholdAnalysis)
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Finished analyzing the scores.')        


    elif args.method == 3:
        # HuggingFace sentiment analysis pipeline with RoBERTa twitter model
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Preprocessing the data.')
        preprocessed = concatenation.apply(preprocess)
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Finished preprocessing the data.')
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Calculating predictions with the Roberta Twitter model.')

        # using autotokenizer pretrained on same model as used instead of my custom preprocessing
        # processed = concatenation.apply(text_preprocess.preprocessText)

        # Use tokenizer from base model, not task model.
        # RoBERTa's max token length is 512.
        tokenizer = AutoTokenizer.from_pretrained('cardiffnlp/twitter-roberta-base')

        # PyTorch
        # TODO: consider tuning classifier model hyperparams: hidden_states, attentions
        model='cardiffnlp/twitter-roberta-base-sentiment'
        task='sentiment'
        """
        # Valid tasks for this model:
        # emoji, emotion, hate, irony, offensive, sentiment
        # stance/abortion, stance/atheism, stance/climate, stance/feminist, stance/hillary
        """

        classifier = AutoModelForSequenceClassification.from_pretrained(model)

        scores_series = preprocessed.apply(getRobertaScore, args=(model, tokenizer, classifier))
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Finished calculating predictions.')
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Analyzing the scores.')
        sentiment_output = scores_series.apply(robertaScoresAnalysis)
        print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Finished analyzing the scores.')


    # Evaluate Original Predictions 
    print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Evaluating the accuracy metrics.')

    if args.method == 0 and args.in_file != '': 
        evaluate(data['annotated_sentiment'], data['sentiment_output'])
    else:
        evaluate(data['annotated_sentiment'], sentiment_output)

    end_time = datetime.now()
    print(f'{datetime.now().strftime("%H:%M:%S.%f")}: Total runtime: {end_time - start_time }')
