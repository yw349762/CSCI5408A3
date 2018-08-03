import tweepy
import json
import csv

import pandas as pd
from sentiment import SentimentAnalysis

consumer_key = '2beMDMlo7GP3J1mJsCZu5CJzy'
consumer_secret = 'gSEL8keHPRwjkIgtD3PWmPnPSE9i4S5uUQhf4ITG4YINvyAPHl'
access_token = '1000023396900536320-BV4LqliYbXEbaYhGd8L0UrrNmtbYXb'
access_token_secret = 'g13Xf7ZQXBkmEiLukJfISRNz8bPHJXPPFY5XV4pHdBgn1'

# OAuth process, using the keys and tokens
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

# Creation of the actual interface, using authentication
api = tweepy.API(auth)

data = pd.read_csv('visitor/HalifaxHarbourwinter/2015HalifaxHarbourvistorwinter.csv', sep=';', error_bad_lines=False)

countNegative=0
countPositive=0
countNetural=0


df1 = data['text']

with open('visitor/HalifaxHarbourwinter/out2015HalifaxHarbourvistorwinter.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["text","Sentiment","Score"])

    for row in df1.iteritems():
        s = SentimentAnalysis(filename='SentiWordNet.txt', weighting='geometric')
        textt = row[1]
        if s.score(textt)==0.0:
                senti="netural"
                countNetural=countNetural+1

        if s.score(textt)<0.0:
                senti = "negative"
                countNegative=countNegative+1
        if s.score(textt) > 0.0:
                senti = "positive"
                countPositive=countPositive+1
        writer.writerow(
                [textt.encode('unicode-escape'),senti, s.score(textt)])
    writer.writerow([countNetural, countNegative, countPositive])

data = pd.read_csv('visitor/HalifaxHarbourwinter/2016HalifaxHarbourvistorwinter.csv', sep=';', error_bad_lines=False)

countNegative=0
countPositive=0
countNetural=0


df1 = data['text']

with open('visitor/HalifaxHarbourwinter/out2016HalifaxHarbourvistorwinter.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["text","Sentiment","Score"])

    for row in df1.iteritems():
        s = SentimentAnalysis(filename='SentiWordNet.txt', weighting='geometric')
        textt = row[1]
        if s.score(textt)==0.0:
                senti="netural"
                countNetural=countNetural+1

        if s.score(textt)<0.0:
                senti = "negative"
                countNegative=countNegative+1
        if s.score(textt) > 0.0:
                senti = "positive"
                countPositive=countPositive+1
        writer.writerow(
                [textt.encode('unicode-escape'),senti, s.score(textt)])
    writer.writerow([countNetural, countNegative, countPositive])

data = pd.read_csv('visitor/HalifaxHarbourwinter/2017HalifaxHarbourvistorwinter.csv', sep=';', error_bad_lines=False)

countNegative=0
countPositive=0
countNetural=0


df1 = data['text']

with open('visitor/HalifaxHarbourwinter/out2017HalifaxHarbourvistorwinter.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    writer.writerow(["text","Sentiment","Score"])

    for row in df1.iteritems():
        s = SentimentAnalysis(filename='SentiWordNet.txt', weighting='geometric')
        textt = row[1]
        if s.score(textt)==0.0:
                senti="netural"
                countNetural=countNetural+1

        if s.score(textt)<0.0:
                senti = "negative"
                countNegative=countNegative+1
        if s.score(textt) > 0.0:
                senti = "positive"
                countPositive=countPositive+1
        writer.writerow(
                [textt.encode('unicode-escape'),senti, s.score(textt)])
    writer.writerow([countNetural, countNegative, countPositive])



