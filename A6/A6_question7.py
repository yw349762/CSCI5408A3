from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

vectorizer = CountVectorizer()

data = pd.read_csv('/Users/yiweizhang/Desktop/CSCI5408/assignment6/A6/A6/train.txt', sep="\t", error_bad_lines=False)
data1 = pd.read_csv('/Users/yiweizhang/Desktop/CSCI5408/assignment6/assignemnt6/test-gold.txt', sep="\t", error_bad_lines=False)

newDF = pd.DataFrame()  # creates a new dataframe that's empty
newDF = newDF.append(data.loc[data.iloc[:, 1] == "bg"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "mk"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "bs"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "hr"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "sr"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "cz"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "sk"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "es-AR"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "es-ES"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "pt-BR"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "pt-PT"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "id"].head(3000))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "my"].head(3000))

npd = np.array(data.iloc[:, 0])
npd1 = np.array(data.iloc[:, 1])
dnpd = np.array(data1.iloc[:, 0])
dnpd1 = np.array(data1.iloc[:, 1])

pipeline1 = Pipeline([
    ('vect', CountVectorizer()),
    ('feature_selection',SelectKBest(score_func=chi2, k=9)),
    ('clf', LinearSVC()),
])
pipeline2 = Pipeline([
    ('vect', CountVectorizer()),
    ('feature_selection',SelectKBest(score_func=chi2, k=9)),
    ('clf', linear_model.LogisticRegression()),
])

pipeline3 = Pipeline([
    ('vect', CountVectorizer()),
    ('feature_selection',SelectKBest(score_func=chi2, k=9)),
    ('clf', DecisionTreeClassifier()),
])

pipeline4 = Pipeline([
    ('vect', CountVectorizer()),
    ('feature_selection',SelectKBest(score_func=chi2, k=9)),
    ('clf', MultinomialNB()),
])


#X_train, X_test, y_train, y_test = train_test_split(npd, npd1, test_size=0.3, random_state=100)
X_train=npd
y_train=npd1
X_test=dnpd
y_test=dnpd1

f= open("accuracy.txt","w+")

pipeline1.fit(X_train, y_train)
y1 = pipeline1.predict(X_test)
print(" LINEAR SVM"+" "+classification_report(y1, y_test))
f.write("LINEAR SVM\n")
f.write(classification_report(y1, y_test)+"\n")

pipeline2.fit(X_train, y_train)
y2 = pipeline2.predict(X_test)
print(" LOGISTIC REGRESSION"+" "+classification_report(y2, y_test))
f.write("LOGISTIC REGRESSION\n")
f.write(classification_report(y2, y_test)+"\n")

pipeline3.fit(X_train, y_train)
y3 = pipeline3.predict(X_test)
print(" DECISION TREES"+" "+classification_report(y3, y_test))
f.write(" DECISION TREES\n")
f.write(classification_report(y3, y_test)+"\n")

pipeline4.fit(X_train, y_train)
y4 = pipeline4.predict(X_test)
print("  NAÏVE BAYES"+" "+classification_report(y4, y_test))
f.write("NAÏVE BAYES\n")
f.write(classification_report(y4, y_test)+"\n")