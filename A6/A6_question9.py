from __future__ import print_function

import numpy as np
import pandas as pd
from sklearn import linear_model, svm
import matplotlib.pyplot as plt
import itertools

from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier

vectorizer = CountVectorizer()

data = pd.read_csv('/Users/yiweizhang/Desktop/CSCI5408/assignment6/A6/A6/train.txt', sep="\t", error_bad_lines=False)
data1 = pd.read_csv('/Users/yiweizhang/Desktop/CSCI5408/assignment6/assignemnt6/test-gold.txt', sep="\t", error_bad_lines=False)

newDF = pd.DataFrame()  # creates a new dataframe that's empty
newDF = newDF.append(data.loc[data.iloc[:, 1] == "bg"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "mk"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "bs"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "hr"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "sr"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "cz"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "sk"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "es-AR"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "es-ES"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "pt-BR"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "pt-PT"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "id"].head(30))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "my"].head(30))

npd = np.array(data.iloc[:, 0])
npd1 = np.array(data.iloc[:, 1])
dnpd = np.array(data1.iloc[:, 0])
dnpd1 = np.array(data1.iloc[:, 1])
Y = vectorizer.fit_transform(npd)
Y1 = vectorizer.fit_transform(dnpd)
#X_train, X_test, y_train, y_test = train_test_split(npd, npd1, test_size=0.3, random_state=100)
X_train=npd
y_train=npd1
X_test=dnpd
y_test=dnpd1
# classifier = svm.SVC(kernel='linear', C=0.01)
# y_pred = classifier.fit(X_train, y_train).predict(X_test)
class_names =["bg","mk","bs","hr","sr","sk","es-AR","es-ES","pt-BR","pt-PT","id","my","cz"]

pipeline1 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LinearSVC()),
])
pipeline2 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', linear_model.LogisticRegression()),
])

pipeline3 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', DecisionTreeClassifier()),
])

pipeline4 = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB()),
])


pipeline1.fit(X_train, y_train)
y1 = pipeline1.predict(X_test)
print(" LINEAR SVM"+" "+classification_report(y1, y_test))

pipeline2.fit(X_train, y_train)
y2 = pipeline2.predict(X_test)
print(" LOGISTIC REGRESSION"+" "+classification_report(y2, y_test))

pipeline3.fit(X_train, y_train)
y3 = pipeline3.predict(X_test)
print(" DECISION TREES"+" "+classification_report(y3, y_test))

pipeline4.fit(X_train, y_train)
y4 = pipeline4.predict(X_test)
print("  NAÏVE BAYES"+" "+classification_report(y4, y_test))



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')




# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')
cnf_matrix1 = confusion_matrix(y_test, y1)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix1, classes=class_names, normalize=True,
                      title='LINEAR SVM')

fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('LINEAR SVM')


cnf_matrix2 = confusion_matrix(y_test, y2)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix2, classes=class_names, normalize=True,
                      title=' LOGISTIC REGRESSION')

fig2 = plt.gcf()
plt.show()
plt.draw()
fig2.savefig(' LOGISTIC REGRESSION')


cnf_matrix3 = confusion_matrix(y_test, y3)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix3, classes=class_names, normalize=True,
                      title='DECISION TREES')

fig3 = plt.gcf()
plt.show()
plt.draw()
fig3.savefig('DECISION TREES')


cnf_matrix4 = confusion_matrix(y_test, y4)
np.set_printoptions(precision=2)
plt.figure()
plot_confusion_matrix(cnf_matrix4, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

fig4 = plt.gcf()
plt.show()
plt.draw()
fig4.savefig('NAÏVE BAYES')

