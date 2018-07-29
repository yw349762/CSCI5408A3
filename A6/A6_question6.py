from sklearn import linear_model
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import svm, datasets

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

#
# npd = np.array(data.iloc[:, 0])
# npd1 = np.array(data.iloc[:, 1])

npd = np.array(data.iloc[:, 0])
npd1 = np.array(data.iloc[:, 1])


dnpd = np.array(data1.iloc[:, 0])
dnpd1 = np.array(data1.iloc[:, 1])
clf = LinearSVC(random_state=100)

#
Y = vectorizer.fit_transform(npd)
Y1 = vectorizer.fit_transform(dnpd)

# X_train, X_test, y_train, y_test = train_test_split(Y, npd1, test_size=0.3, random_state=100)
X_train=Y
y_train=npd1
X_test=Y1
y_test=dnpd1




clf.fit(X_train, y_train)

clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
# print(clf_gini.fit(X_train, y_train))


nclf = MultinomialNB()
nclf.fit(X_train, y_train)


lnclf = linear_model.LogisticRegression(C=1e5)
lnclf.fit(X_train, y_train)


classifier = svm.SVC(kernel='linear', C=0.01)
y_pred = classifier.fit(X_train, y_train).predict(X_test)


#y_true = y_train


y_true = y_pred


accuracy_score(y_true, clf.predict(X_test))
print(accuracy_score(y_true, clf.predict(X_test)))
print(accuracy_score(y_true, clf_gini.predict(X_test)))
print(accuracy_score(y_true, nclf.predict(X_test)))
print(accuracy_score(y_true, lnclf.predict(X_test)))

# print(clf_gini.predict(X_train))
# print(nclf.predict(X_train))
