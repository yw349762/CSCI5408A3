import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

vectorizer = CountVectorizer()

data = pd.read_csv('/Users/yiweizhang/Desktop/CSCI5408/assignment6/A6/A6/train.txt', sep="\t", error_bad_lines=False)

newDF = pd.DataFrame()  # creates a new dataframe that's empty
newDF = newDF.append(data.loc[data.iloc[:, 1] == "bg"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "mk"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "bs"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "hr"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "sr"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "cz"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "sk"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "es-AR"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "es-ES"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "pt-BR"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "pt-PT"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "id"].head(300))
newDF = newDF.append(data.loc[data.iloc[:, 1] == "my"].head(300))

npd = np.array(data.iloc[:, 0])
npd1 = np.array(data.iloc[:, 1])


setfeature=0


Y = vectorizer.fit_transform(npd)

# feature extraction
test1 = SelectKBest(score_func=chi2, k=5)
fit1 = test1.fit(Y, npd1)
features1 = fit1.transform(Y)
clf1 = LinearSVC(random_state=100)
clf1.fit(features1, npd1)
classifier1 = svm.SVC(kernel='linear', C=0.01)
y_pred1= classifier1.fit(features1, npd1).predict(features1)


#y_true = y_train


y_true1 = y_pred1

print("accuracy when feature is 5 is \n")
print(accuracy_score(y_true1, clf1.predict(features1)))
print("\n")



# feature extraction
test4 = SelectKBest(score_func=chi2, k=7)
fit4 = test4.fit(Y, npd1)
features4 = fit4.transform(Y)
clf4 = LinearSVC(random_state=100)
clf4.fit(features4, npd1)
classifier4 = svm.SVC(kernel='linear', C=0.01)
y_pred4= classifier4.fit(features4, npd1).predict(features4)


#y_true = y_train


y_true4 = y_pred4


print("accuracy when feature is 7 is \n")
print(accuracy_score(y_true4, clf4.predict(features4)))
print("\n")


# feature extraction
test2 = SelectKBest(score_func=chi2, k=8)
fit2 = test2.fit(Y, npd1)
features2 = fit2.transform(Y)
clf2 = LinearSVC(random_state=100)
clf2.fit(features2, npd1)
classifier2 = svm.SVC(kernel='linear', C=0.01)
y_pred2= classifier2.fit(features2, npd1).predict(features2)


#y_true = y_train


y_true2 = y_pred2


print("accuracy when feature is 8 is \n")
print(accuracy_score(y_true2, clf2.predict(features2)))
print("\n")



# feature extraction
test3 = SelectKBest(score_func=chi2, k=9)
fit3 = test3.fit(Y, npd1)
features3 = fit3.transform(Y)
clf3 = LinearSVC(random_state=100)
clf3.fit(features3, npd1)
classifier3 = svm.SVC(kernel='linear', C=0.01)
y_pred3= classifier3.fit(features3, npd1).predict(features3)


#y_true = y_train


y_true3 = y_pred3


print("accuracy when feature is 9 is \n")
print(accuracy_score(y_true3, clf3.predict(features3)))
print("\n")