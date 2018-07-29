from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, chi2
import numpy as np

iris = load_iris()
X, y = iris.data, iris.target
# Select 3 top features
selection = SelectKBest(chi2, k=3)
selection.fit_transform(X, y)
print(selection.scores_)
print(len(X))
print(len(y))







numarray = np.empty(shape=[0, 1])

for i in range(300):
    numarray = np.append(numarray, [[0]], axis=0)

for i in range(300):
    numarray = np.append(numarray, [[1]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[2]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[3]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[4]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[5]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[6]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[7]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[8]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[9]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[10]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[11]], axis=0)
for i in range(300):
    numarray = np.append(numarray, [[12]], axis=0)
