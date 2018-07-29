import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA

vectorizer = CountVectorizer()

data = pd.read_csv('/Users/yiweizhang/Desktop/CSCI5408/assignment6/A6/A6/train.txt', sep="\t", error_bad_lines=False)

# newDF = pd.DataFrame()  # creates a new dataframe that's empty
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "bg"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "mk"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "bs"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "hr"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "sr"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "cz"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "sk"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "es-AR"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "es-ES"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "pt-BR"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "pt-PT"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "id"].head(300))
# newDF = newDF.append(data.loc[data.iloc[:, 1] == "my"].head(300))

npd = np.array(data.iloc[:, 0])
npd1 = np.array(data.iloc[:, 1])

Y = vectorizer.fit_transform(npd)

# feature extraction
test = SelectKBest(score_func=chi2, k=5)
fit = test.fit(Y, npd1)
features = fit.transform(Y)
# summarize selected features
#print(features[3:5, :])
#print(features[3:5, :])


print(features.shape[0])
print(features)
