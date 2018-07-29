import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2

vectorizer = CountVectorizer()

data = pd.read_csv("train.txt", sep="\t", error_bad_lines=False)

newDF = pd.DataFrame() #creates a new dataframe that's empty
newDF=newDF.append(data.loc[data.iloc[:,1]=="bg"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="mk"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="bs"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="hr"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="sr"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="cz"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="sk"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="es-AR"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="es-ES"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="pt-BR"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="pt-PT"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="id"].head(300))
# newDF=newDF.append(data.loc[data.iloc[:,1]=="my"].head(300))
# newDF.to_csv('train300.txt', header=True, index=False, sep='\t', mode='a')

npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
plt.figure()
p1=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color="red")





newDF=newDF.append(data.loc[data.iloc[:,1]=="mk"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p2=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color="blue")


newDF=newDF.append(data.loc[data.iloc[:,1]=="bs"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p3=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color="grey")


newDF=newDF.append(data.loc[data.iloc[:,1]=="hr"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p4=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color="black")



newDF=newDF.append(data.loc[data.iloc[:,1]=="sr"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p5=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color="green")



newDF=newDF.append(data.loc[data.iloc[:,1]=="cz"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p6=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color="yellow")



newDF=newDF.append(data.loc[data.iloc[:,1]=="sk"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p7=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color=[0,0.5,0.5])



newDF=newDF.append(data.loc[data.iloc[:,1]=="es-AR"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p8=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color=[0,0.5,0.9])

newDF=newDF.append(data.loc[data.iloc[:,1]=="es-ES"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p9=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color=[0.9,0.5,0.9])


newDF=newDF.append(data.loc[data.iloc[:,1]=="pt-BR"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p10=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color=[0.9,0.5,0.9])


newDF=newDF.append(data.loc[data.iloc[:,1]=="pt-PT"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p11=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color=[0.9,0.1,0.9])


newDF=newDF.append(data.loc[data.iloc[:,1]=="id"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p12=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color=[0.9,0.3,0.9])


newDF=newDF.append(data.loc[data.iloc[:,1]=="my"].head(300))
npd=np.array(newDF.iloc[:,0])
Y=vectorizer.fit_transform(npd).toarray()
pca = PCA(n_components=2)
pca.fit(Y)
X=pca.fit_transform(Y)
p13=plt.scatter(X[:, 0], X[:, 1], alpha=0.2,color=[0.9,0.7,0.9])



plt.legend((p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13),("bg","mk","bs","hr","sr","cz","sk","es-AR","es-ES","pt-BR","pt-PT","id","my"))
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Question3.png')

