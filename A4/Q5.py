import matplotlib.pyplot as plt
import matplotlib
import squarify  # pip install squarify (algorithm for treemap)
from mpl_toolkits.mplot3d import Axes3D
from pandas import read_csv
import numpy as np

data = read_csv('QUESTION5.csv')
fig = plt.figure(figsize=(12, 10))
ax = Axes3D(fig)
X=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17]

Y=[47,1,1,67,1,1,2,2,187,5,2,20,1,4,1,2,1,2]
Z=[200114,10000,710000,372612,100000,100,20000,125800,243260,310600,400000,72650,60000,188775,380000,13000,80000,382500]
my_xticks = data['COMMUNITY']
plt.xticks(X, my_xticks)
plt.xlabel('COMMUNITY')
plt.ylabel('value')


plt.title("Summary of community")
ax.scatter(X, Y, Z)

fig1 = plt.gcf()
fig1.savefig('question5.png',dpi=48)
plt.show()