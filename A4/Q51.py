import matplotlib.pyplot as plt
import matplotlib
import squarify  # pip install squarify (algorithm for treemap)
from mpl_toolkits.mplot3d import Axes3D
from pandas import read_csv
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x =[1,2,3,4,5,6,7,8,9,10]
y =[1,2,3,4,5,6,7,8,9,10]
z =[1,2,3,4,5,6,7,8,9,10]



ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()