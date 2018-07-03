
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

results = Counter()

data = read_csv('permittypecount.csv')


# Create a new figure of size 10x6 points, using 100 dots per inch
plt.figure()
x = np.array([0,1,2,3,4,5])
y = data['COUNT(BP_ID)']
my_xticks = data['PERMIT_TYPE']
plt.bar(x,height=y, facecolor='green', width=0.8, alpha=0.75)
plt.xticks(x, my_xticks)
plt.xlabel('PERMIT_TYPE')
plt.ylabel('Count')
plt.title("Summary of permit type")
plt.grid(True)
# Workaround for blank image saving
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Question2.png', dpi=200)
