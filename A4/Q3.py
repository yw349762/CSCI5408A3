
from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

results = Counter()

data = read_csv('permitdatediff.csv')


# Create a new figure of size 10x6 points, using 100 dots per inch
plt.figure()
x =np.arange(75640)
y = data['DATEDIFF']
print(len(y))
my_xticks = data['BP_ID']
plt.scatter(x,y)
plt.xticks(x,  my_xticks, rotation='vertical')
plt.xlabel('BP_ID')
plt.ylabel('Count')
plt.title("Summary of permit time gaps")
# Workaround for blank image saving
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Question3.png')
