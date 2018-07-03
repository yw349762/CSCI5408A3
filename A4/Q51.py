import matplotlib.pyplot as plt
import matplotlib
import squarify  # pip install squarify (algorithm for treemap)
from mpl_toolkits.mplot3d import Axes3D
from pandas import read_csv
import numpy as np
from pylab import *
import matplotlib.pyplot as plt
from pandas import read_csv
from mpl_toolkits.basemap import Basemap

fig=plt.figure()
plt.subplot(2, 3, 1)
# Data downloaded from 'http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.csv'
# Setting up custom style
style.use(['seaborn-poster'])

quakeFrame = read_csv('permitcount.csv')

lngs = quakeFrame['XLOC'].astype('float')
lats = quakeFrame['YLOC'].astype('float')
mags = quakeFrame['COUNT(BP_ID)'].astype('float')
#.apply(lambda x: 2 ** x)

earth = Basemap()
# 105.3,-13.9,151.6,22.1 Phillipines
# earth = Basemap(llcrnrlon=105.3,llcrnrlat=-13.9,urcrnrlon=151.6,urcrnrlat=22.1)
earth.drawcoastlines(color='0.50', linewidth=0.25)
#earth.fillcontinents(color='0.95')
#earth.bluemarble(alpha=0.95)
earth.shadedrelief()
plt.scatter(lngs, lats, mags,
            c='blue',alpha=0.5, zorder=10)
plt.xlabel("Q1 plot the number of permits per location")



plt.subplot(2, 3, 2)
data = read_csv('permittypecount.csv')
# Create a new figure of size 10x6 points, using 100 dots per inch
x = np.array([0,1,2,3,4,5])
y = data['COUNT(BP_ID)']
my_xticks = data['PERMIT_TYPE']
plt.bar(x,height=y, facecolor='green', width=0.8, alpha=0.75)
plt.xticks(x, my_xticks)
plt.xlabel('PERMIT_TYPE')
plt.ylabel('Count')
plt.title("Summary of permit type")
plt.grid(True)


plt.subplot(2, 3, 3)
data = read_csv('permitdatediff.csv')
# Create a new figure of size 10x6 points, using 100 dots per inch
x =np.arange(75640)
y = data['DATEDIFF']
print(len(y))
my_xticks = data['BP_ID']
plt.scatter(x,y)
plt.xticks(x,  my_xticks, rotation='vertical')
plt.xlabel('BP_ID')
plt.ylabel('Count')
plt.title("Summary of permit time gaps")


plt.subplot(2, 3, 4)
df = read_csv('valueprojectcount.csv')
df = df.set_index("ALTERNATE_BUILDING_TYPE")
df = df[["spurf","spurf1"]]
df2 = df.sort_values(by="spurf1", ascending=False)

# treemap parameters
x = 0.
y = 0.
width = 100.
height = 100.
cmap = matplotlib.cm.viridis
mini, maxi = df2.spurf.min(), df2.spurf.max()

norm = matplotlib.colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in df2.spurf]
colors[1] = "#FBFCFE"

labels = ["%s\n%d" % (label) for label in zip(df2.index, df2.spurf)]


# make plot
ax = fig.add_subplot(111, aspect="equal")
ax = squarify.plot(df2.spurf, color=colors, label=labels, ax=ax, alpha=.7)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Question 4\n", fontsize=14)

# color bar
# create dummy invisible image with a color map
img = plt.imshow([df2.spurf], cmap=cmap)
img.set_visible(False)
colorbar(img, orientation="vertical", shrink=.96)



plt.subplot(2, 3, 5)
data = read_csv('QUESTION5.csv')
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
plt.show()

fig1 = plt.gcf()
fig1.savefig('Question6.png',dpi=48)
