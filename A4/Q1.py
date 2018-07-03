from pylab import *
import matplotlib.pyplot as plt
from pandas import read_csv
from mpl_toolkits.basemap import Basemap

# Data downloaded from 'http://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/4.5_month.csv'
# Setting up custom style
style.use(['seaborn-poster'])

quakeFrame = read_csv('permitcount.csv')

lngs = quakeFrame['XLOC'].astype('float')
lats = quakeFrame['YLOC'].astype('float')
mags = quakeFrame['COUNT(BP_ID)'].astype('float')
#.apply(lambda x: 2 ** x)


plt.figure(figsize=(14, 8))
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


# Workaround for blank image saving
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Question1.png', dpi=350)


