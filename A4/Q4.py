#reference https://gist.github.com/gVallverdu/0b446d0061a785c808dbe79262a37eea
import matplotlib.pyplot as plt
import matplotlib
import squarify  # pip install squarify (algorithm for treemap)
from pandas import read_csv

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
fig = plt.figure(figsize=(12, 10))
fig.suptitle("Question 4", fontsize=20)
ax = fig.add_subplot(111, aspect="equal")
ax = squarify.plot(df2.spurf, color=colors, label=labels, ax=ax, alpha=.7)
ax.set_xticks([])
ax.set_yticks([])
ax.set_title("Question 4\n", fontsize=14)

# color bar
# create dummy invisible image with a color map
img = plt.imshow([df2.spurf], cmap=cmap)
img.set_visible(False)
fig.colorbar(img, orientation="vertical", shrink=.96)
#plt.show()
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('Question4.png', dpi=200)


