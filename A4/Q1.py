from pandas import read_csv
import matplotlib.pyplot as plt
from collections import Counter

results = Counter()

data = read_csv('permittypecount.csv')

#data['features'] = data.text.str.split()

print(data['PERMIT_TYPE'])

#data.features.apply(lambda x : results.update([e.strip(':,.').lower() for e in x if e.startswith('@') and len(e)>1]) )
#print(results.most_common(20))

res = results.most_common(20)

# Create a new figure of size 10x6 points, using 100 dots per inch
plt.figure(figsize=(10,6), dpi=100)
plt.bar(x=[n*10 for n in range(1,21)],height=[b for a, b in res], facecolor='green', width=6, alpha=0.75)
plt.xticks([n*10 for n in range(1,21)], [a for a, b in res],rotation=45, horizontalalignment='right')

plt.xlabel('Mentions')
plt.ylabel('Count')
plt.title("Summary of mentions")
plt.axis([0, 210, 0, 4000])
plt.grid(True)

# Workaround for blank image saving
fig1 = plt.gcf()
plt.show()
plt.draw()
fig1.savefig('mention_bp.png', dpi=200)

