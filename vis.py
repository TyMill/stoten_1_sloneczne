import matplotlib.pyplot as plt

df.hist(bins=50, figsize=(20,15))
plt.show()

plt.scatter(df['DO'], df['BOD'])
plt.xlabel('DO')
plt.ylabel('BOD')
plt.show()


import seaborn as sns

plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f")
plt.show()

plt.figure(figsize=(12,6))
plt.plot(df.index, df['DO'], label='DO')
plt.plot(df.index, df['BOD'], label='BOD')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
