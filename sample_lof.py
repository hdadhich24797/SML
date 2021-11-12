import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor

data = pd.read_csv("Iris_Data.csv")
#Plotting graph between PetalLengthCm & SepalLengthCm
sns.set_style("whitegrid")
sns_plot=sns.FacetGrid(data, hue="species", size=6,palette=['darkorange','black','red'])\
   .map(plt.scatter, "petal_length", "sepal_length") \
   .add_legend()
plt.show()

a = list(data['petal_length'])
x= np.array(a)
b = list(data['sepal_length'])
y= np.array(b)
z=np.array([a,b])
X=z.T
clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
X_scores = clf.negative_outlier_factor_
plt.title("Local Outlier Factor (LOF)")
plt.scatter(a,b, edgecolor='grey', s=30, label='Data points',facecolors='none')
 #plotting circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
plt.scatter(a,b, s=1300 * radius, edgecolors='pink',facecolors='none', label='Outlier scores')
plt.axis('tight')
plt.xlim((0, 10))
plt.ylim((1, 10))
legend = plt.legend(loc='upper left')
legend.legendHandles[0]._sizes = [10]
legend.legendHandles[1]._sizes = [20]
plt.show()