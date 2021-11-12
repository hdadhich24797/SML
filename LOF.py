import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score

df = pd.read_csv('dataset.csv', sep = ',')
#everything except last column

#df = df[df.columns[:]]
#last_col = df.iloc[:, -1]
#last_col.replace({0:-1}, inplace=True)
#print(last_col.head(2000))
df = df[df.columns[:]]

# print(X.head())
X = np.array(df.head(2000))
#X = np.array(df)
model = LocalOutlierFactor(n_neighbors = 20, algorithm = 'auto', metric = "euclidean", contamination = 'auto')
print("Object created")
y_pred = model.fit_predict(X)
print("Prediction complete")
outlier_index = np.where(y_pred == -1)
print("Outlier index found")
outlier_values = df.iloc[outlier_index]
print("Outlier Values:")
print(outlier_values)
X_scores = model.negative_outlier_factor_
print(X_scores)
print("Done")

#print(y_pred)
#print(accuracy_score(last_col.head(2000).to_numpy(), y_pred))
