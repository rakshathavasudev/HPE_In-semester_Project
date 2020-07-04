import matplotlib
matplotlib.use('Agg')
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification


#Random Forest model for prediction

dataframe = read_csv('all_indices.csv',usecols=[5]).dropna()
d=read_csv('all_indices.csv',usecols=[1,2,3,4]).dropna()
dataset = dataframe.values
dataset = dataset.astype('float32')
y= dataset.flatten()
d = d.values
X = d.astype('float32')
# print(X,y)
# X, y = make_classification(n_samples=100, n_features=4,n_informative=2, n_redundant=0,random_state=0, shuffle=False)

clf = RandomForestClassifier(max_depth=4, random_state=0,warm_start=True, oob_score=True,)
# print(X,y)
# clf.set_params(n_estimators=700)
clf.fit(X, y)

print(clf.feature_importances_)
print(1 - clf.oob_score_)

# print(clf.predict([[-0.5,0.6]]))
print(clf.score(X,y))