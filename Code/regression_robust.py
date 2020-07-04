import matplotlib
matplotlib.use('Agg')
import numpy as np
from pandas import read_csv
from matplotlib import pyplot as plt
from scipy import stats
from sklearn import linear_model, datasets


n_samples = 50
n_outliers = 5

dataframe = read_csv('nbr2.csv',usecols=[1]).dropna()
dataset = dataframe.values
dataset = dataset.astype('float32')
y= dataset.flatten()
A=np.arange(0,28)
X = np.reshape(A, (-1, 1))

np.random.seed(0)

print(X,y)

# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y)

# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)

inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)
print(line_X,line_y,line_y_ransac)

# Compare estimated coefficients
print("Estimated coefficients (true, linear regression, RANSAC):")
print( lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.',
            label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.',
            label='Outliers')
plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Linear regressor')
plt.plot(line_X, line_y_ransac, color='cornflowerblue', linewidth=lw,
         label='RANSAC regressor')

slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(),y)
# print(slope_intercept(line_X,line_y_ransac))
print(slope,intercept,std_err)
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.savefig("ransac.png")
plt.show()