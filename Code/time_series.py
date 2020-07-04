import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os
import tkinter
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from patsy import dmatrices

import pandas as pd

from pandas.plotting import register_matplotlib_converters
from pyproj import Proj, transform
from osgeo import gdal

# Run this in command line fmask_usgsLandsatStacked.py -o cloud.img --scenedir scene_directory
#This is a command line scripts that process an untarred USGS Landsat scene. This command will take a given 
#scene directory, find the right images, and create an output file called cloud.img:




ind = pd.read_csv("all_indices.csv").dropna()
# ind2=pd.read_csv('ndmi.csv')
# ind3=pd.read_csv('nbr.csv')
# ind4=pd.read_csv('nbr.csv')
# print(ind)

df=ind
fig, ax = plt.subplots(figsize=(10, 10))

# Add x-axis and y-axis
ax.plot(ind['Date'],
        ind['ndvi'],
        color='purple')

# Set title and labels for axes
ax.set(xlabel="Date",
       ylabel="NDVI indices",
       title="NDVI timeseries")
plt.setp(ax.get_xticklabels(), rotation=45)
plt.show()

plt.savefig('NDVI.png')

fig2, bx = plt.subplots(figsize=(10, 10))
bx.plot(ind['Date'],ind['ndmi'],color='red')
bx.set(xlabel="Date",ylabel="NDMI indices",title="NDMI timeseries")
plt.setp(bx.get_xticklabels(), rotation=45)
plt.show()
plt.savefig('NDMI.png')
fig2, bx = plt.subplots(figsize=(10, 10))
bx.plot(ind['Date'],ind['nbr'],color='red')
bx.set(xlabel="Date",ylabel="NBR indices",title="NDMI timeseries")
plt.setp(bx.get_xticklabels(), rotation=45)
plt.show()
plt.savefig('NBR.png')
fig2, bx = plt.subplots(figsize=(10, 10))
bx.plot(ind['Date'],ind['nbr2'],color='red')
bx.set(xlabel="Date",ylabel="NBR2 indices",title="NDMI timeseries")
plt.setp(bx.get_xticklabels(), rotation=45)
plt.show()
plt.savefig('NBR2.png')
  

# df['LOG_ndvi'] = np.log(df['positive_ndvi'])

# print(df['LOG_ndvi'])

# bx.plot(ind['Start'],
#         ind['LOG_ndvi'],
#         color='purple')

# # Set title and labels for axes
# bx.set(xlabel="Date",
#        ylabel="log NDVI indices",
#        title="NDVI timeseries")
# plt.setp(bx.get_xticklabels(), rotation=45)
# plt.show()

# plt.savefig('NDVI-log.png')
def fun(t,s,i,a):
        y=i+t*s+a*np.sin(2*np.pi*t/365)#0.008919839005556405 0.04831602354648665
        return y
ndvi=[]
ndmi=[]
nbr=[]
nbr2=[]


#Optimal values obtained after Robust regression ransac are passed for each spectrum variable
for i  in range(0,len(df)*30,30):
        k=i
        if k>365:
                k=k%365
        ndvi.append(fun(k,0.008919839005556405,0.04831602354648665, 0.398))
        ndmi.append(fun(k,0.0014202476490869328,0.07457688025474432,0.395862775))
        nbr.append(fun(k,0.0027404141917694246,0.18217220716866586 ,0.468143138))
        nbr2.append(fun(k,0.0018662524149107294 ,0.18907923887258973 ,0.390172863))

# print(y)
def sig(y):
        m=max(y)
        print(m,1/(1+np.power(np.e,-m)))
        Y=[]
        for i in range(len(y)):
                Y.append(1/(1+np.power(np.e,-y[i])))
        return Y
# print(Y)
W,X,Y,Z=sig(ndvi),sig(ndmi),sig(nbr),sig(nbr2)
fig2, gx = plt.subplots(figsize=(10, 10))
gx.plot(ind['Date'],W,color='purple')
gx.set(xlabel="Date",ylabel="NDVI",title="NDVI timeseries")
plt.setp(gx.get_xticklabels(), rotation=45)
plt.savefig('NDVI-time.png')

fig2, gx = plt.subplots(figsize=(10, 10))
gx.plot(ind['Date'],X,color='purple')
gx.set(xlabel="Date",ylabel="NDMI",title="NDMI timeseries")
plt.setp(gx.get_xticklabels(), rotation=45)
plt.savefig('NDMI-time.png')

fig2, gx = plt.subplots(figsize=(10, 10))
gx.plot(ind['Date'],Y,color='purple')
gx.set(xlabel="Date",ylabel="NBR",title="NBR timeseries")
plt.setp(gx.get_xticklabels(), rotation=45)
plt.savefig('NBR-time.png')

fig2, gx = plt.subplots(figsize=(10, 10))
gx.plot(ind['Date'],Z,color='purple')
gx.set(xlabel="Date",ylabel="NBR2",title="NBR2 timeseries")
plt.setp(gx.get_xticklabels(), rotation=45)
plt.savefig('NBR2-time.png')

