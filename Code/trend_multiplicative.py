import matplotlib
matplotlib.use('Agg')
from pandas import read_csv
import pandas
from matplotlib import pyplot
from statsmodels.tsa.seasonal import seasonal_decompose
# series = read_csv('ndvi2.csv',usecols=[0])
# series1 = read_csv('ndvi2.csv',usecols=[1])
# series1=series1+2
# # print(series1)
# df=pandas.concat([series,series1],axis=1)
# df.columns = range(df.shape[1])




#Function to plot time series multiplicative model
def fun(name):
    df=read_csv(name,header=0, index_col=0,low_memory=False).dropna().astype('float32')
    print(df)
    result = seasonal_decompose(df, model='multiplicative',freq=12)
    result.plot()
    n=name.replace('.csv','')
    pyplot.savefig(n+"-trend.png")

fun('ndvi2.csv')
fun('ndmi.csv')
fun('nbr.csv')
fun('nbr2.csv')
