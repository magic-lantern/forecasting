---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.1.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region -->
# Name Analysis and forecasting

Exploration of the United States Social Security Administration babynames dataset made available via R package [babynames](http://hadley.github.io/babynames/).

* Visualizing baby names over time
* Using [Prophet](https://facebook.github.io/prophet/) to forecast if popularity will continue to rise or fall
* Using ARIMA from [StatsModels](https://www.statsmodels.org/) to forecast if popularity will continue to rise or fall
* Evaluation of name popularity using 'phonetic' algorithms
* Probability of child having another child in their same class with a simlar name

As the babynames package is in R, and I generally prefer Python, I've used the [rpy2](https://rpy2.bitbucket.io) library to pull in the appropriate portions of babynames

## Setup

All required packages should be in the first code cell. If you use conda, you likely have most requirements already met. Some specific steps I had to follow from my existing conda environment:

```
# do this if you only want rpy2 and already have R installed.
pip install rpy2
# otherwise run
# conda install rpy2
conda install tzlocal line_profiler
conda install -c conda-forge fbprophet
conda install -c conda-forge abydos
```

For R, make sure you have the package babynames installed by running `install.packages('babynames')` in an R session

## Table of Contents:
* [Load Baby Names Data](#load_data)
* [Prediction with Prophet](#prophet)
* [Prediction with ARIMA](#arima)
* [Combine names based on pronounciation](#sound_analysis)
* [Name popularity](#popularity)
* [Gender overlap in naming](#gender)
<!-- #endregion -->

## To Do:

* Get someone smart to review this work!

```python
import pandas as pd
import numpy as np
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import localconverter
# enable automatic conversion between R dataframes and Pandas Dataframes
pandas2ri.activate()

from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot

import matplotlib.pyplot as plt

import os
import pickle
import itertools

from timeit import default_timer as timer

import multiprocessing
from multiprocessing import Pool
import psutil
from functools import partial

# to prevent some warnings later on
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
```

```python
%load_ext line_profiler
```

```python
import seaborn as sns
sns.set()
sns.set_style("ticks", {
    'axes.grid': True,
    'grid.color': '.9',
    'grid.linestyle': u'-',
    'figure.facecolor': 'white', # axes
})
sns.set_context("notebook")
```

<a id='load_data'></a>


## Loading the name data

Summary of data available in the R package [babynames](http://hadley.github.io/babynames/):

**babynames:** For each year from 1880 to 2017, the number of children of each sex given each name. All names with more than 5 uses are given. (Source: http://www.ssa.gov/oact/babynames/limits.html)

**applicants:** The number of applicants for social security numbers (SSN) for each year for each sex. (Source: http://www.ssa.gov/oact/babynames/numberUSbirths.html)

**lifetables:** Cohort life tables data (Source: http://www.ssa.gov/oact/NOTES/as120/LifeTables_Body.html)

It also includes the following data set from the US Census:

**births:** Number of live births by year, up to 2017. (Source: an Excel spreadsheet from the Census that has since been removed from their website and https://www.cdc.gov/nchs/data/nvsr/nvsr66/nvsr66_01.pdf)

```python
bn_filename = 'babynames.pickle'

if os.path.isfile(bn_filename):
    with open(bn_filename, 'rb') as f:
        orig_df = pickle.load(f)
else:
    bn = importr('babynames')
    orig_df = bn.__rdata__.fetch('babynames')['babynames']
    # fix some data types
    orig_df['year'] = orig_df.year.astype('int32')

    with open(bn_filename, 'wb') as f:
        pickle.dump(orig_df, f)
```

```python
# only want to look at more recent trends
df = orig_df[orig_df.year >= 1990].copy()
```

```python
df.head()
```

```python
df.tail()
```

```python
tops = df[(df.sex == 'F') & ((df.year >= 2012) & (df.year <= 2017))].sort_values(['year', 'n'], ascending=False).groupby('year', as_index=False).head(20)
print('top femail names 2012 - 2017 by number of times in top 20 per year')
print(tops.groupby('name', as_index=False)['n'].count().sort_values('n', ascending=False))
```

```python
a = df[df.name.isin(['Abigail', 'Sophia', 'Sofia', 'Olivia', 'Avery'])]
a_plt = sns.relplot(x='year', y='n', hue='name', style='sex', kind='line', ci=None, data=a)
a_plt.fig.set_size_inches(7.5, 5)
```

Suppose your selected name is Sophia and soundalikes


<a id='prophet'></a>


## Prophet for prediction of naming trends

Prophet is an R/Python library available from: https://facebook.github.io/prophet/

As Prophet seems to be focused on more frequent data, I'm not convinced it works well with low resolution/infrequent data such as this yearly name data. Of course, I may just not understand how to properly specify parameters to get it to work right. *Update:* After having gone through an exercise (below) of similar analysis using ARIMA, I believe the problem is due to the nature of the data and the complexities with baby naming rather than with Prophet/ARIMA.

The input to Prophet is always a dataframe with two columns: **ds** and **y**. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

```python
# select counts without respect to gender
a = orig_df[(orig_df.name.isin(['Sophia', 'Sofia']))].copy()
#a = df[df.name.isin(['Sophia', 'Sofia'])]
pdf = a.groupby(['year'])['n'].sum().reset_index()
pdf.rename(columns={"year": "ds", "n": "y"}, inplace=True)
pdf['ds'] = pd.to_datetime(pdf['ds'], format='%Y')
```

Prophet with just most basic parameters detects vary basic linear trend lines. Prediction is not very detailed and likely worse than what a human would deduce when looking at a graph of the data.

```python
m = Prophet(daily_seasonality=False, weekly_seasonality=False, seasonality_mode='multiplicative')
m.fit(pdf)
# predict 10 years out
future = m.make_future_dataframe(periods=10, freq='Y')
forecast = m.predict(future)
fig = m.plot(forecast)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# show actual data in green
plt.plot(pdf.ds, pdf.y, 'g')
```

**Note:** In this case, we have no within year seasonality for naming (which might actually exist, but we only have year-end totals), but Prophet fits a function depicting within year seasonality anyway

```python
fig = m.plot_components(forecast)
```

Prophet with additional parameters follows the actuals much closer. Prediction still is not very detailed and likely worse than what a human would deduce when looking at a graph of the data and comparing to other naming trends.

```python
m = Prophet(daily_seasonality=False, weekly_seasonality=False, seasonality_mode='multiplicative', changepoint_prior_scale = 0.5)#, yearly_seasonality = 1000)
m.add_seasonality(name='cust2', period=2, fourier_order=5)
m.add_seasonality(name='cust3', period=3, fourier_order=5)
m.add_seasonality(name='cust4', period=4, fourier_order=5)
m.add_seasonality(name='cust5', period=5, fourier_order=5)
m.fit(pdf)
# predict 10 years out
future = m.make_future_dataframe(periods=10, freq='Y')
forecast = m.predict(future)
fig = m.plot(forecast, plot_cap=False)
a = add_changepoints_to_plot(fig.gca(), m, forecast)

# show actual data in green
plt.plot(pdf.ds, pdf.y, 'g')
```

<a id='arima'></a>


## ARIMA for prediction of naming trends

Although this ARIMA section is more detailed than the Prophet section, my sense is that due to outside influences for naming trends (popular media, name saturation, cultural/social trends) it is not possible to accurately predict naming trends from historical naming counts alone.

Resources I found useful when applying ARIMA:
 * https://www.kaggle.com/magiclantern/co2-emission-forecast-with-python-arima-v2/notebook
 * https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
 * https://stats.stackexchange.com/questions/44992/what-are-the-values-p-d-q-in-arima

```python
# to do - add this section
import statsmodels
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from matplotlib.pylab import rcParams
from pandas.plotting import autocorrelation_plot
```

```python
rcParams['figure.figsize'] = 15, 12
```

```python
a = orig_df[(orig_df.name.isin(['Sophia', 'Sofia']))].copy()
ts = a.groupby(['year'])['n'].sum()
# if a.year[0] is a string it treats it as a year; if it is an integer, it treats it as nanoseconds
# see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets for options
ts.index = pd.date_range(str(a.year[0]), periods=len(ts), freq='Y')
ts.head()
```

```python
# don't have monthly data, so this doesn't work
#decomposition = seasonal_decompose(ts)
```

```python
def TestStationaryPlot(ts, plot_label = None):
    rol_mean = ts.rolling(window = 12, center = False).mean()
    rol_std = ts.rolling(window = 12, center = False).std()
    
    plt.plot(ts, color = 'blue',label = 'Original Data')
    plt.plot(rol_mean, color = 'red', label = 'Rolling Mean')
    plt.plot(rol_std, color ='black', label = 'Rolling Std')
    plt.xticks(fontsize = 25)
    plt.yticks(fontsize = 25)
    
    plt.xlabel('Time in Years', fontsize = 25)
    plt.ylabel('Total Emissions', fontsize = 25)
    plt.legend(loc='best', fontsize = 25)
    if plot_label is not None:
        plt.title('Rolling Mean & Standard Deviation (' + plot_label + ')', fontsize = 25)
    else:
        plt.title('Rolling Mean & Standard Deviation', fontsize = 25)
    plt.show(block= True)
    
def TestStationaryAdfuller(ts, cutoff = 0.01):
    ts_test = adfuller(ts, autolag = 'AIC')
    ts_test_output = pd.Series(ts_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    
    for key,value in ts_test[4].items():
        ts_test_output['Critical Value (%s)'%key] = value
    print(ts_test_output)
    
    if ts_test[1] <= cutoff:
        print("Strong evidence against the null hypothesis, reject the null hypothesis. Data has no unit root, hence it is stationary")
    else:
        print("Weak evidence against null hypothesis, time series has a unit root, indicating it is non-stationary ")
```

```python
TestStationaryPlot(ts, 'original')
```

```python
# while most of the data is farily flat for the time period shown (1880 - 1980)
#visual inspection of data doesn't seem to be stationary
TestStationaryAdfuller(ts)
```

```python
# this is what is looked at for stationarity (is that a word?)
#12*(nobs/100)^{1/4}
nobs = len(ts)
12*(nobs/100)**(1/4)
```

```python
moving_avg = ts.rolling(5).mean()
moving_avg_diff = ts - moving_avg
moving_avg_diff.head(13)
```

```python
moving_avg_diff.dropna(inplace=True)
TestStationaryPlot(moving_avg_diff, 'moving')
TestStationaryAdfuller(moving_avg_diff)
```

### Autocorrelation Plots

When looking at all data available (1880 to present) it appears that the current value seems to be highly correlated with the previous 14 or so values; closer to the present value is more correlated. Tested a few different names and correlation lag may be as high as 20.

```python
fig, ax = plt.subplots(figsize = (15, 5))
ax = autocorrelation_plot(ts, ax=ax)
#plt.vlines(range(0, 121, 2), -1, 1, linestyle="dashed")
```

These are autocorrelation plots from Statsmodel

```python
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(ts, lags=50, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(ts, lags=50, ax=ax2)
```

```python
p = range(0, 20) #max based on plots above
d = range(0, 3)
q = range(0, 3)

p = range(0, 21) #max based on plots above
d = range(0, 21)
q = range(0, 21)

pdq = list(itertools.product(p, d, q)) # Generate all different combinations of p, q and q triplets

# for ARIMA model building, only use first 90%, save last 10% for testing/validation
n = .9
train = ts.head(int(len(ts) * .9))
```

Perform a grid search to find the best ARIMA model


# ```python
%%time

import warnings
warnings.filterwarnings('ignore') # ignore warning messages in this cell

aic_results = pd.DataFrame(columns=['aic', 'param'])
for param in pdq:
    try:
        # found that SARIMAX works with a wider range of p,d,q values
        #mod = ARIMA(ts,
        #            order=param,
        #            freq='A-DEC') #see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets for options
        mod = SARIMAX(train,
                      order=param,
                      enforce_stationarity=False,
                      enforce_invertibility=False)
        # some methods fail to converge; increasing max iterations helps
        # some methods have verbose output
        #results = mod.fit(maxiter=200, method='nm')
        results = mod.fit()
        if np.isnan(results.aic):
            a = 1
            #print('No AIC generated for ARIMA:{}'.format(param))
        else:
            #print('ARIMA:{} - AIC:{}'.format(param, results.aic))
            aic_results = aic_results.append({
                'aic': results.aic,
                'param': param},
                ignore_index=True)
    except Exception as e:
        # lots of ARIMA values end in an exception - guess I need to understand this better
        print('exception', e)
        continue
    if len(aic_results) % 20 == 0:
        print('Checked', len(aic_results), 'parameters so far.')

aic_results = aic_results.sort_values('aic').reset_index(drop=True)
print('Best AIC found:', aic_results.head(1).aic.values[0], 'with parameters', aic_results.head(1).param.values[0])

model = SARIMAX(ts, order=aic_results.head(1).param.values[0], enforce_stationarity=False)
model_fit = model.fit()
print(model_fit.summary())

warnings.filterwarnings('default')
# ```


Parallel version of grid search for faster processing

```python
def calc_arima(df_in):
    aic_results = pd.DataFrame(columns=['aic', 'param'])
    print('Checking', len(df_in), 'parameters')
    for param in df_in.param.values:
        try:
            # found that SARIMAX works with a wider range of p,d,q values
            #mod = ARIMA(train,
            #            order=param,
            #            freq='A-DEC') #see https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#anchored-offsets for options
            mod = SARIMAX(train,
                          order=param,
                          enforce_stationarity=False,
                          enforce_invertibility=False)
            # some methods fail to converge with default iterations of 50; increasing max iterations helps
            # some methods have verbose output
            #results = mod.fit(maxiter=200, method='nm')
            results = mod.fit(maxiter=150)
            if np.isnan(results.aic):
                a = 1
                #print('No AIC generated for ARIMA:{}'.format(param))
            else:
                #print('ARIMA:{} - AIC:{}'.format(param, results.aic))
                aic_results = aic_results.append({
                    'aic': results.aic,
                    'param': param},
                    ignore_index=True)
        except Exception as e:
            # lots of ARIMA values end in an exception - guess I need to understand this better
            print('exception', e)
            continue
    return aic_results
```

```python
len(pdq)
```

```python
%%time

import warnings
warnings.filterwarnings('ignore') # ignore warning messages in this cell

pdq_input_df = pd.DataFrame(columns=['aic', 'param'])
pdq_input_df['param'] = pdq

num_processes = psutil.cpu_count(logical=False) - 2
num_partitions = num_processes * 8 #smaller batches to get more frequent status updates
with Pool(processes=num_processes) as pool:
    df_split = np.array_split(pdq_input_df, num_partitions)
    df_out = pd.concat(pool.map(calc_arima, df_split),ignore_index=True)
    
aic_results = df_out.sort_values('aic').reset_index(drop=True)
print('Best AIC found:', aic_results.head(1).aic.values[0], 'with parameters', aic_results.head(1).param.values[0])

model = SARIMAX(train, order=aic_results.head(1).param.values[0], enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit()
print(model_fit.summary())

warnings.filterwarnings('default')
```

```python
num_processes = psutil.cpu_count(logical=False)
num_partitions = num_processes * 4 #smaller batches to get more frequent status updates
print(num_processes, num_partitions)
```

```python
# plot residual errors
residuals = pd.DataFrame(model_fit.resid)
residuals.plot(figsize=(12,6))
plt.show()
residuals.plot(kind='kde', figsize=(12, 6))
plt.show()
print(residuals.describe())
```

```python
model_fit.plot_diagnostics(figsize=(12, 12))
plt.show()
```

Now generate prediction for all data + future

```python
ts.index[-1] + pd.DateOffset(years=10)
```

```python
end_period = ts.index[-1] + pd.DateOffset(years=10)
pred = model_fit.get_prediction(start=ts.index[0], end=end_period, dynamic=False)
pred_ci = pred.conf_int()
pred_ci.head()
```

```python
ax = ts.plot(label='observed', figsize=(16, 8))
pred.predicted_mean.plot(ax=ax, label='One-step ahead forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='g', alpha=.3, label='CI')

ax.set_xlabel('Time (years)')
ax.set_ylabel('Children with Name')
plt.legend()

plt.show()
```

```python
# if want to just check last 10 years of acutals, use this
#forecast = pred.predicted_mean[ts.index[-10]:ts.index[-1]]
#truth = ts[-10:]
# if wnat to check all actuals vs predictions, use this
forecast = pred.predicted_mean[:ts.index[-1]]
truth = ts

# Compute the mean square error
mse = ((forecast - truth) ** 2).mean()
print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
print('The Root Mean Square Error (RMSE) of the forcast: {:.4f}'
      .format(np.sqrt(sum((forecast-truth)**2)/len(forecast))))
```

Now try generating a dynamic forecast for last 10 years of actual data

```python
pred_dynamic = model_fit.get_prediction(start=ts.index[-10], end=ts.index[-1], dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()
```

```python
ax = ts.plot(label='observed', figsize=(16, 8))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], 
                color='g', 
                alpha=.3,
                label='CI')

# # ax.fill_betweenx(ax.get_ylim(), 
# #                  ts.index[-10], 
# #                  ts.index[-1],
# #                  alpha=.1, zorder=-1)

ax.set_xlabel('Time (years)')
ax.set_ylabel('Children with Name')

plt.legend()
plt.show()
```

```python
# Extract the predicted and true values of our time series (just last 10 years for this dynamic forecast)
forecast = pred_dynamic.predicted_mean
original = ts[-10:]

# Compute the mean square error
mse = ((forecast - original) ** 2).mean()
print('The Mean Squared Error (MSE) of the forecast is {}'.format(round(mse, 2)))
print('The Root Mean Square Error (RMSE) of the forcast: {:.4f}'
      .format(np.sqrt(sum((forecast-original)**2)/len(forecast))))
```

<a id='sound_analysis'></a>


### Combine names based on how they sound

As each entry in the babynames dataset is based on spelling, evaluate different sound based algorithms to see how to combine/reduce number of names.

See the [soundconflation](soundconflation.ipynb) notebook for analysis of various phonetic algorithms.

End result is that the Abydos python library has lots of options and I like the Beider & Morse algorithm the best for the purposes of this analysis.

For combination of names based on Beider Morse algorithm, will just look at most recent year.


#### Export names file for testing/evaluation of phonetic algorithms

```python
names = pd.DataFrame()
#apdf = df[['year', 'sex', 'name', 'n', 'prop']][df.year == 1990].head(100).copy()
names['name'] = df.name.unique()
print(names.shape)

with open('ssn_names_only.pickle', 'wb') as f:
    pickle.dump(names, f)
```

Build the data frame of names and beidermorse values

```python
from abydos import phonetic as ap

def process_df(function, label, df_in):
    # this is is beidermorse specific; remove this function partial and you can actually parallelize
    # any method that doesn't require additional parameters
    func = partial(function, language_arg = 'english')
    df_in[label] = df_in.name.map(func)
    return df_in

def parallelize(inputdf, function, label):
    num_processes = psutil.cpu_count(logical=False)
    num_partitions = num_processes * 2 #smaller batches to get more frequent status updates (if function provides them)
    func = partial(process_df, function, label)
    with Pool(processes=num_processes) as pool:
        df_split = np.array_split(inputdf, num_partitions)
        df_out = pd.concat(pool.map(func, df_split))
    return df_out

filename = 'names_beidermorse.pickle'

if os.path.isfile(filename):
    with open(filename, 'rb') as f:
        names = pickle.load(f)
else:
    names = parallelize(names, ap.BeiderMorse().encode, 'beidermorse')
    # convert string of names space separated to a set for much faster lookup
    names['bmset'] = names['beidermorse'].str.split().apply(set)

    with open(filename, 'wb') as f:
        pickle.dump(names, f)

names.head()
```

```python
checklist = names[names.name == 'Sofia'].beidermorse.values[0].split()
def check_fn(input):
    return any(x in input.split() for x in checklist)

names[names.beidermorse.map(check_fn)].head()
```

Create key-value version of Beider Morse mapping - found this improves performance

```python
def f(name, bmset):
    return pd.DataFrame(zip([name] * len(bmset), list(bmset)), columns=('name', 'beidermorse'))
kv_names = pd.concat([f(n,b) for n, b in zip(names['name'], names['bmset'])])
```

```python
def calc_sound_totals(df_in):
    df_out = df_in.copy()
    df_out['counted'] = False
    df_out['alt_n'] = 0
    df_out['alt_prop'] = 0.0
    df_out['mapped_to'] = ''
    
    filt_names = kv_names.merge(df_out, on='name')[['name', 'beidermorse']]

    # process each row of dataframe
    def create_df_out_n(row):
        # should do no further processing if this row has already been counted
        if (df_out.loc[(df_out.name == row['name']) &
                (df_out.year == row.year) &
                (df_out.sex == row.sex) ,
                'counted'].all() == True):
            return

        # find matching names        
        checklist = filt_names[filt_names.name == row['name']].beidermorse.values
        found = filt_names[filt_names.beidermorse.isin(checklist)]['name'].unique()
        
        # aggregate count, excluding counted names, for all found names into df_out_name
        df_out.loc[(df_out.name == row['name']) &
                (df_out.year == row.year) &
                (df_out.sex == row.sex) ,
                'alt_n'] = df_out[(df_out.name.isin(found)) & 
                               (df_out.year == row.year) &
                               (df_out.sex == row.sex) &
                               (df_out.counted == False)]['n'].sum()
        
        # set counted flag for found names in group
        # ? how to update just group ?
        df_out.loc[(df_out.name.isin(found)) & (df_out.year == row.year) & (df_out.sex == row.sex), 'counted'] = True
        df_out.loc[(df_out.name.isin(found)) & (df_out.year == row.year) & (df_out.sex == row.sex), 'mapped_to'] = row['name']

    start = timer()
    gdf = df_out.groupby(['year', 'sex'])
    for name, group in gdf:
        print('processing name:', name)
        g = group.sort_values('n', ascending=False).copy()
        g.apply(create_df_out_n, axis=1)

    end = timer()
    print('create out_n', end - start, 'seconds')
    
    # create df_out_prop
    def create_df_out_prop(row):
        df_out.loc[(df_out.name == row['name']) &
                (df_out.year == row.year) &
                (df_out.sex == row.sex) ,
                'alt_prop'] = row['alt_n'] / gsum

    start = timer()
    for name, group in gdf:
        gsum = group['alt_n'].sum()
        group.apply(create_df_out_prop, axis=1)
    end = timer()
    print('create out_prop', end - start, 'seconds')

    return df_out
```

This process takes about 20 minutes... While this is the improved version, need to improve much more; seems parallelization is the next step

```python
alt = df[df.year == 2017].copy()
alt['counted'] = False
alt['alt_n'] = 0
alt['alt_prop'] = 0.0
alt.head()

filename = 'calc_sound_totals.pickle'

if os.path.isfile(filename):
    with open(filename, 'rb') as f:
        out = pickle.load(f)
else:
    out = calc_sound_totals(alt)

    with open(filename, 'wb') as f:
        pickle.dump(out, f)
```

```python
out.sort_values(['year', 'sex', 'name']).head()
```

```python
out[out.name.isin(['Abigail', 'Sophia', 'Sofia', 'Olivia', 'Avery'])].head()
```

```python
len(out.mapped_to.unique())
```

<a id='popularity'></a>


### Name popularity

Primary/Secondary school is the location most children will interact with a large number of their birth cohort. While many schools use on or around September 1 for the kindergarten cutoff age, it can actually vary quite a bit - see http://ecs.force.com/mbdata/MBQuest2RTanw?rep=KK3Q1802

In highschool, college, and adulthood interactions are with a broader range of ages than in early school years, so a larger time period might be more appropriate for that type of analysis.

To Do:
* Should consider regional distributions of names, but where to get that data?
* Should consider ethnic popularity by region, but where to get that data?

National school class size: 21

Source: https://nces.ed.gov/fastfacts/display.asp?id=28, though only has public schools from 2011-2012, so is outdated

National Average Public School size: 503

Source: https://www.publicschoolreview.com/average-school-size-stats/national-data

Suppose we want to see how many other kids would have names like 'Sophia':

```python
from scipy.stats import binom

# p = probability/proportion of name
# n = size of group to consider
def calc_dup_prob(p, n):
    k = 2
    ans = 0.0
    for x in range(k, n + 1):
        ans += binom.pmf(x, n, p)
        # alternatively see https://en.wikipedia.org/wiki/Binomial_distribution#Cumulative_distribution_function
        # comb(n, x) * p**x * (1 - p)**(n-x))
    return ans

school_size = 503
class_size = 21

print("Probability of child with same name in a given class: ", calc_dup_prob(out[out.name == 'Sophia'].alt_prop.sum(), class_size))
print("Probability of child with same name in a given school: ", calc_dup_prob(out[out.name == 'Sophia'].alt_prop.sum(), school_size))
```

<a id='gender'></a>


### Check if name is mostly one gender or is there some amount of overlap

Gender overlap levels were selected to approximately balance the high and medium overlap groups

```python
out[(out.name == 'Sophia') & (out.mapped_to == 'Sophia')]
```

```python
mnames = out[(out.alt_n > 0) & (out.sex == 'M')]
fnames = out[(out.alt_n > 0) & (out.sex == 'F')]
both = out[out.name.isin(set(mnames.mapped_to.unique()) & set(fnames.mapped_to.unique()))].copy()
print('Number of names that are both gender:', len(both)//2)
```

```python
both.head()
```

```python
# # to evaluate all names and report values
# for name in both.name.unique():
#     fcount = both[(both.name == name) & (both.sex == 'F')].alt_n.values[0]
#     mcount = both[(both.name == name) & (both.sex == 'M')].alt_n.values[0]
#     if ((fcount <= (mcount + mcount * .20)) & (fcount >= (mcount - mcount * .20))):
#         print('name: ', name, 'has high gender overlap')
#     elif ((fcount <= (mcount + mcount * .40)) & (fcount >= (mcount - mcount * .40))):
#         print('name: ', name, 'has medium gender overlap')
#     else:
#         print('name: ', name, 'has low gender overlap')
        
def gender_eval(name):
    fcount = both[(both.name == name) & (both.sex == 'F')].alt_n.values[0]
    mcount = both[(both.name == name) & (both.sex == 'M')].alt_n.values[0]
    if ((fcount <= (mcount + mcount * .40)) & (fcount >= (mcount - mcount * .40))):
        return 'high'
    elif ((fcount <= (mcount + mcount * .70)) & (fcount >= (mcount - mcount * .70))):
        return 'medium'
    else:
        return 'low'

both['gender_overlap'] = both.name.apply(gender_eval)
both.gender_overlap.value_counts()
```

```python
# look at most common highly overlapping names
samp = both[both.gender_overlap == 'high'].sort_values('alt_n', ascending=False).head(5).name
both[both.name.isin(samp)].sort_values(['name', 'sex'])
```

```python
# random sample of medium overlapping names
samp = samp = both[both.gender_overlap == 'medium'].name.sample(n=3, random_state=2213)
both[both.name.isin(samp)].sort_values(['name', 'sex'])
```

```python
# random sample of low overlapping names
samp = samp = both[both.gender_overlap == 'low'].name.sample(n=3, random_state=2213)
both[both.name.isin(samp)].sort_values(['name', 'sex'])
```

```python

```
