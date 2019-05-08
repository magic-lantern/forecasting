---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.0'
      jupytext_version: 1.0.5
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

## Name Analysis and forecasting

Exploration of the United States Social Security Administration babynames dataset made available via R package [babynames](http://hadley.github.io/babynames/).

* Plotting names
* Using [Prophet](https://facebook.github.io/prophet/) to forecast if popularity will continue to rise or fall

As the babynames package is in R, and I generally prefer Python, I've used the [rpy2](https://rpy2.bitbucket.io) library to pull in the appropriate portions of babynames


### To Do:

* Combine name counts based on soundex or metaphone algorithm - possibly using [Fuzzy](https://github.com/yougov/Fuzzy) or [Jellyfish](https://github.com/jamesturk/jellyfish)
* Check if name is mostly one gender or another
* Calculate probability that child with name X will have 1 or more other children with the same name in a class of 21 (number selected based on data at https://nces.ed.gov/fastfacts/display.asp?id=28, though only has public schools from 2011-2012, so may be outdated)
  * As children enter school based on birthdate around a cutoff date, would need to consider +/- 2 years
  * Should also consider regional popularity, but where to get that data?
* Compare ARIMA to Prophet for forecast - As Prophet seems to be focused on more frequent data, I'm not convinced it works well with low resolution/infrequent data such as this yearly name data. Of course, I may just not understand how to properly specify parameters to get it to work right. Some URLs with more information regarding ARIMA in python:
  * https://machinelearningmastery.com/arima-for-time-series-forecasting-with-python/
  * https://blog.exploratory.io/is-prophet-better-than-arima-for-forecasting-time-series-fa9ae08a5851
* Currently just looking at data since 1990. Would including more historical data improve forecast?

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

from timeit import default_timer as timer

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

    with open(bn_filename, 'wb') as f:
        pickle.dump(orig_df, f)
```

```python
orig_df.head()
```

```python
# only want to look at more recent trends
df = orig_df[orig_df.year >= 1990].copy()
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


The input to Prophet is always a dataframe with two columns: **ds** and **y**. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

```python
# select counts without respect to gender
a = df[df.name.isin(['Sophia', 'Sofia'])]
pdf = a.groupby(['year'])['n'].sum().reset_index()
pdf.rename(columns={"year": "ds", "n": "y"}, inplace=True)
pdf['ds'] = pd.to_datetime(pdf['ds'], format='%Y')
```

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

```python
fig = m.plot_components(forecast)
```

```python
df.head()
```

```python
df.shape
```

### Combine names based on how they sound

As each entry in the babynames dataset is based on spelling, evaluate different sound based algorithms to see how to combine/reduce number of names.


#### Export names file for testing/evaluation of phonetic algorithms

```python
names = pd.DataFrame()
#apdf = df[['year', 'sex', 'name', 'n', 'prop']][df.year == 1990].head(100).copy()
names['name'] = df.name.unique()
print(names.shape)

with open('ssn_names_only.pickle', 'wb') as f:
    pickle.dump(names, f)
```

#### See the [soundconflation](soundconflation.ipynb) notebook for analysis of various phonetic algorithms.

End result is that the Abydos python library has lots of options and I like the Beider & Morse algorithm the best for the purposes of this analysis.


```python
from abydos import phonetic as ap
import multiprocessing
from multiprocessing import Pool
import psutil
from functools import partial

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

names[names.beidermorse.map(check_fn)]
```

```python
# setup test dataset
alt = df[df.year == 1990].sample(n=10, random_state=2213).copy()
alt.shape
```

```python
alt.sort_values(['year', 'sex', 'name'], inplace=True)
```

```python
alt['counted'] = False
alt['alt_n'] = 0
alt['alt_prop'] = 0.0
```

```python
# add another year
alt = alt.append({'year': 1991,
                  'sex': 'F',
                  'name': 'Michael',
                  'n': 42,
                  'prop': 0.000030,
                 }, ignore_index = True)
# add similar names to existing year
alt = alt.append({'year': 1990,
                  'sex': 'F',
                  'name': 'Mychael',
                  'n': 42,
                  'prop': 0.000030,
                 }, ignore_index = True)
# add similar names, different gender to existing year
alt = alt.append({'year': 1990,
                  'sex': 'M',
                  'name': 'Mychael',
                  'n': 42,
                  'prop': 0.000030,
                 }, ignore_index = True)
# if added too many times
# alt = alt.drop(alt[(alt.name == 'Michael') & (alt.n == 42)].index)
# alt = alt.drop(alt[(alt.name == 'Mychael') & (alt.n == 42)].index)
```

```python
alt['counted'] = False
alt['alt_n'] = 0
alt['alt_prop'] = 0.0

# process each row of dataframe
def create_alt_n(row):
    print(row['name'])

    # should do no further processing if this row has already been counted
    if (row['counted'] == True):
        return

    # find matching names
    checklist = names[names.name == row['name']].beidermorse.values[0].split()
    find = lambda i: any(x in i for x in checklist)
    found = names[names.bmset.map(find)].name
    
    # aggregate count, excluding counted names, for all found names into alt_name
    alt.loc[(alt.name == row['name']) &
            (alt.year == row.year) &
            (alt.sex == row.sex) ,
            'alt_n'] = alt[(alt.name.isin(found)) & 
                           (alt.year == row.year) &
                           (alt.sex == row.sex) &
                           (alt.counted == False)]['n'].sum()
    
    # set counted flag for found names in group
    # ? how to update just group ?
    alt.loc[(alt.name.isin(found)) & (alt.year == row.year) & (alt.sex == row.sex), 'counted'] = True

gdf = alt.groupby(['year', 'sex'])
for name, group in gdf:
    g = group.sort_values('n', ascending=False).copy()
    g.apply(create_alt_n, axis=1)

# create alt_prop
def create_alt_prop(row):
    alt.loc[(alt.name == row['name']) &
            (alt.year == row.year) &
            (alt.sex == row.sex) ,
            'alt_prop'] = row['alt_n'] / gsum

for name, group in gdf:
    print(name)
    gsum = group['alt_n'].sum()
    group.apply(create_alt_prop, axis=1)    
    
alt
```

The above sample data set appears to be correct; now run process for all years

```python
def f(name, bmset):
    return pd.DataFrame(zip([name] * len(bmset), list(bmset)), columns=('name', 'beidermorse'))
kv_names = pd.concat([f(n,b) for n, b in zip(names['name'], names['bmset'])])
```

```python
# this is very slow look at optimizing/parallelizing
# see: https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
# potentially look at https://github.com/jmcarpenter2/swifter
#   see realted blog post: https://medium.com/@jmcarpenter2/swiftapply-automatically-efficient-pandas-apply-operations-50e1058909f9

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
        if (row['counted'] == True):
            return

        #print('evaluating ', row['name'], ' ------------- ', row['counted'])
        # find matching names        
        checklist = filt_names[filt_names.name == row['name']].beidermorse.values
        found = filt_names[filt_names.beidermorse.isin(checklist)]['name'].unique()
        
        #print('after found section:\n', df_out.loc[(df_out.name.isin(found)) & (df_out.year == row.year) & (df_out.sex == row.sex)])

        # aggregate count, excluding counted names, for all found names into df_out_name
        df_out.loc[(df_out.name == row['name']) &
                (df_out.year == row.year) &
                (df_out.sex == row.sex) ,
                'alt_n'] = df_out[(df_out.name.isin(found)) & 
                               (df_out.year == row.year) &
                               (df_out.sex == row.sex) &
                               (df_out.counted == False)]['n'].sum()
        
        #print('after sum section:\n', df_out.loc[(df_out.name.isin(found)) & (df_out.year == row.year) & (df_out.sex == row.sex)])

        # set counted flag for found names in group
        # ? how to update just group ?
        df_out.loc[(df_out.name.isin(found)) & (df_out.year == row.year) & (df_out.sex == row.sex), 'counted'] = True
        df_out.loc[(df_out.name.isin(found)) & (df_out.year == row.year) & (df_out.sex == row.sex), 'mapped_to'] = row['name']
        
        #print('end of function:\n', df_out.loc[(df_out.name.isin(found)) & (df_out.year == row.year) & (df_out.sex == row.sex)])

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

```python
alt['counted'] = False
alt['alt_n'] = 0
alt['alt_prop'] = 0.0

out = calc_sound_totals(alt)
```

```python
out.sort_values(['year', 'sex', 'name'])
```

```python

```
