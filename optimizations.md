---
jupyter:
  jupytext:
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

## Explorations in how to improve performance

```python
import pandas as pd
import numpy as np
import os
import pickle
from timeit import default_timer as timer
```

```python
%load_ext line_profiler
```

```python
with open('babynames.pickle', 'rb') as f:
    orig_df = pickle.load(f)
orig_df.shape
```

```python
with open('names_beidermorse.pickle', 'rb') as f:
    names = pickle.load(f)
names.shape
```

```python
# this takes about 35 seconds
start = timer()
def f(name, bmset):
    return pd.DataFrame(zip([name] * len(bmset), list(bmset)), columns=('name', 'beidermorse'))

kv_names = pd.concat([f(n,b) for n, b in zip(names['name'], names['bmset'])])
end = timer()
print(end - start, 'seconds')
print(kv_names.shape)
```

```python
# setup test dataset
df = orig_df[orig_df.year == 1990].sample(frac=.01, random_state=2213).copy()
#df = orig_df[orig_df.year == 1990].copy()
df.sort_values(['year', 'sex', 'name'], inplace=True)
df.shape
```

# ```python
start = timer()
calc_sound_totals(alt)
end = timer()
print(end - start, 'seconds')
# right now it takes ~54 seconds to process 500 randomly selected records
# ~26 seconds to process 250 randomly selected records
# 95% or so of time is spend in the first block "out_n"
# ```


%lprun -f create_df_out_n calc_sound_totals(alt)

```python
# this is very slow look at optimizing/parallelizing
# see: https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
# potentially look at https://github.com/jmcarpenter2/swifter
#   see realted blog post: https://medium.com/@jmcarpenter2/swiftapply-automatically-efficient-pandas-apply-operations-50e1058909f9

# process each row of dataframe
def create_df_out_n(row):
    # should do no further processing if this row has already been counted
    if (row['counted'] == True):
        return

    # find matching names
    #checklist = names[names.name == row['name']].beidermorse.values[0].split()
    #find = lambda i: any(x in i for x in checklist)
    #found = names[names.bmset.map(find)].name
    # this new method takes about half the time - 15 seconds for 250 rows
    #checklist = names[names.name == row['name']].bmset.values[0]
    #found = kv_names[kv_names.beidermorse.isin(checklist)]['name'].unique()
    
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

# create df_out_prop
def create_df_out_prop(row, gsum):
    df_out.loc[(df_out.name == row['name']) &
            (df_out.year == row.year) &
            (df_out.sex == row.sex) ,
            'alt_prop'] = row['alt_n'] / gsum


    
def calc_sound_totals():
    gdf = df_out.groupby(['year', 'sex'])
    for name, group in gdf:
        print('processing name:', name)
        g = group.sort_values('n', ascending=False).copy()
        g.apply(create_df_out_n, axis=1)

    for name, group in gdf:
        gsum = group['alt_n'].sum()
        group.apply(create_df_out_prop, axis=1, args=(gsum,))
```

```python
df_out = df.copy()
df_out['counted'] = False
df_out['alt_n'] = 0
df_out['alt_prop'] = 0.0

filt_names = kv_names.merge(df, on='name')[['name', 'beidermorse']]

#%lprun -f create_df_out_n calc_sound_totals()
start = timer()
out = calc_sound_totals()
end = timer()
print('took:', end - start, 'seconds')
```

# ```python
# this is very slow look at optimizing/parallelizing
# see: https://stackoverflow.com/questions/26187759/parallelize-apply-after-pandas-groupby
# potentially look at https://github.com/jmcarpenter2/swifter
#   see realted blog post: https://medium.com/@jmcarpenter2/swiftapply-automatically-efficient-pandas-apply-operations-50e1058909f9

def calc_sound_totals(df_in):
    df_out = df_in.copy()
    df_out['counted'] = False
    df_out['alt_n'] = 0
    df_out['alt_prop'] = 0.0

    # process each row of dataframe
    def create_df_out_n(row):
        # should do no further processing if this row has already been counted
        if (row['counted'] == True):
            return

        # find matching names
        checklist = names[names.name == row['name']].beidermorse.values[0].split()
        find = lambda i: any(x in i for x in checklist)
        found = names[names.bmset.map(find)].name

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
# ```


```python
%%timeit
rn = 'Michael'
checklist = names[names.name == rn].bmset.values[0]
found = kv_names[kv_names.beidermorse.isin(checklist)]['name'].unique()
```

```python
%%timeit
rn = 'Michael'
checklist = names[names.name == rn].beidermorse.values[0].split()
find = lambda i: any(x in i for x in checklist)
found = names[names.bmset.map(find)].name
```

```python
%%timeit
rn = 'Michael'
checklist = kv_names[kv_names.name == rn].beidermorse
found = kv_names[kv_names.beidermorse.isin(checklist)]['name'].unique()
```

```python
kn = kv_names.values
```

```python
%%timeit
rn = 'Michael'
bm = kn[np.where(kn == rn)[0]][:,1]
out = [kn[np.where(kn == x)[0]][:,0] for x in bm]
found = np.unique(np.concatenate( out, axis=0 ))
```

```python
ni = kv_names.set_index('name')
bi = kv_names.set_index('beidermorse')
```

```python
%%timeit
rn = 'Michael'
found = bi.loc[ni.loc[rn].beidermorse].name.unique()
```

```python
%%timeit
rn = 'Michael'
checklist = kv_names[kv_names.name == rn].beidermorse
found = kv_names.merge(checklist, on='beidermorse')['name'].unique()
```

```python
# if I filter the list to just those in this year/group, performance is much improved.
sub = kv_names.merge(df, on='name')[['name', 'beidermorse']]
```

```python
%%timeit
rn = 'Michael'
checklist = sub[sub.name == rn].beidermorse.values
found = sub[sub.beidermorse.isin(checklist)]['name'].unique()
```

```python
%prun -l 10 kv_names[kv_names.beidermorse.isin(['zYsDki'])]['name'].unique()
```

```python
checklist = kv_names[kv_names.name == 'Michael'].beidermorse
%prun -l 10 kv_names[kv_names.beidermorse.isin(checklist)]['name'].unique()
```

```python
%%timeit
sub = kv_names.merge(df, on='name')[['name', 'beidermorse']]
```

```python

```
