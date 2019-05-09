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

# Evaluation of techniques for name conflation

A little background and details on early phonetic algorithms: https://stackabuse.com/phonetic-similarity-of-words-a-vectorized-approach-in-python/
* Metaphone: https://en.wikipedia.org/wiki/Metaphone#Metaphone_3
* Soundex: https://en.wikipedia.org/wiki/Soundex

Some Python libraries to fill desired need:
* Fuzzy - Has a problem with encoding
* Phonetics - https://pypi.org/project/phonetics/
* Jellyfish - Works great, though only supports a few different algorithms
* Metaphone: https://pypi.org/project/Metaphone/ A Python implementation of the Metaphone and Double Metaphone algorithms
* Pyphonetics - https://github.com/Lilykos/pyphonetics and https://pypi.org/project/pyphonetics/
* https://github.com/japerk/nltk-trainer/blob/master/nltk_trainer/featx/phonetics.py Soundex, Metaphone, NYSIIS, Caverphone
* Abydos - https://pypi.org/project/abydos/ seems to be the granddaddy of them all. Lots of algorithms implemented - from the early Soundex to more modern algorithms such as Beider-Morse Phonetic Matching; also includes some non-English algorithms. 

Some examples and research behind phonetic algorithms:
  * MetaSoundex: http://www.informaticsjournals.com/index.php/gjeis/article/view/19822
  * Comparison of Caverphone, DMetaphone, NYSIIS, Soundex: https://www.scitepress.org/Papers/2016/59263/59263.pdf, recommends use of Metaphone for English dictionary words and NYSIIS for street names.
  * "Analysis and Comparative Study on Phonetic Matching Techniques"  https://pdfs.semanticscholar.org/9cbc/abee9d8911c65d2d4847bb612bae2f0c83af.pdf
  * "Phonetic Matching: A Better Soundex" (aka Beider-Morse algorithm) https://www.stevemorse.org/phonetics/bmpm2.htm
  * "Study Existing Various Phonetic Algorithms and Designing and Development of a working model for the New Developed Algorithm and Comparison by implementing it with Existing Algorithm(s)" http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.677.3003&rep=rep1&type=pdf

As an FYI, Apache Solr includes the following phonetic matching algorithms:
* Beider-Morse Phonetic Matching (BMPM)
* Daitch-Mokotoff Soundex
* Double Metaphone
* Metaphone
* Soundex
* Refined Soundex
* Caverphone
* Kölner Phonetik a.k.a. Cologne Phonetic
* NYSIIS

```python
import pandas as pd
import numpy as np
import pickle
```

```python
import fuzzy
```

```python
soundex = fuzzy.Soundex(10)
soundex('fuzzy')
```

```python
dmeta = fuzzy.DMetaphone()
dmeta('fuzzy')
```

```python
fuzzy.nysiis('fuzzy')
'FASY'
```

```python
"""
for name in names:
    print('Name: ', name)
    print('Soundex: ', soundex(name))
    print('Metaphone: ', dmeta(name))
    print('Nysiis: ', fuzzy.nysiis(name))
    """
```

```python
name = 'Sarah'
n2 = name.encode(encoding='ascii',errors='strict')
n2
```

The fuzzy library has some encoding errors that seem to require modification of fuzzy source code to resolve.


# ```python
#names = ['Sophia', 'Sofia', 'Seth']
names = ['Sarah', 'Sara']
names = ['Jamie', 'Jenna', 'Joanna', 'Jenny', 'Jaime']
for n in names:
    name = n.encode(encoding='ascii',errors='strict')
    print(name)
    print('Soundex: ', soundex(name))
    print('Metaphone: ', dmeta(name))
    print('Nysiis: ', fuzzy.nysiis(n))
# ```


```python
import jellyfish

names = ['Sarah', 'Sara', 'Seth', 'Beth', 'Aaron', 'Erin']
names = ['Jamie', 'Jenna', 'Joanna', 'Jenny', 'Jaime']
for name in names:
    print('Name: ', name)
    print('   Metaphone: ', jellyfish.metaphone(name))
    print('   Soundex: ', jellyfish.soundex(name))
    print('   Nysiis: ', jellyfish.nysiis(name))
    print('   Match Rating: ', jellyfish.match_rating_codex(name))
```

```python
from abydos import phonetic as ap
```

And there are some hybrid phonetic algorithms that employ multiple underlying
phonetic algorithms:
    - Oxford Name Compression Algorithm (ONCA) (:py:class:`.ONCA`)
    - MetaSoundex (:py:class:`.MetaSoundex`)

```python
ap.ONCA()
```

```python
s = ap.Soundex()
```

```python
s.encode("Sara")
```

```python
bm = ap.BeiderMorse()
```

```python
print(bm.encode('Sarah', language_arg='english'))
print(bm.encode('Jean', language_arg='english'))
print(bm.encode('Jean', language_arg='french'))
print(bm.encode('Jean', match_mode='exact'))
print(bm.encode('Jean'))
```

```python
# go through all names, add column for metaphone, soundex, nysiis
# find where mataphone/soundex/sysiis don't match
# which algorithm results in the most reduction of unique values?

# once you have a name, calculing edit distance could be useful to identify common mispellings or mistakes
# soundex similar sounding words/?

# in baby_names_analysis, I was only looking at data from 1990 to now.
# Would including more historical data help with prediction?
```

```python
import os
os.getcwd()
```

## Combine names based on how they sound

As each entry in the babynames dataset is based on spelling, evaluate different sound based algorithms to see how to combine/reduce number of names.

```python
# load name dataset of unique names exported from R package babynames
with open('ssn_names_only.pickle', 'rb') as f:
    names = pickle.load(f)
names.shape
```

### Evaluate Jellyfish Library

```python
import jellyfish

df = names.copy()
df['metaphone'] = df.name.map(jellyfish.metaphone)
df['soundex'] = df.name.map(jellyfish.soundex)
df['nysiis'] = df.name.map(jellyfish.nysiis)
df['matchrating'] = df.name.map(jellyfish.match_rating_codex)
```

```python
df.head()
```

```python
print("Unique values:")
print("    Names: ", df.name.nunique())
print("    Metaphone: ", df.metaphone.nunique())
print("    Soundex: ", df.soundex.nunique())
print("    NYSIIS: ", df.nysiis.nunique())
print("    Match Rating Codex: ", df.matchrating.nunique())
```

```python
# need to find values that are conflated to the same Soundex, Metaphone, etc for comparison.
# ? start with most conflated?
```

```python
tf = df.copy()
```

```python
tf[['name', 'soundex']].groupby(['soundex']).agg('count').reset_index().sort_values('name', ascending=False).head()
#tf.sort_values()
```

```python
tf[tf.soundex == 'J500'].head()
```

```python
tf[['name', 'matchrating']].groupby(['matchrating']).agg('count').reset_index().sort_values('name', ascending=False).head()
```

```python
tf[tf.matchrating == 'KL'].head()
```

```python
tf[tf.name.isin(['Sofie', 'Sophie'])].head()
```

```python
df[df.metaphone == 'SF'].sample(n=10)
```

```python
df[df.soundex == 'S100'].sample(n=10)
```

```python
df[df.nysiis == 'SAFY'].sample(n=10)
```

```python
df[df.matchrating.isin(['SF', 'SPH'])].sample(n=10)
```

### Looks like the NYSIIS and Match Rating Codex algorithms give the best results here


### Evaluate the Abydos library

```python
from abydos import phonetic as ap

apdf = names.copy()
```

```python
%%timeit -n1 -r1

apdf['onca'] = apdf.name.map(ap.ONCA().encode)
```

```python
%%timeit -n1 -r1

apdf['metasoundex'] = apdf.name.map(ap.MetaSoundex().encode)
```

```python
%%timeit -n1 -r1

apdf['caverphone'] = apdf.name.map(ap.Caverphone().encode)
```

```python
%%timeit -n1 -r1

apdf['daitchmokotoff'] = apdf.name.map(ap.DaitchMokotoff().encode)
```

### Beider Morse is about 250x - 500x slower than the other algorithms, so split data set and run in multiple processes

%%timeit result for full data set if run serially
# ```
4min 1s ± 0 ns per loop (mean ± std. dev. of 1 run, 1 loop each)
# ```

Using multiprocessing, on a machine with 4 cores, almost splits time in 4:

# ```
1min 22s
# ```

```python
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
```

```python
from datetime import datetime 
start_time = datetime.now() 

apdf = parallelize(apdf, ap.BeiderMorse().encode, 'beidermorse')

time_elapsed = datetime.now() - start_time 
print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
```

```python
ap.BeiderMorse().encode('Seth', language_arg = 'english')
```

```python
%%timeit -n1 -r1

apdf['parmarkumbharana'] = apdf.name.map(ap.ParmarKumbharana().encode)
```

```python
apdf.head()
```

```python
repr(apdf['daitchmokotoff'][0])
```

```python
apdf.daitchmokotoff.map(repr).nunique()
```

```python
print("Unique values:")
print("    Names:            ", apdf.name.nunique())
print("    ONCA:             ", apdf.onca.nunique())
print("    MetaSoundex:      ", apdf.metasoundex.nunique())
print("    Caverphone:       ", apdf.caverphone.nunique())
print("    Daitchmokotoff:   ", apdf.daitchmokotoff.map(repr).nunique())
print("    Beidermorse:      ", apdf.beidermorse.nunique())
print("    ParmarKumbharana: ", apdf.parmarkumbharana.nunique())
```

```python
temp = apdf.copy()
temp['daitchmokotoff'] = temp.daitchmokotoff.map(repr)
for c in temp.columns:
    if c == 'name':
        continue
    top = temp[['name', c]].\
          groupby([c]).\
          agg('count').\
          reset_index().\
          sort_values('name', ascending=False).\
          rename(columns={'name':'count'})
    print(top.head())
    print('Sample of names matching the most common value')
    print(temp[temp[c] ==  top.head(1).values[0][0]]['name'].head())
```

```python
check = ['Sophia', 'Sofia']
#names[names.sort_values('name').name.astype(str).str[0:3] == 'Aar'
apdf[names.name.isin(check)]
```

### Now look at names resulting matching each of the different phonetic algorithms

```python
apdf[apdf.onca == 'S100'].sample(n=10)
```

```python
apdf[apdf.metasoundex == '4100'].sample(n=10)
```

```python
apdf[apdf.caverphone == 'SFA1111111'].sample(n=10)
```

```python
tdf = apdf.copy()
tdf['daitchmokotoff'] = apdf.daitchmokotoff.map(repr)
tdf[tdf.daitchmokotoff.str.contains('470000')].sample(n=10)
```

```python
checklist = apdf[apdf.name == 'Sofia'].beidermorse.values[0].split()
def check_fn(input):
    return any(x in input.split() for x in checklist)

apdf[apdf.beidermorse.map(check_fn)]
```

```python
apdf[apdf.parmarkumbharana.isin(['SF', 'SPH'])].sample(n=20)
```

### Results

This is highly subjective, but it appears that most phonetic algorithms over conflate source names. The algorithms have too many false positives and thus too many names that are not similar in pronounciation/spelling are assigned the same code.

# ```
    MetaSoundex:       1279
    ONCA:              3193
    Soundex:           3413
    Caverphone:        6083
    Daitchmokotoff:    6679
    Metaphone:        10739
    ParmarKumbharana: 13875
    NYSIIS:           16694
    Match Rating:     24109
    Beidermorse:      51501

    Total Unique Names: 77092
# ```

As a native English speaker, it appears that algorithms with more the approximately 10,000 uniques from Metaphone work the best; from Jellyfish this includes NYSIIS, and Match Rating Codex. From Abydos this includes Beider & Morse or Parmar & Kumbharana algorithms. Since Abydos includes all of the algorithms from Jellyfish as well as many others, Abydos seems the better library.

Match Rating Codex and the Parmar & Kumbharana algorithms appear to be similar; names that I think should be assigned a similar category are not. However when looking at the combine results of the multiple categories, the results appear pretty good.

Of course, if someone is a non-native English speaker in a community of non-native English speakers, different algorithms may work better.

By splitting and combining the various options provided by Beider & Morse, I think the results are the best, so will use that for additional analysis.


### Perform sample analysis to see how Beider & Morse algroithm might work

```python
names = apdf[['name', 'beidermorse']].copy()
names['bmset'] = names['beidermorse'].str.split().apply(set)
```

```python
# setup test dataset
data = [{'year': 1990.0,
  'sex': 'F',
  'name': 'Bayleigh',
  'n': 11,
  'prop': 5.36e-06},
 {'year': 1990.0,
  'sex': 'F',
  'name': 'Dyesha',
  'n': 8,
  'prop': 3.89e-06},
 {'year': 1990.0,
  'sex': 'F',
  'name': 'Latrivia',
  'n': 7,
  'prop': 3.41e-06},
 {'year': 1990.0,
  'sex': 'F',
  'name': 'Leinaala',
  'n': 9,
  'prop': 4.38e-06},
 {'year': 1990.0,
  'sex': 'F',
  'name': 'Michael',
  'n': 278,
  'prop': 0.00013535},
 {'year': 1990.0,
  'sex': 'M',
  'name': 'Cordarious',
  'n': 14,
  'prop': 6.51e-06},
 {'year': 1990.0,
  'sex': 'M',
  'name': 'Jeromy',
  'n': 115,
  'prop': 5.346e-05},
 {'year': 1990.0,
  'sex': 'M',
  'name': 'Kelcie',
  'n': 6,
  'prop': 2.79e-06},
 {'year': 1990.0,
  'sex': 'M',
  'name': 'Nelson',
  'n': 931,
  'prop': 0.00043279},
 {'year': 1990.0,
  'sex': 'M',
  'name': 'Shade',
  'n': 11,
  'prop': 5.11e-06},
 {'year': 1991.0,
  'sex': 'F',
  'name': 'Michael',
  'n': 42,
  'prop': 3e-05},
 {'year': 1990.0,
  'sex': 'F',
  'name': 'Mychael',
  'n': 42,
  'prop': 3e-05},
 {'year': 1990.0,
  'sex': 'M',
  'name': 'Mychael',
  'n': 42,
  'prop': 3e-05}]
alt = pd.DataFrame.from_dict(data)
alt.sort_values(['year', 'sex', 'name'], inplace=True)
alt.shape
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

```python

```
