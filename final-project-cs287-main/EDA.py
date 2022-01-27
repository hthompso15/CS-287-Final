# CODE STRUCTURE
#  X. BASIC
#  |-- imports
#  |-- local variables
#  |-- helper functions
#  0. EDA for GENuS
#  1. EDA for DEMOGRAPHIC DATA
#  2. EDA for DIET DATA
#  |-- 2.1 create summary table for data.flat
#  |-- 2.2 create summary table for data.flat_sum
#  3. EDA for UNFAO-IDDS
#  4. EDA for FlavorDB

import pandas as pd
import os
from datacleaning import data
import math
import numpy as np

### local variables ###
directory = "clean2.0"
destination = "stats"
if not os.path.exists(destination):
    os.mkdir(destination)

### helper functions ###
def make_dict(lst1, lst2):
    return {lst1[i]: lst2[i] for i in range(len(lst1))}

def float_to_int(x):
    y = int(x)
    return y

def drop_water(water_list, dataframe):
    print("\ndropping water...")

    for wa in water_list:
        try:
            dataframe = dataframe.drop(columns=wa, inplace=False)
        except:
            print(wa, "not in the dataframe")
    return dataframe

def shannon_weiner(food_arr):
    tot = 0
    #print(food_arr)
    sums = np.nansum(food_arr)
    nans = 0
    for i in range(0,len(food_arr)):
        if(np.isnan(food_arr[i])):
            nans += 1
            continue
        w = float(food_arr[i])/sums
        if(w != 0):
            tot += float(w)*float(math.log(w,2))
        #print(tot)
    return (-1*tot)/math.log(len(food_arr)-nans,2)

def berry_index(food_arr):
    tot = 0
    #print(food_arr)
    sums = np.nansum(food_arr)
    for i in range(0,len(food_arr)):
        if(np.isnan(food_arr[i])):
            continue
        w = food_arr[i]/sums
        tot += math.pow(w,2)
    return 1-tot

def count_row(food_arr):
    return np.sum(food_arr)

### 0. EDA for GENuS
# PLOTTING IDEAS for data.GENuS_USA
# line charts of selected crops?

filename = "data.GENuS_USA.csv"
shannon_scores = []
berry_scores = []

genus_USA = pd.read_csv(os.path.join(directory, filename), index_col=False)
genus_USA = genus_USA.iloc[:,1:-1]
vals = genus_USA.values
years = vals[:,0]

# print("Years")
# print(years)

for row in vals:
    food_arr = row[2:]
    #shannon_scores.append(entropy(food_arr[~np.isnan(food_arr)]))
    shannon_scores.append(shannon_weiner(food_arr))
    berry_scores.append(berry_index(food_arr))
#
# print("\nShannon")
# print(shannon_scores)
# print("\nBerry")
# print(berry_scores)

crop_diversity = pd.DataFrame(
    {'Year': years,
     'Shannon': shannon_scores,
     'Berry': berry_scores
    })

crop_diversity.to_csv(os.path.join(destination,'crop_diversity.csv'))
print('crop_diversity.csv -- saved!')

### 1. EDA for DEMOGRAPHIC DATA
# PLOTTING IDEAS for data.demographics
# stat summary tables?

print("START: demographic data EDA...")
# 'SEQN' unique ids
# 'SDDSRVYR' tells you the wave of the data.
"""
2, NHANES 2001-2002 Public Release, 11039
3, NHANES 2003-2004 Public Release, 10122
4, NHANES 2005-2006 Public Release, 10348
5, NHANES 2007-2008 Public Release, 10149
6, NHANES 2009-2010 Public Release, 10537
7, NHANES 2011-2012 public release, 9756
8, NHANES 2013-2014 public release, 10175
9, NHANES 2015-2016 public release, 9971 
10, NHANES 2017-2018 public release	, 9254
"""
# 'RIAGENDR' 1:male; 2: female
# 'RIDAGEYR' 0-79, 80 (80+)
# 'INDFMPIR' family income / poverty line, 0-5

var_of_interest = ['SEQN', 'SDDSRVYR', 'RIAGENDR', 'RIDAGEYR', 'INDFMPIR', ]

# adults
data.demographics = pd.read_csv(os.path.join(directory, "data.demographics.csv"), index_col=0)
age20plus = data.demographics[data.demographics['RIDAGEYR'] >= 20.0]
#len(age20plus) n = 44,790
age20plus = age20plus.filter(var_of_interest)
#age20plus['SEQN'] = age20plus['SEQN'].apply(float_to_string)

# create a col of ids that are age >= 20
age20plus_ids = age20plus['SEQN']
#print(age20plus.describe())

# non-infants
age2above = data.demographics[data.demographics['RIDAGEYR'] > 2.0]
#len(age2above) # n = 72,076
age2above = age2above.filter(var_of_interest)
#age2above['SEQN'] = age2above['SEQN'].apply(float_to_string)

# create a col of ids that are age > 2
age2above_ids = age2above['SEQN']
#print(age2above.describe())

print("FINISH: demographic data EDA...")

### 2. EDA for DIET DATA
print("START: diet data EDA...")

# PLOTTING IDEAS for data.flat data
# dietary diversity distribution of food from home vs. away from home
# dietary diversity distribution of weekday vs weekends Note: 24-hour dietary recall is in the same day

# PLOTTING IDEAS for data.flat_sum
# a typical american diet -- plot median of each commodity using waffle chart
# Note: Drop all kinds of water

"""
86,86A,8601000000,"Water, direct, all sources"
86,86A,8601100000,"Water, direct, tap"
86,86A,8601200000,"Water, direct, bottled"
86,86A,8601300000,"Water, direct, other"
86,86A,8601400000,"Water, direct, source-NS"
86,86B,8602000000,"Water, indirect, all sources"
86,86B,8602100000,"Water, indirect, tap"
86,86B,8602200000,"Water, indirect, bottled"
86,86B,8602300000,"Water, indirect, other"
86,86B,8602400000,"Water, indirect, source-NS"
"""

water = ['8601000000', '8601100000', '8601200000', '8601300000', '8601400000', '8602000000', '8602100000', '8602200000',
         '8602300000', '8602400000']

# VAR Keys
# 'days':  Intake day of the week (1-7)
# 'home?': Did you eat this meal at home? {1: yes, 2:no, 7:refused, 9:don't know}

## 2.1 create summary table for data.flat
data.flat = pd.read_csv(os.path.join(directory, "data.flat.csv"), index_col=0)
data.flat = drop_water(water, data.flat)

# change age20plus_ids ->  age2above_ids for diff pop
old_enough = data.flat['SEQN'].isin(age20plus_ids)
flat_age = data.flat[old_enough]

flat_age_stats = flat_age.describe()
flat_age_stats = flat_age_stats.rename(columns=data.fcidcode_to_fcidname) # covert FCID code to FCID name
print(flat_age_stats.sort_values(by='count', axis=1, ascending=False)) # sort the summary table


## 2.2 create summary table for data.flat_sum
data.flat_sum = pd.read_csv(os.path.join(directory,"data.flat_sum.csv"), index_col=0)
data.flat_sum = drop_water(water, data.flat_sum)
# print(len(data.flat_sum)) # n = 80597

# find matches between diet and demo
# n = 63,414
len(set(age2above['SEQN']).intersection(set(data.flat_sum.index)))
# n = 39,755
print('\nThe number should be 39755: ', len(set(age20plus['SEQN']).intersection(set(data.flat_sum.index))))

# change age20plus_ids ->  age2above_ids for diff pop
old_enough2 = data.flat_sum.index.isin(age20plus_ids)
flat_sum_age = data.flat_sum[old_enough2]

flat_sum_age_stats = flat_sum_age.describe()
flat_sum_age_stats = flat_sum_age_stats.rename(columns=data.fcidcode_to_fcidname) # covert FCID code to FCID name
print(flat_sum_age_stats.sort_values(by='50%', axis=1, ascending=False)) # sort the summary table
print("FINISH: diet data EDA...")

print("START: Diet Diversity calc...")
shannon_diet = []
berry_diet = []

vals = flat_sum_age.values
SEQN = flat_sum_age.index

for row in vals:
    shannon_diet.append(shannon_weiner(row))
    berry_diet.append(berry_index(row))

### 3. EDA for UNFAO-IDDS
# PLOTTING IDEAS for data.flat_sum_10
# CDF/PDF/Hist
data.flat_10 = pd.read_csv(os.path.join(directory,"data.flat_10.csv"), index_col=0)
flat_10_age = data.flat_10[old_enough2]

shannon_10 = []
berry_10 = []

vals = flat_10_age.values
#SEQN = flat_10_age.index

for row in vals:
    shannon_10.append(shannon_weiner(row))
    berry_10.append(berry_index(row))

flat_10_1s0s = flat_10_age.astype(bool).astype(int)
IDDS = flat_10_1s0s.sum(axis=1)
print("END: Diet Diversity calc...")

print("START: Flavor Diversity calc...")

### 4. EDA for FlavorDB
# PLOTTING IDEAS for data.flat_flavor
# Heat map

data.flat_flavor = pd.read_csv(os.path.join(directory,"data.flat_flavor.csv"), index_col=0)
flat_flavor_age = data.flat_flavor[old_enough2]

shannon_flavor = []
berry_flavor = []

vals = flat_flavor_age.values

for row in vals:
    #food_arr = row[2:]
    shannon_flavor.append(shannon_weiner(row))
    berry_flavor.append(berry_index(row))
print("END: Flavor Diversity calc...")

diet_diversity = pd.DataFrame(
    {
     'Shannon_diet': shannon_diet,
     'Berry_diet': berry_diet,
     'Shannon_10': shannon_10,
     'Berry_10': berry_10,
     'IDDS_10' : IDDS,
     'Shannon_flavor': shannon_flavor,
     'Berry_flavor': berry_flavor
    })
diet_diversity.set_index(SEQN)

diet_diversity.to_csv(os.path.join(destination,'diet_diversity.csv'))
print('diet_diversity.csv -- saved!')

age20plus['SEQN'] = age20plus['SEQN'].apply(float_to_int)
age20plus = age20plus.set_index('SEQN')
age20plus.to_csv(os.path.join(destination,'demo_age20plus.csv'))
print('demo_age20plus.csv -- saved!')

diet_demo_age20plus_merged = diet_diversity.merge(age20plus, left_index=True, right_index=True)

shannon_of_the_year = make_dict(crop_diversity['Year'], crop_diversity['Shannon'])
berry_of_the_year = make_dict(crop_diversity['Year'], crop_diversity['Berry'])

wave_to_year ={2: 2001, 3: 2003, 4: 2005, 5: 2007, 6: 2009, 7: 2011, 8: 2013, 9: 2015, 10: 2017}
diet_demo_age20plus_merged['SDDSRVYR'] = diet_demo_age20plus_merged['SDDSRVYR'].map(wave_to_year)
diet_demo_age20plus_merged['shannon_crop_6yr_prior'] = diet_demo_age20plus_merged['SDDSRVYR'] - 7
diet_demo_age20plus_merged['shannon_crop_6yr_prior'] = diet_demo_age20plus_merged['shannon_crop_6yr_prior'].map(shannon_of_the_year)
diet_demo_age20plus_merged['berry_crop_6yr_prior'] = diet_demo_age20plus_merged['SDDSRVYR'] - 7
diet_demo_age20plus_merged['berry_crop_6yr_prior'] = diet_demo_age20plus_merged['berry_crop_6yr_prior'].map(berry_of_the_year)

diet_demo_age20plus_merged.to_csv(os.path.join(destination,'diet_demo_age20plus_merged.csv'))
print('diet_demo_age20plus_merged.csv -- saved!')

# EDA plot
import seaborn as sns
import matplotlib.pyplot as plt
os.system('pip install pywaffle')
from pywaffle import Waffle

destination = 'plot'
if not os.path.exists(destination):
    os.mkdir(destination)

# LINE CHART
genus_USA  = genus_USA.drop(['Area'], axis=1)
genus_pivot = genus_USA.set_index('Year')
genus_pivot_sorted = genus_pivot.sort_values(by=2011, axis=1, ascending=False)
genus_pivot_sorted_first20 = genus_pivot_sorted.iloc[:, : 20]

plt.figure(figsize=(10, 6))
lineplot = sns.lineplot(data=genus_pivot_sorted_first20)
lineplot.legend(loc='center left', bbox_to_anchor=(1.0, 0.5), ncol=1)
lineplot.set(
       ylabel='per Person / Day / Gram',
       title='Top 20 US Food Supply Trend')
plt.savefig(os.path.join(destination,'lineplot.png'), dpi=300, bbox_inches='tight')

# Line plot for crop diversity
sns.set_style("whitegrid")
fig,axs = plt.subplots(1,2,figsize=(10.5,4))
sns.lineplot(data=crop_diversity.set_index('Year')['Shannon'],color='red', dashes = False, ax=axs[0])
sns.set_palette("husl")
sns.lineplot(data=crop_diversity.set_index('Year')['Berry'], color='blue', dashes = False, ax=axs[1])
plt.savefig(os.path.join(destination,'crop_lineplot.png'), dpi=300, bbox_inches='tight')

# Waffle chart
flat_sum_age_stats_sorted = flat_sum_age_stats.sort_values(by='50%', axis=1, ascending=False)
flat_sum_age_stats_sorted_tP = flat_sum_age_stats_sorted.transpose()
df_50 = flat_sum_age_stats_sorted_tP['50%']
df_50_dict = df_50.to_dict()
labels=[f"{k} ({int(v / sum(df_50_dict.values()) * 100)}%)" for k, v in df_50_dict.items()]

fig = plt.figure(
    FigureClass=Waffle,
    values=df_50[:10],
    title={
        'label': 'Median American Diet 2001-2018 ',
        'loc': 'left',
        'fontdict': {'fontsize': 26}
    },
    rows=9,
    figsize=(20, 30),
    labels=labels[:10],
    legend={
        'loc': 'lower left',
        'bbox_to_anchor': (0, -0.2),
        'ncol':5,
        'framealpha': 0,
        'fontsize': 18
    },
    cmap_name="tab20"
)
plt.savefig(os.path.join(destination,'waffle.png'), dpi=300, bbox_inches='tight')
