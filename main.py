from typing import Any
## import packages
import pandas as pd
import os
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import networkx as nx
import plotly.express as px
from sklearn.manifold import TSNE
import plotly.graph_objects as go
from lightgbm import LGBMClassifier
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
from plotly.subplots import make_subplots
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.metrics import roc_curve, precision_recall_curve

"""To suppress all the warnings"""
import warnings

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

import matplotlib.pyplot as plt
def plot_correlation(df, cmap='RdBu_r'):
    size = len(df.columns)
    fig, ax = plt.subplots(figsize=(1.3 * size, 1. * size))
    corr = df.corr()

    im = ax.matshow(corr, cmap=cmap)
    for (i, j), z in np.ndenumerate(corr):
        ax.text(j, i, '{:0.2f}'.format(z), ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))

    plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=90)
    plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
    cb = plt.colorbar(im)
    ax.tick_params(labelsize=14)
    plt.show()

warnings.filterwarnings("ignore")
"""Check the directory"""
print(os.getcwd())
"""Import the required csv file"""
df: TextFileReader | DataFrame | Any = pd.read_csv(r'depression.csv', error_bad_lines = False, warn_bad_lines = False, sep=r'	')
"""Print the uploaded file"""
print(df)
"""Check the head and tail of the file"""
print(df.head())
print(df.tail())
"""Basic descriptives of the file"""
print(df.describe())
"""Summary of the dataset"""
print(df.info())
"""Check the full columns and rows"""
print(df.head(0))
print(df.head(1))
"""Search for nulls"""
print(df.isnull().sum())

df = df[ df['testelapse'] <= df['testelapse'].quantile(0.975) ]
df = df[ df['testelapse'] >= df['testelapse'].quantile(0.025) ]
df = df[ df['surveyelapse'] <= df['surveyelapse'].quantile(0.975) ]
df = df[ df['surveyelapse'] >= df['surveyelapse'].quantile(0.025) ]

DASS_keys = {'Depression': [3, 5, 10, 13, 16, 17, 21, 24, 26, 31, 34, 37, 38, 42],
             'Anxiety': [2, 4, 7, 9, 15, 19, 20, 23, 25, 28, 30, 36, 40, 41],
             'Stress': [1, 6, 8, 11, 12, 14, 18, 22, 27, 29, 32, 33, 35, 39]}

DASS_bins = {'Depression': [(0, 10), (10, 14), (14, 21), (21, 28)],
             'Anxiety': [(0, 8), (8, 10), (10, 15), (15, 20)],
             'Stress': [(0, 15), (15, 19), (19, 26), (26, 34)]}

for name, keys in DASS_keys.items():
    # Subtract one to match definition of DASS score in source
    df[name] = (df.filter(regex='Q(%s)A' % '|'.join(map(str, keys))) - 1).sum(axis=1)

    bins = DASS_bins[name]
    bins.append((DASS_bins[name][-1][-1], df[name].max() + 1))
    bins = pd.IntervalIndex.from_tuples(bins, closed='left')
    df[name + '_cat'] = np.arange(len(bins))[pd.cut(df[name], bins=bins).cat.codes]

dass = df[DASS_keys.keys()]
dass_cat = df[[k + '_cat' for k in DASS_keys.keys()]]

df[[k + '_cat' for k in DASS_keys.keys()] + list(DASS_keys.keys())].head()

# Add personality types to data
personality_types = ['Extraversion', 'Agreeableness', 'Conscientiousness', 'EmotionalStability', 'Openness']

# Invert some entries
tipi = df.filter(regex='TIPI\d+').copy()
tipi_inv = tipi.filter(regex='TIPI(2|4|6|8|10)').apply(lambda d: 7 - d)
tipi[tipi.columns.intersection(tipi_inv.columns)] = tipi_inv

# Calculate scores
for idx, pt in enumerate( personality_types ):
    df[pt] = tipi[['TIPI{}'.format(idx + 1), 'TIPI{}'.format(6 + idx)]].mean(axis=1)

personalities = df[personality_types]

character = pd.concat([dass, personalities], axis=1)
plot_correlation(character, cmap='viridis')

print(DASS_keys)

age_group = [
    'below 20',
    '20 to 24',
    '25 to 29',
    '30 to 34',
    '35 to 39',
    '40 to 49',
    '50 to 59',
    'above 60',
]

def label_age(row):
    if row['age'] < 20:
        return age_group[0]
    elif row['age'] < 25:
        return age_group[1]
    elif row['age'] < 30:
        return age_group[2]
    elif row['age'] < 35:
        return age_group[3]
    elif row['age'] < 40:
        return age_group[4]
    elif row['age'] < 50:
        return age_group[5]
    elif row['age'] < 60:
        return age_group[6]
    elif row['age'] > 60:
        return age_group[7]

df['age_group'] = df.apply(lambda row: label_age(row), axis=1)
print(df.head(2))

def make_pie_chart(data, series, title):
    temp_series = data[ series ].value_counts()
        # what we want to show in our charts

    labels = ( np.array(temp_series.index) )
    sizes = ( np.array( ( temp_series / temp_series.sum() ) *100) )

    trace = go.Pie(labels=labels,
                   values=sizes)
    layout= go.Layout(
        title= title,
        title_font_size= 24,
        title_font_color= 'red',
        title_x= 0.45,
    )
    fig = go.Figure(data= [trace],
                    layout=layout)
    fig.show()

make_pie_chart(df, 'age_group', 'Distribution by Age')

temp = df.copy()
temp['gender'].replace({
    1: "Male",
    2: "Female",
    3: "Non-binary",
    0: "Unanswered",
},
    inplace=True)

make_pie_chart(temp, 'gender', 'Distribution by Gender')