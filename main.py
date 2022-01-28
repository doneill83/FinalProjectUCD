from typing import Any
## import packages
import pandas as pd
import os
from datetime import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

"""To suppress all the warnings"""
import warnings

from pandas import DataFrame
from pandas.io.parsers import TextFileReader

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

