# -*- coding: utf-8 -*-
"""
Created on Sat Feb 27 11:24:44 2021

@author: x
"""

import pandas as pd
import pyreadstat
import networkx as nx
import matplotlib.pyplot as plt
from operator import itemgetter

df=pd.read_csv('./data/global_terror_data.csv')
df.describe()

terror = df[["country_txt", "attacktype1", "targtype1"]]

G=nx.from_pandas_dataframe(terror, "country", "attack", "target")