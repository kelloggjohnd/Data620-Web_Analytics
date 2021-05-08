# -*- coding: utf-8 -*-
"""
Created on Fri May  7 22:02:09 2021

@author: x
"""

import nltk
import nltk.corpus
from nltk.corpus import stopwords
nltk.download("stopwords")
from nltk.tokenize import word_tokenize
from nltk.corpus import PlaintextCorpusReader

import random
random.seed(42)

import pandas as pd
import numpy as np

import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from pyvis.network import Network

pd.set_option("display.max_rows",500)

import warnings
warnings.filterwarnings('ignore')

corpus = pd.read_csv('data/corpus.csv')
corpus.head()

speeches = pd.read_csv('data/presidential_speeches.csv')
speeches.head()

stop_words = set(stopwords.words('english'))

ob=open('data/obama1.txt', encoding="utf8")
raw_o=ob.read()
#raw_o = pd.read_csv('data/obama1.txt', header = None)
raw_o=raw_o.lower()
word_tokens = word_tokenize(raw_o)
obama = [word for word in word_tokens if word not in stop_words]
obama = [word for word in obama if word.isalpha()]

wa=open('data/washington.txt', encoding="utf8")
raw_wa=wa.read()
raw_wa=raw_wa.lower()
word_tokens = word_tokenize(raw_wa)
washington = [word for word in word_tokens if word not in stop_words]
washington = [word for word in washington if word.isalpha()]

ro=open('data/roosevelt.txt', encoding="utf8")
raw_ro=ro.read()
raw_ro=raw_ro.lower()
word_tokens = word_tokenize(raw_ro)
roosevelt = [word for word in word_tokens if word not in stop_words]
roosevelt = [word for word in roosevelt if word.isalpha()]

ke=open('data/kennedy.txt', encoding="utf8")
raw_ke=ke.read()
raw_ke=raw_ke.lower()
word_tokens = word_tokenize(raw_ke)
kennedy = [word for word in word_tokens if word not in stop_words]
kennedy = [word for word in kennedy if word.isalpha()]

cl=open('data/clinton.txt', encoding="utf8")
raw_cl=cl.read()
raw_cl=raw_cl.lower()
word_tokens = word_tokenize(raw_cl)
clinton = [word for word in word_tokens if word not in stop_words]
clinton = [word for word in clinton if word.isalpha()]

re=open('data/reagan.txt', encoding="utf8")
raw_re=re.read()
raw_re=raw_re.lower()
word_tokens = word_tokenize(raw_re)
reagan = [word for word in word_tokens if word not in stop_words]
reagan = [word for word in reagan if word.isalpha()]

obama_df = pd.DataFrame(obama, columns = ['tokens'])
obama_df['speaker'] = 'obama'
obama_df = nltk.FreqDist(w.lower() for w in obama_df)
obama_df = list(obama_df)[:200]


washington_df = pd.DataFrame(washington, columns = ['tokens'])
washington_df['speaker'] = 'washington'

roosevelt_df = pd.DataFrame(roosevelt, columns = ['tokens'])
roosevelt_df['speaker'] = 'roosevelt'

kennedy_df = pd.DataFrame(kennedy, columns = ['tokens'])
kennedy_df['speaker'] = 'kennedy'

kennedy_df = pd.DataFrame(kennedy, columns = ['tokens'])
kennedy_df['speaker'] = 'kennedy'

clinton_df = pd.DataFrame(clinton, columns = ['tokens'])
clinton_df['speaker'] = 'clinton'

reagan_df = pd.DataFrame(reagan, columns = ['tokens'])
reagan_df['speaker'] = 'reagan'

speakers = [obama_df, washington_df, kennedy_df, clinton_df, reagan_df, roosevelt_df]
speakers = pd.concat(speakers)

G=nx.from_pandas_dataframe(speakers, "tokens", "speaker")
print (nx.info(G))

degree_G = nx.degree_centrality(G)
degree_sort = sorted(degree_G.items(),
      key=lambda x:x[1], 
      reverse=True)
degree_df = pd.DataFrame(degree_sort, columns=['tokens', 'speaker'])
degree_df[degree_df.speaker.isin(speakers)]

view_pyvis = True

if view_pyvis:
    N = Network(height='1000px',
                    width='1000px',
                    bgcolor='#222222', 
                    font_color='white', 
                    notebook=True)
    N.barnes_hut()
    for n in G.nodes():
        N.add_node(n)
    for e in G.edges():
        N.add_edge(e[0], e[1])
    N.show('network_2.html')

#%matplotlib inline
plt.rcParams['figure.figsize'] = (20, 25)
nx.draw(G, node_color='bisque', with_labels=True)
