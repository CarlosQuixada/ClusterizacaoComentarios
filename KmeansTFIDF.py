from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics
import csv
from sklearn.cluster import KMeans, MiniBatchKMeans
from nltk.corpus import stopwords
import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np
dataPath_treino = 'C:/Users/Carlos/PycharmProjects/ClassificadorOneClassRambo/arquivoTreino/store_reviews.csv'
dataPath_teste = 'C:/Users/Carlos/PycharmProjects/ClassificadorOneClassRambo/arquivoTreino/product_reviews.csv'
logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')

categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
def getBase():
    Dados = []
    print('========= Buscando Dados Treino =============')
    with open(dataPath_treino, 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
            Dados.append(row[0])
    print('=========== DONE =============================')

    #print('========= Buscando Dados Teste =============')
    #with open(dataPath_teste, 'rb') as file:
    #    reader = csv.reader(file)
    #    for row in reader:
    #        Dados.append(row[0])
    #print('=========== DONE =============================')

    return Dados

print("Loading 2 documentos:")
print(categories)

dataset = getBase()

print("%d documents" % len(dataset))
print()

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
pt_stop_words = set(stopwords.words('portuguese'))
pt_stop_words.add('pra')
pt_stop_words.add('para')
vectorizer = TfidfVectorizer(max_df=0.75, max_features=5000,min_df=2, stop_words=pt_stop_words,ngram_range=(1, 4))
X = vectorizer.fit_transform(dataset)
print("done in %fs" % (time() - t0))
print("n_samples: %d, n_features: %d" % X.shape)
print()
n_cluster= 4
km = KMeans(n_clusters=n_cluster, init='k-means++', max_iter=100, n_init=1)

print("Clustering sparse data with %s" % km)
t0 = time()
km.fit(X)
print("done in %0.3fs" % (time() - t0))
print()
print("Top terms per cluster:")

order_centroids = km.cluster_centers_.argsort()[:, ::-1]

terms = vectorizer.get_feature_names()
for i in range(n_cluster):
    print("Cluster %d:" % i, end='')
    for ind in order_centroids[i, :10]:
        print(' %s' % terms[ind], end='')
    print()