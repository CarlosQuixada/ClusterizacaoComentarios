# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib.pyplot as plt
from PreProcessamento import PreProcesso
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

import sys
reload(sys)
sys.setdefaultencoding('utf-8')


#dataPath_treino = 'C:/Users/Carlos/PycharmProjects/ClassificadorOneClassRambo/arquivoTreino/store_reviews.csv'
dataPath_treino = 'C:/Users/Carlos/PycharmProjects/ClassificadorOneClassRambo/arquivoTreino/teste.csv'

#dataPath_teste = 'C:/Users/Carlos/PycharmProjects/ClassificadorOneClassRambo/arquivoTreino/product_reviews.csv'
p = PreProcesso()

def getDadosTreino():
    print('========= Buscando Dados Treino =============')
    Dados = []
    with open(dataPath_treino, 'rb') as file:
        reader = csv.reader(file)
        for row in reader:
            Dados.append(row[0])
    print('=========== DONE =============================')
    return Dados

def preProcessarCsv(frase):
    frase = p.prePorcessar(frase)
    return frase

def criarMatrizTreino():
    print('========= Criando Matriz de Treino =============')
    data = getDadosTreino()
    treino = []
    for frase in data:
        treino.append(preProcessarCsv(frase))
    print('=========== DONE =============================')
    return treino

def criarMatrizTeste():
    print('========= Criando Matriz de Teste =============')
    data = getDadosTeste()
    teste = []
    for frase in data:
        teste.append(preProcessarCsv(frase))
    print('=========== DONE =============================')
    return teste

def treinarClassificador(teste):
    treino = criarMatrizTreino()

    pt_stop_words = set(stopwords.words('portuguese'))
    pt_stop_words.add('pra')
    pt_stop_words.add('para')

    vectorizer = TfidfVectorizer(max_df=0.75, max_features=5000, lowercase=False, min_df=2,stop_words=pt_stop_words,
                                 ngram_range=(1, 4),
                                 use_idf=True)
    data_train = vectorizer.fit_transform(treino)
    print(data_train)
    for i in data_train:
        print(i)
    return data_train

def clusterizar(data):
    print('========== Clusterização ==========')
    db = DBSCAN(eps=0.001,min_samples=3).fit(data)
    core_samples_mask = np.zeros_like(db.labels_,dtype=bool)
    core_samples_mask[db.core_sample_indices_]= True
    labels = db.labels_
    print(labels)
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print('========== Clusterização Done ==========')
    print('Número estimado de clusters: %d' % n_clusters_)

treino = criarMatrizTreino()

data = treinarClassificador(treino)

clusterizar(data)
