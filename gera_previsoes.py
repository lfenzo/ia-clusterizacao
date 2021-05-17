"""
Atividade IV - Inteligência Artificial
Implementação dos Algoritmos K-Means e Single Link

Integrantes:    Enzo Laragnoit Fernandes        759641
                Gabriel Viana Teixeira          795465
                Guilherme Pereira Fantini       795468
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

from k_means import KMeansClustering as KMC
from single_link import SingleLinkClustering as SLK

def save_predictions(ids: list, pred: list, dataset: str, k: int, alg: str, dest_dir: str):
    """
    Salva as classificações realizadas.
    """

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    pred_to_save = pd.DataFrame().from_dict({
        'id': ids,
        'label': pred,
    })

    pred_to_save.to_csv(f'./{dest_dir}/{alg}_k({k})_{dataset}.csv', index = False)

if __name__ == '__main__':

    datasets = {
        'c2ds1-2sp':  pd.read_csv('datasets/c2ds1-2sp.txt', sep = '\t'),
        'c2ds3-2g':  pd.read_csv('datasets/c2ds3-2g.txt', sep = '\t'),
        'monkey': pd.read_csv('datasets/monkey.txt', sep = '\t')
    }

    #
    #
    # classificação utilizando o Kmeans
    #
    #
    km_clf = KMC(max_iter = 20)

    # K entre 2 e 5 nos dois primeiros datasets
    for key in ['c2ds1-2sp', 'c2ds3-2g']:

        for k in range(2, 6):

            km_clf.fit(data = datasets[key][['d1', 'd2']].values, k = k)
            predictions = km_clf.predict(data = datasets[key][['d1', 'd2']].values)

            save_predictions(ids = datasets[key]['sample_label'].to_list(),
                             pred = predictions,
                             dataset = key,
                             k = k,
                             alg = 'kmeans',
                             dest_dir = 'previsoes')

    # K entre 5 e 12 para o terceiro dataset
    for k in range(5, 13):

        km_clf.fit(data = datasets['monkey'][['D1', 'D2']].values, k = k)
        predictions = km_clf.predict(data = datasets['monkey'][['D1', 'D2']].values)

        save_predictions(ids = datasets['monkey']['sample_label'].to_list(),
                         pred = predictions,
                         dataset = 'monkey',
                         k = k,
                         alg = 'kmeans',
                         dest_dir = 'previsoes')

    exit()
    #
    #
    # classificação utilizando o Single Link
    #
    #
    sl_clf = SLK(verbose = True)

    # K entre 2 e 5 nos dois primeiros datasets
    for key in ['c2ds1-2sp', 'c2ds3-2g']:

        predictions = sl_clf.fit_predict(data = datasets[key][['d1', 'd2']].values,
                                         k_min = 2,
                                         k_max = 5)

        for k, pred in predictions.items():

            save_predictions(ids = datasets[key]['sample_label'].to_list(),
                             pred = pred,
                             dataset = key,
                             k = k,
                             alg = 'singlelink',
                             dest_dir = 'previsoes')


    predictions = sl_clf.fit_predict(data = datasets['monkey'][['D1', 'D2']].values,
                                     k_min = 2,
                                     k_max = 13)

    for k, pred in predictions.items():

        save_predictions(ids = datasets['monkey']['sample_label'].to_list(),
                         pred = pred,
                         dataset = key,
                         k = k,
                         alg = 'singlelink',
                         dest_dir = 'previsoes')

