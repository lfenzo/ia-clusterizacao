"""
Atividade IV - Inteligência Artificial
Implementação dos Algoritmos K-Means e Single Link

Integrantes:    Enzo Laragnoit Fernandes        759641
                Gabriel Viana Teixeira          795465
                Guilherme Pereira Fantini       795468
"""

import os
import pandas as pd

from sklearn.metrics import adjusted_rand_score

if __name__ == '__main__':

    reference_cluster_files = [file for file in os.listdir('./datasets/') if '.clu' in file]

    kmeans_pred_files     = [file for file in os.listdir('./previsoes') if 'kmeans'     in file]
    singlelink_pred_files = [file for file in os.listdir('./previsoes') if 'singlelink' in file]

    # Para cada uma das partições reais, calcula o Indice de Rand Ajustado
    for reference_file in reference_cluster_files:


        dataset_prefix = reference_file.split('R')[0]


        """ Obtem todas as classificações feitas com o algoritmo KMeans """
        corr_pred_files = [file for file in kmeans_pred_files if dataset_prefix in file]

        for file in corr_pred_files:

            real = pd.read_csv(f'./datasets/{reference_file}', sep = '\t', names = ['id', 'label'])
            pred = pd.read_csv(f'./previsoes/{file}')

            real_array = real[['label']].values.ravel()
            pred_array = pred[['label']].values.ravel()

            ajd_rand_score = adjusted_rand_score(pred_array, real_array)

            k_count = int(file.split(')')[0].split('(')[1])


        """ Obtem todas as classificações feitas com o algoritmo KMeans """
        corr_pred_files = [file for file in singlelink_pred_files if dataset_prefix in file]

        for file in corr_pred_files:

            real = pd.read_csv(f'./datasets/{reference_file}', sep = '\t', names = ['id', 'label'])
            pred = pd.read_csv(f'./previsoes/{file}')

            real_array = real[['label']].values.ravel()
            pred_array = pred[['label']].values.ravel()

            ajd_rand_score = adjusted_rand_score(pred_array, real_array)

            k_count = int(file.split(')')[0].split('(')[1])
