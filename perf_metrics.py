"""
Atividade IV - Inteligência Artificial
Implementação dos Algoritmos K-Means e Single Link

Integrantes:    Enzo Laragnoit Fernandes        759641
                Gabriel Viana Teixeira          795465
                Guilherme Pereira Fantini       795468
"""

import os
import itertools
import pandas as pd

from sklearn.metrics import adjusted_rand_score


if __name__ == '__main__':

    reference_cluster_files = [file for file in os.listdir('./datasets/') if '.clu' in file]

    # informações usadas para construir o dataframe
    perf_info = {
        'alg': [],
        'dataset': [],
        'k': [],
        'rand': [],
    }

    # Para cada uma das partições reais, calcula o Indice de Rand Ajustado
    for reference_file in reference_cluster_files:

        dataset_prefix = reference_file.split('R')[0]

        for alg in ['kmeans', 'singlelink']:

            # encontra todos os arquivos de previsão daquele algoritmo
            alg_files = [file for file in os.listdir('./previsoes') if alg in file]

            # encontra todos os arquivos correspondentes ao 'reference_file' que são do algortimo 'alg'
            corr_pred_files = [file for file in alg_files if dataset_prefix in file]

            for file in corr_pred_files:

                real = pd.read_csv(f'./datasets/{reference_file}', sep = '\t', names = ['id', 'label'])
                pred = pd.read_csv(f'./previsoes/{file}')

                real_array = real[['label']].values.ravel()
                pred_array = pred[['label']].values.ravel()

                rand = adjusted_rand_score(real_array, pred_array)

                k_count = int(file.split(')')[0].split('(')[1])

                perf_info['alg'].append(alg)
                perf_info['dataset'].append(dataset_prefix)
                perf_info['k'].append(k_count)
                perf_info['rand'].append(round(rand, 4))

    perf_report = pd.DataFrame().from_dict(perf_info)
    perf_report.to_csv('performance_report.csv', index = False)

