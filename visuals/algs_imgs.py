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


def get_k_count(filename):
    return int(filename.split('(')[1].split(')')[0])


def load_classifications(dataset_prefix, alg) -> dict:
    """
    Obtem todos os dataframes relativos ao algoritmo e ao conjunto de dados especificados
    """

    dataframes = []
    pred_data_files = []

    for file in os.listdir('../previsoes/.'):
        if dataset_prefix in file and alg in file:
            pred_data_files.append(file)

    for file in os.listdir('../datasets/.'):
        if dataset_prefix in file and '.txt' in file:
            reference_file = file

    for file in pred_data_files:

        df_pred = pd.read_csv(f'../previsoes/{file}')
        df_ref  = pd.read_csv(f'../datasets/{reference_file}', sep = '\t')

        dataframes.append({
            'k': get_k_count(file),
            'info': pd.merge(df_pred, df_ref,
                             how = 'inner',
                             left_on = 'id',
                             right_on = 'sample_label')
        })

    return dataframes


if __name__ == '__main__':

    perf_data = pd.read_csv('../performance_report.csv')

    for alg in perf_data['alg'].unique():

        for dataset_prefix in ['c2ds1', 'c2ds3', 'monkey']:

            alg_dataset = load_classifications(dataset_prefix, alg)

            if dataset_prefix != 'monkey':

                fig, axs = plt.subplots(2, 2, figsize = (6.7, 6.7), dpi = 120)

                for i in range(2):
                    for j in range(2):

                        plot_info = alg_dataset.pop(0)

                        axs[i, j].scatter(plot_info['info']['d1'].values, plot_info['info']['d2'].values,
                                          c = plot_info['info']['label'],
                                          s = 12)

                        axs[i, j].set_title(plot_info['k'])

                fig.savefig('coisa.jpeg')

            else:

                pass

