"""
Atividade IV - Inteligência Artificial
Implementação dos Algoritmos K-Means e Single Link

Integrantes:    Enzo Laragnoit Fernandes        759641
                Gabriel Viana Teixeira          795465
                Guilherme Pereira Fantini       795468
"""

import os
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use('pgf')

def load_original_datasets(file_prefixes):
    """
    Para cada um dos conjuntos de referencia, obtem um dataframe com todas as informações
    """

    feature_names = ['sample_label', 'd1', 'd2']
    dataframes = []

    for prefix in file_prefixes:

        data_files = []

        for file in os.listdir('../datasets/.'):
            if prefix in file:
                data_files.append(file)

        for file in data_files:

            if '.clu' in file:
                df1 = pd.read_csv(f'../datasets/{file}', sep = '\t', names = ['sample_label', 'class'])

            elif '.txt' in file:
                df2 = pd.read_csv(f'../datasets/{file}', sep = '\t', names = feature_names, skiprows = 1)

        dataframes.append( pd.merge(df1, df2, how = 'inner', on = 'sample_label') )

    return dataframes


if __name__ == '__main__':

    reference_datasets = load_original_datasets(file_prefixes = ['c2ds1', 'c2ds3', 'monkey'])

    fig, axs = plt.subplots(1, 3, figsize = (6.7, 2.5), dpi = 200)

    titles = ['Monkey', 'Globulars', 'Spirals']

    for i, dataset in enumerate(reference_datasets):

        axs[i].scatter(dataset['d1'].values, dataset['d2'].values,
                       c = dataset['class'].values,
                       s = 5,
                       cmap = 'Set1_r')

        axs[i].set_xlabel('d1', fontsize = 8)
        axs[i].set_ylabel('d2', fontsize = 8)

        axs[i].tick_params(axis = 'both', labelsize = 5)
        axs[i].set_title(titles.pop(), fontsize = 10)

    fig.tight_layout()

    fig.savefig(f'{__file__.replace(".py", "")}.pgf', format = 'pgf', bbox_inches = 'tight')
    fig.savefig(f'{__file__.replace(".py", "")}.jpeg', bbox_inches = 'tight')
