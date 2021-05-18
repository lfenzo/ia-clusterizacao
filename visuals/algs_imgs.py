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


def load_classifications(dataset_prefix, alg, perf_dataset) -> dict:
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
        df_ref  = pd.read_csv(f'../datasets/{reference_file}',
                              sep = '\t',
                              names = ['sample_label', 'd1', 'd2'],
                              skiprows = 1)

        selected_perf_row = perf_dataset[ (perf_dataset['k'] == get_k_count(file)) &
                                          (perf_dataset['alg'] == alg) &
                                          (perf_dataset['dataset'] == dataset_prefix) ]

        dataframes.append({
            'k': get_k_count(file),
            'perf': selected_perf_row.rand.values,
            'info': pd.merge(df_pred, df_ref,
                             how = 'inner',
                             left_on = 'id',
                             right_on = 'sample_label')
        })

    return sorted(dataframes, key = lambda item: item['k'])


if __name__ == '__main__':

    perf_data = pd.read_csv('../performance_report.csv')

    for alg in perf_data['alg'].unique():

        for dataset_prefix in ['c2ds1-2sp', 'c2ds3-2g', 'monkey']:

            alg_dataset = load_classifications(dataset_prefix, alg, perf_data)
            n_rows, n_cols, aspect = (2, 2, (6.7, 6.7)) if dataset_prefix != 'monkey' else (3, 3, (6.7, 5.5))

            fig, axs = plt.subplots(n_rows, n_cols, figsize = aspect, dpi = 120)

            for i in range(n_rows):
                for j in range(n_cols):

                    try:

                        plot_info = alg_dataset.pop(0)

                        axs[i, j].scatter(plot_info['info']['d1'].values, plot_info['info']['d2'].values,
                                          c = plot_info['info']['label'],
                                          cmap = 'plasma',
                                          s = 5)

                        axs[i, j].set_title(f'$k = {plot_info["k"]}$ IRA: {plot_info["perf"][0]}', fontsize = 10)

                        axs[i, j].tick_params(axis = 'x', which = 'both', bottom = False,
                                    top = False, labelbottom = False)

                        axs[i, j].tick_params(axis = 'y', which = 'both', right = False,
                                    left = False, labelleft = False)

                    except IndexError as e:

                        axs[i, j].axis('off')

            fig.suptitle(f'{alg.title()} no Conjunto {dataset_prefix.title()}')
            fig.tight_layout()

            fig.savefig(f'{alg}_{dataset_prefix}.jpeg', bbox_inches = 'tight')
