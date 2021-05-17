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

    for file in reference_cluster_files:
        real = pd.read_csv(f'datasets/{file}', sep = '\t', names = ['id', 'label'])

        print(real.head())


