import numpy as np

from utils import dist
from random import randint

class KMeansClustering(object):

    def __init__(self, data):
        self._centroids = None
        self._data = data[['d1', 'd2']].values
        self._labels = data[['sample_label']]

    def fit(self, k: int, max_iter = 20):
        """
        Utiliza os dados passados na instanciação para calcular os centroides.

        Parametros
        -------------

        `k`: número de clusters a serem formados nos dados.
        `max_iter`: número de iteração a serem realizadas no algoritmo. Default = 20.
        """

        if not k >= 2:
            raise ValueError(f'O valor de "k" deve ser maior ou igual a 2. Valor passado: {k}')

        # está iniciando um novo 'fit'
        if self._centroids != None:
            self._centroids = None

        self.get_initial_centroids(k = k, size = len(self._data))

        # inicializa os clusters (inicialmente todos estão vazios)
        cluster_objects = {c: [] for c in range(k)}

        for _ in range(max_iter):

            for obj in self._data:

                #calcula a distancia do objeto para todos os centroides
                distancias = np.array([dist(obj, c) for c in self._centroids])

                # adiciona o objeto ao centroide cuja a distancia é a menor entre todos os centroides
                cluster_objects[ distancias.argmin() ].append(obj)

            # atualiza os centroides calculando a média de cada atributos que está em cada cluster
            self.update_centroids(cluster_objects)

        return self._centroids

    def get_initial_centroids(self, k, size):
        """
        Obtém os centroides iniciais escolhendo aleatoriamente uma objeto para cada
        centroide.
        """

        initial_indexes = []

        while len(initial_indexes) < k:
            new_index = randint(0, size)
            if new_index not in initial_indexes:
                initial_indexes.append(new_index)

        self._centroids = [self._data[i] for i in initial_indexes]

    def predict(self, data):
        """
        Associa cada um dos objetos passados em `data` a um dos `k` clusters.
        Retorna um array contendo 
        """

        if self._centroids is None:
            raise AttributeError('O clusterizador ainda não foi treinado. Utilize o método "fit"')

        correspondent_cluster = []

        for obj in data:

            distancias = np.array([dist(obj, c) for c in self._centroids])
            correspondent_cluster.append( distancias.argmin() )

        return correspondent_cluster

    def get_centroids(self):
        return self._centroids

    def update_centroids(self, cluster_objects):
        for idx in cluster_objects.keys():
            self._centroids[idx] = np.mean(cluster_objects[idx], axis = 0)

    def set_initial_centroids(self, indexes):
        self._centroids = [self._data[i] for i in indexes]

    def get_data(self):
            return self._data
