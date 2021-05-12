import random
import numpy as np

from utils import dist

class KMeansClustering(object):

    def __init__(self, max_iter = 20):
        self._centroids = None
        self._max_iter = max_iter

    def fit(self, k: int, data = None):
        """
        Treina o clusterizador com os dados passados em 'data'

        Parametros
        -------------

        `k`: número de clusters a serem formados nos dados.
        `data`: conjunot de dados utilizado no treinamento modelo.
        """

        if data is None:
            raise ValueError('Não foi passado um conjunto de dados para o treinamento')

        if not k >= 2:
            raise ValueError(f'O valor de "k" deve ser maior ou igual a 2. Valor passado: {k}')

        # está iniciando um novo 'fit'
        if self._centroids != None:
            self._centroids = None

        self.__set_initial_centroids(k = k, data = data)

        # inicializa os clusters (inicialmente todos estão vazios)
        cluster_objects = {c: [] for c in range(k)}

        for _ in range(self._max_iter):

            for obj in data:

                #calcula a distancia do objeto para todos os centroides
                distancias = np.array([dist(obj, c) for c in self._centroids])

                # adiciona o objeto ao centroide cuja a distancia é a menor entre todos os centroides
                cluster_objects[ distancias.argmin() ].append(obj)

            # atualiza os centroides calculando a média de cada atributos que está em cada cluster
            self.__update_centroids(cluster_objects)

        return self._centroids

    def __set_initial_centroids(self, k, data):
        """
        Obtém os centroides iniciais escolhendo aleatoriamente uma objeto para cada
        centroide.
        """

        self._centroids = [data[i] for i in random.sample(range(0, len(data) - 1), k)]

    def predict(self, data):
        """
        Associa cada um dos objetos passados em `data` a um dos `k` clusters.

        Retorna um array contendo ´len(data)` elementos indicando a classe de
        cada um dos objetos.
        """

        if self._centroids is None:
            raise AttributeError('O clusterizador ainda não foi treinado. Utilize o método "fit"')

        correspondent_cluster = []

        for obj in data:

            distancias = np.array([dist(obj, c) for c in self._centroids])
            correspondent_cluster.append( distancias.argmin() )

        return correspondent_cluster

    def __update_centroids(self, cluster_objects):
        """
        Atualiza os centroides calculando a média de cada um dos atributos
        para todos os elementos que estão em cada cluster.
        """

        for idx in cluster_objects.keys():
            self._centroids[idx] = np.mean(cluster_objects[idx], axis = 0)

    def get_centroids(self):
        """
        Returna os centroides encontrados para o conjunto e dados passad no método
        'fit' do objeto.

        Returna
        -----------
        `_centroids`: lista com os valores relativos a cada um dos atributos de cada centroides.
        """

        return self._centroids
