import numpy as np

from utils import dist

class SingleLinkClustering(object):

    def __init__(self):
        ...

    def fit_predict(self, k: int, data = None):
        """
        Treina o clusterizador com os dados passados em 'data' e já realiza a classificação.

        Parametros
        -------------

        `k`: número de clusters a serem formados nos dados.
        `data`: conjunot de dados utilizado no treinamento modelo.
        """

        if data is None:
            raise ValueError('Não foi passado um conjunto de dados para o treinamento')

        if not k >= 2:
            raise ValueError(f'O valor de "k" deve ser maior ou igual a 2. Valor passado: {k}')

        clusters = { idx: [obj] for idx, obj in enumerate(data) }

        # realiza o agrupamento dos objetos no número de clusters desejados
        while len(clusters) > k:

            # distancia dos clusters, todos para todos
            distances = self.__cluster_distances(clusters)

            # obtem quais são os clusters mais proximos um do outro considerando a distancia single-link
            idx_from, idx_to = self.__closest_clusters(distances)

            # atualizar os clusters juntando os dois clusterws mais proximos
            self.__update_clusters(idx_from, idx_to, clusters)

    def __predict(self, data, clusters):

        correspondent_cluster = []

        for obj in data:
            for key, values in clusters.items():
                if obj in values[0]:
                    correspondent_cluster.append(key)

        return correspondent_cluster


    def __update_clusters(self, merge_from, merge_to, clusters):
        """
        Realiza a união entre os objetos de um cluster com o outro

        """

        return clusters[merge_from].extend( clusters.pop(merge_to) )


    def __closest_clusters(self, distances: dict) -> (int, int):
        """
        Obtem os indices dos clusters que serão juntados com base nas distancias

        Retorna
        ---------
        `(a, b)`: indices dos clusters a serem agrupados
        """

        menor_por_cluster = {}

        # obtem a menor distancia do cluster para todos os outros clusters
        for i, clu in enumerate(distances.keys()):

            minimo = min(zip(distances[clu].values(), distances[clu].keys()))

            menor_por_cluster[i] = {
                'from': clu,
                'to':   minimo[1],
                'dist': minimo[0],
            }

        smallest = np.inf

        # obtem a menor distancia de todas e os clusters corresopndentes à essa distancia
        for key in menor_por_cluster.keys():

            if menor_por_cluster[key]['dist'] < smallest:

                smallest = menor_por_cluster[key]['dist']
                merge_from = menor_por_cluster[key]['from']
                merge_to = menor_por_cluster[key]['to']

        return (merge_from, merge_to)

    def __cluster_distances(self, clusters) -> dict:
        """
        Obtem as distancias "todos para todos" entre os clusters.
        Não calcula a distancia de um cluster para ele mesmo.

        Parametros
        ----------
        `clusters`: dicionário contendo os clusters con seus objetos

        Retorna
        -----------
        `all_distances`: dicionário com os distancias entre cada um dos clusters.
        """

        all_distances = {}

        for c1 in clusters.keys():

            c1_to_others = {}

            for c2 in [key for key in clusters.keys() if key != c1]:
                c1_to_others[c2] = self.__singlelink_dist(clusters[c1], clusters[c2])

            all_distances[c1] = c1_to_others

        return all_distances

    def __singlelink_dist(self, c1: list, c2: list) -> float:
        """ Encontra a menor distancia entre dois clusters cada qual com pelo menos um objeto.

        Paremetros
        ------------
        `c1` e `c2`; listas com os objetos de cada um dos objetos dos clusters.

        Returna
        ----------
        `dist`: menor distancia entre um objeto de c1 e um objeto de c2.
        """

        distances = []

        for obj_c1 in c1:
            for obj_c2 in c2:
                distances.append( dist(obj_c1, obj_c2) )

        return min(distances)
