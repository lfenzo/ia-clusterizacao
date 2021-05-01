import pandas as pd
import matplotlib.pyplot as plt

from k_means import KMeansClustering as KMC

dataset = pd.read_csv('datasets/c2ds3-2g.txt', sep = '\t')

fig, axs = plt.subplots(1, 2, figsize = (14, 7), dpi = 120)

clusterizador = KMC(data = dataset)
clusterizador.fit(k = 3, max_iter = 20)
data_to_predict = dataset.iloc[:, 1:].values
clusters = clusterizador.predict(data_to_predict)

axs[0].scatter(dataset['d1'], dataset['d2'], c = clusters)

for centroid in clusterizador.get_centroids():
    axs[0].scatter(centroid[0], centroid[1], c = 'red', s = 100)

clusterizador = KMC(data = dataset)
clusterizador.fit(k = 2, max_iter = 20)
data_to_predict = dataset.iloc[:, 1:].values
clusters = clusterizador.predict(data_to_predict)

axs[1].scatter(dataset['d1'], dataset['d2'], c = clusters)

for centroid in clusterizador.get_centroids():
    axs[1].scatter(centroid[0], centroid[1], c = 'red', s = 100)

fig.tight_layout()
fig.savefig('teste.jpeg', bbox_inches = 'tight')
