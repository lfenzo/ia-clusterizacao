import pandas as pd
import matplotlib.pyplot as plt

from k_means import KMeansClustering as KMC

dataset = pd.read_csv('datasets/c2ds3-2g.txt', sep = '\t')
train_data = dataset[['d1', 'd2']].values

fig, axs = plt.subplots(figsize = (14, 7), dpi = 120)

clusterizador = KMC()

clusterizador.fit(k = 4, data = train_data)

data_to_predict = dataset.iloc[:, 1:].values
clusters = clusterizador.predict(data_to_predict)

axs.scatter(dataset['d1'], dataset['d2'], c = clusters)

for centroid in clusterizador.get_centroids():
    axs.scatter(centroid[0], centroid[1], c = 'red', s = 100)

fig.savefig('teste.jpeg', bbox_inches = 'tight')
