import numpy as np
from sklearn import datasets

def kmeans(dataset, clusters, threshold):
	#pegando #clusters ids de dataset.
	ids = np.random.randint(low=0, high=dataset.shape[0]-1, size=clusters)
	#definindo os centros como sendo os pontos 'pegados' de ids.
	centers = dataset[ids, :]
	closest = np.zeros(dataset.shape[0])
	#adapto meus pesos até que se atinja um limiar.
	divergence = 2*threshold
	while(divergence > threshold):
		divergence = 0
		#para cada linha do meu conjunto de dados..
		for i in range(dataset.shape[0]):
			#pega uma linha i do dataset e verifica se ela é mais próxima do cluster 1 ou do cluster 2 ou do...
			row = dataset[i, :]
			euclidian_distance = np.zeros(clusters)
			for j in range(clusters):
				#calcula a distancia euclidiana daquela linha para todos os clusters.
				euclidian_distance[j] = np.linalg.norm(row-centers[j, :])
			#pega o primeiro elemento, ou seja, o cluster vitorioso, o cluster mais próximo do ponto.
			ids = np.argsort(euclidian_distance)[0]
			#para aquele ponto do meu conjunto de dados, sei qual é o cluster mais perto dele.
			closest[i] = ids

		old_centers = np.copy(centers)
		#sabendo qual o cluster mais próximo para cada ponto, devo agora adaptar meus centros.
		for i in range(clusters):
			#pego todos os elementos próximos daquele cluster
			ids = np.where(closest==i)[0]
			#pego a média das colunas daqueles elementos para reatribuir o novo centro.
			centers[i] = np.mean(dataset[ids], axis=0)
		#norma de frobenius 	
		divergence = np.sqrt(np.sum((old_centers - centers)**2))
		print(divergence)

	return (centers)

iris = datasets.load_iris()
kmeans(iris.data, clusters=3, threshold=0.1)



