import numpy as np

def knn(dataset, query_point, k):
	#coluna referente a classe desejada.
	class_id = dataset[:, dataset.shape[1]-1]
	#para cada linha do dataset
	distances_neighbors = np.zeros(dataset.shape[0])
	for i in range(dataset.shape[0]):
	#aplicando distancia euclidiana.
		distances_neighbors[i] = np.linalg.norm(query_point-dataset[i, 0:(dataset.shape[1]-1)])
	#obtendo os IDs das distâncias ordenadas.
	print(distances_neighbors)
	ids = np.argsort(distances_neighbors)[0:k]
	#realizando contagem dos elementos com maior ocorrência.
	labels = np.bincount(class_id[ids])
	#retornando a classe do elemento mais votado.
	return np.argmax(labels)

dataset = np.array([[34, 78, 0],[32, 65, 0],[26,89,1],[22,56,1]])

print(knn(dataset, np.array([26,80]), 1))
