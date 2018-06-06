import numpy as np

def weight(xi, xj, sigma):
	#função de calculo da distância definida pelo algoritmo.
	euclidian_distance = np.linalg.norm(xi-xj)
	return (np.exp(-euclidian_distance**2/(2*sigma**2)))

def dwnn(dataset, query_point, sigma):
	#pegando a coluna que representa a classe.
	class_id = dataset[:, dataset.shape[1]-1]
	#criando o vetor de pesos que irá armazenar os pesos de cada linha do dataset.
	weights = np.zeros(dataset.shape[0])
	#para cada linha do dataset, calcula o peso para o ponto de consulta.
	for i in range(dataset.shape[0]):
		weights[i] = weight(query_point, dataset[i, 0:dataset.shape[1]-1], sigma)
	#cada uma das possíveis classes que são produzidas = class_id
	#multiplica por cada possível classe produzida e pondera pelo somatório dos pesos.
	y = np.sum(weights*class_id)/np.sum(weights)

	return y

dataset = np.array([[1,1], [2,2], [3,3], [4,4], [5,5], [6,6], [7,7], [8,8], [9,9], [10,10]])

result = dwnn(dataset, 4.5, sigma=1)
print(result)
