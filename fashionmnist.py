"""Imports"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

"""Carregando o dataset"""

dataset = keras.datasets.fashion_mnist
((imagens_treino, identificacoes_treino), (imagens_teste, identificacoes_teste)) = dataset.load_data()

"""Análise e exibição dos dados"""

print(len(imagens_treino))   #60.000 imagens para treino
print(imagens_treino.shape)  #array de 28 linhas por 28 colunas
plt.imshow(imagens_treino[0])
print(plt.title(identificacoes_treino[0]))   #identificacao de uma imagem especifica
total_de_classificacoes = identificacoes_treino.max() - identificacoes_treino.min()
nomes_de_classificacoes = ['Camiseta', 'Calça', 'Pullover', 'Vestido', 'Casaco', 'Sandália', 'Camisa', 'Tênis', 'Bolsa', 'Bota']
for idx in range(10):
    plt.subplot(2, 5, idx+1)
    plt.imshow(imagens_treino[idx])
    plt.title(nomes_de_classificacoes[idx])

"""Normalização"""

imagens_treino = imagens_treino/float(255)

"""Criação das layers"""

modelo = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),  #achata a imagem de 28,28 para 1 dimensao
    keras.layers.Dense(256, activation=tf.nn.relu),#cria a segunda coluna (hidden layer) com 256 nós e a RELU como função de ativação
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation=tf.nn.softmax) #ultima camada com 10 saidas
])

"""Compilação e treinamento do modelo"""

modelo.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
historico = modelo.fit(imagens_treino, identificacoes_treino, epochs=5, validation_split=0.2)

"""Salvando o modelo"""

modelo.save('modelo.h5')

"""Visualização das acurácias de treino e validação por época"""

plt.plot(historico.history['acc'])
plt.plot(historico.history['val_acc'])
plt.title('Acurácia por épocas')
plt.xlabel('épocas')
plt.ylabel('acurácia')
plt.legend(['treino', 'validação'])

"""Visualização das perdas de treino e validação por época"""

plt.plot(historico.history['loss'])
plt.plot(historico.history['val_loss'])
plt.title('Perdas por épocas')
plt.xlabel('épocas')
plt.ylabel('perda')
plt.legend(['treino', 'validação'])

"""Teste do modelo e do modelo salvo"""

modelo_salvo = load_model('modelo.h5')
testes = modelo.predict(imagens_teste)
print('resultado teste:', np.argmax(testes[1]))
print('número da imagem de teste:', identificacoes_teste[1])

testes_modelo_salvo = modelo_salvo.predict(imagens_teste)
print('resultado teste modelo salvo:', np.argmax(testes_modelo_salvo[1]))
print('número da imagem de teste:', identificacoes_teste[1])

"""Avaliação do modelo"""

perda_teste, acuracia_teste = modelo.evaluate(imagens_teste, identificacoes_teste)
print('Perda do teste:', perda_teste)
print('Acurácia do teste:', acuracia_teste)