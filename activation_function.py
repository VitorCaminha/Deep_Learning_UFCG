import numpy as np
import pandas as pd
import tensorflow as tf
import os # para criar pastas
from matplotlib import pyplot as plt # para mostrar imagens


def net(x_train, y_train, x_test, y_test, epoch=5, activation='relu'):

	# Monta uma rede neural simples, com duas camadas e 512 neurônios por camada
	x_input = tf.keras.layers.Input(shape=(784,))
	l1 = tf.keras.layers.Dense(512, activation=activation)(x_input)
	l2 = tf.keras.layers.Dense(512, activation=activation)(l1)
	logit = tf.keras.layers.Dense(10, activation='softmax')(l2)

	model = tf.keras.Model(inputs=x_input, outputs=logit)

	model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
              metrics=['sparse_categorical_accuracy'])

	fit = model.fit(x_train, y_train,
                    batch_size=64,
                    epochs=epoch,
                    validation_split=0.2)
		
	history = fit.history

	test_scores = model.evaluate(x_test, y_test, verbose=2)

	return history, test_scores


if __name__ == '__main__':
	
	(x_input, y_input), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	x_input = x_input.reshape(60000, 784).astype('float32') / 255
	x_test = x_test.reshape(10000, 784).astype('float32') / 255

	epoch = 50

	# cria uma rede neural para cada função de ativação
	sig_train_score, sig_test_score = net(x_input, y_input, x_test, y_test, epoch, activation='sigmoid')
	tanh_train_score, tanh_test_score = net(x_input, y_input, x_test, y_test, epoch, activation='tanh')
	relu_train_score, relu_test_score = net(x_input, y_input, x_test, y_test, epoch, activation='relu')
	softmax_train_score, softmax_test_score = net(x_input, y_input, x_test, y_test, epoch, activation='softmax')
	elu_train_score, elu_test_score = net(x_input, y_input, x_test, y_test, epoch, activation='elu')

	losses = [sig_train_score['loss'], tanh_train_score['loss'], relu_train_score['loss'], softmax_train_score['loss'], elu_train_score['loss']]
	acc = 'sparse_categorical_accuracy'
	accuracies = [sig_train_score[acc], tanh_train_score[acc], relu_train_score[acc], softmax_train_score[acc],	elu_train_score[acc]]
	
	test_score = {'Sigmoid': sig_test_score, 'Tanh': tanh_test_score, 'ReLU': relu_test_score, 'Softmax': softmax_test_score, 'ELU': elu_test_score}
	print(test_score)

	plt.style.use('ggplot')

	df = pd.DataFrame(np.array(accuracies) * 100)
	df = df.T

	df2 = pd.DataFrame(np.array(losses))
	df2 = df2.T

	df.plot()
	plt.title('Accuracy')
	plt.legend(['Sigmoid', 'Tanh', 'ReLU', 'Softmax', 'ELU'])
	plt.ylabel('accuracy')
	plt.xlabel('epoch')

	df2.plot()
	plt.title('Loss')
	plt.legend(['Sigmoid', 'Tanh', 'ReLU', 'Softmax', 'ELU'])
	plt.ylabel('loss')
	plt.xlabel('epoch')
	
	plt.show()
