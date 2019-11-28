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
    accuracies = [sig_train_score[acc], tanh_train_score[acc], relu_train_score[acc], softmax_train_score[acc], elu_train_score[acc]]
    
    val_losses = [sig_train_score['val_loss'], tanh_train_score['val_loss'], relu_train_score['val_loss'], softmax_train_score['val_loss'], elu_train_score['val_loss']]
    acc = 'val_sparse_categorical_accuracy'
    val_accuracies = [sig_train_score[acc], tanh_train_score[acc], relu_train_score[acc], softmax_train_score[acc], elu_train_score[acc]]

    test_score = {'Sigmoid': sig_test_score, 'Tanh': tanh_test_score, 'ReLU': relu_test_score, 'Softmax': softmax_test_score, 'ELU': elu_test_score}
    print(test_score)

    plt.style.use('ggplot')

    df = pd.DataFrame(np.array(accuracies) * 100)
    df = df.T

    df2 = pd.DataFrame(np.array(losses))
    df2 = df2.T

    df3 = pd.DataFrame([np.array(losses[0]), np.array(val_losses[0])])
    df3 = df3.T

    df4 = pd.DataFrame([np.array(losses[1]), np.array(val_losses[1])])
    df4 = df4.T

    df5 = pd.DataFrame([np.array(losses[2]), np.array(val_losses[2])])
    df5 = df5.T

    df6 = pd.DataFrame([np.array(losses[3]), np.array(val_losses[3])])
    df6 = df6.T

    df7 = pd.DataFrame([np.array(losses[4]), np.array(val_losses[4])])
    df7 = df7.T

    fig1, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(df)
    ax1.set_title('Precisão')
    ax1.legend(['Sigmoid', 'Tanh', 'ReLU', 'Softmax', 'ELU'])
    ax1.set(xlabel='Época', ylabel='Precisão (%)')

    ax2.plot(df2)
    ax2.set_title('Custo')
    ax2.legend(['Sigmoid', 'Tanh', 'ReLU', 'Softmax', 'ELU'])
    ax2.set(xlabel='Época', ylabel='Erro')

    fig2, ((ax3, ax4), (ax5, ax6), (ax7,ax8)) = plt.subplots(3, 2, sharex=True, gridspec_kw={'hspace': 0.8})
    fig2.suptitle('Regularização')
    ax3.plot(df3)
    ax3.set_title('Sigmoid')
    ax3.legend(['Treinamento', 'Validação'])

    ax4.plot(df4)
    ax4.set_title('Tanh')
    ax4.legend(['Treinamento', 'Validação'])

    ax5.plot(df5)
    ax5.set_title('ReLU')
    ax5.legend(['Treinamento', 'Validação'])
    ax5.set(ylabel='Erro')

    ax6.plot(df6)
    ax6.set_title('Softmax')
    ax6.legend(['Treinamento', 'Validação'])
    ax6.set(xlabel='Época')

    ax7.plot(df7)
    ax7.set_title('ELU')
    ax7.legend(['Treinamento', 'Validação'])
    ax7.set(xlabel='Época')

    fig2.delaxes(ax8)

    plt.show()