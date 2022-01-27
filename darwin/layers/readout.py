'''
Implementations of readout layers.
'''

import numpy as np
import tensorflow as tf

from tensorflow.keras.regularizers import l1
from tensorflow.keras.layers import Activation

from qkeras import QDense, QActivation
from qkeras.quantizers import quantized_bits

from darwin.quantization import fxp_quantization

class SingleLayerDense:
    '''Readout layer implementation containing a single layer of neurons. Transient is used during training. This readout is useful to simulate response of dynamic systems.

    Parameters
    ----------
    nbit_quant : int
        Number of bits used for the quantization.
    '''

    def __init__(self, nbit_quant):
        self.nbit_quant = nbit_quant

    def fit(self, states, y_train):
        '''Custom code for readout training.

        Parameters
        ----------
        states : numpy.ndarray
            Internal states of the hidden layer.
        y_train : numpy.ndarray
            Training labels.
        '''
        transient = min(int(y_train.shape[0] / 10), 100)
        self.W_out = fxp_quantization(np.dot(np.linalg.pinv(states[transient:, :]), y_train[transient:, :]).T, nbit_quant=self.nbit_quant)

    def predict(self, states):
        '''Helper functions to predict new samples.

        Parameters
        ----------
        states : numpy.ndarray
            Internal states of the hidden layer.

        Returns
        -------
        numpy.ndarray
            Predicted labels or values.
        '''
        outputs = np.zeros((states.shape[0], self.W_out.shape[0]))
        for n in range(states.shape[0]):
            outputs[n, :] = fxp_quantization(np.dot(self.W_out, states[n, :]), nbit_quant=32)
        return outputs

class MultiLayerDense:
    '''Implementation of a readout layer consisting in a multi-layer fully-connected classifier or regressor. This class is meant to be used only in combination with objects derived from one of the darwin.networks classes.

    Parameters
    ----------
    n_inputs : int
        The number of input values for the readout layer
    readout_layers : dict
        Collection of dictionaries indicating the structure and parameters of dense layers used for the readout implementation
    readout_optimizer : tensorflow.keras.optimizers
        Function that belongs to the Keras optimizer library
    readout_loss : tensorflow.keras.losses
        Function that belongs to the Keras loss functions library
    readout_metrics : tensorflow.keras.metrics
        Function that belongs to the Keras metrics library
    nbit_quant : int
        Number of bits used for the quantization. None indicates full precision will be used
    seed : int
        Initial value for the random number generator.
    '''

    def __init__(self, n_inputs, readout_layers, readout_optimizer, readout_loss, readout_metrics, nbit_quant, seed=0):
        self.n_inputs = n_inputs
        self.readout_layers = readout_layers
        self.readout_optimizer = readout_optimizer
        self.readout_loss = readout_loss
        self.readout_metrics = readout_metrics
        self.nbit_quant = nbit_quant

        np.random.RandomState(seed)
        tf.random.set_seed(seed)

        self.init_readout()

    def init_readout(self):
        '''Initializes the readout layer with user's specified parameters'''
        self.readout = tf.keras.models.Sequential()
        self.readout.add(tf.keras.layers.Input(shape=(self.n_inputs,)))
        for layer in self.readout_layers:
            self.readout.add(
                QDense(
                    units=layer['units'],
                    kernel_quantizer=quantized_bits(self.nbit_quant,
                                                    0,
                                                    alpha=1),
                    bias_quantizer=quantized_bits(self.nbit_quant,
                                                  0,
                                                  alpha=1),
                    kernel_initializer='lecun_uniform',
                    kernel_regularizer=l1(0.0001)
                )
            )
            if hasattr(layer['activation'], '__name__'):
                self.readout.add(QActivation(layer['activation'](self.nbit_quant)))
            elif layer['activation'] == 'softmax':
                self.readout.add(Activation('softmax'))
            elif layer['activation'] == 'identity':
                continue
            else:
                print('Activation function not supported!')
        self.readout.compile(loss=self.readout_loss, optimizer=self.readout_optimizer, metrics=self.readout_metrics)

    def fit(self, X_train, y_train, X_val, y_val, batch_size, epochs):
        '''Wrapper for Keras' fit function.

        Parameters
        ----------
        X_train : numpy.ndarray
            Tensor representing the training set.
        y_train : numpy.ndarray
            Tensor representing the training labels.
        X_val : numpy.ndarray
            Tensor representing the validation set.
        y_val : numpy.ndarray
            Tensor representing the validation labels.
        batch_size : int
            Dimension of the training batch.
        epochs : int
            Number of training epochs.

        Returns
        -------
        tf.history
            A Tensorflow history object containing a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values.

        '''
        history = self.readout.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=True, validation_data=(X_val, y_val))
        return history

    def predict(self, X):
        '''Wrapper for Keras' predict function.

        Parameters
        ----------
        X : numpy.ndarray
            Input tensor.

        Returns
        -------
        numpy.ndarray
            Collection of predicted probabilities.

        '''
        return self.readout.predict(X)