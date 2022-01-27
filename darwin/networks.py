'''
Implementation of hidden layers for supported Reservoir Computing models.
'''

import numpy as np

from sklearn.metrics import accuracy_score

from darwin.layers.readout import SingleLayerDense, MultiLayerDense
from darwin.quantization import fxp_quantization, downcast
from darwin.activations import *

class EchoStateNetwork:
    '''Echo State Network with no teacher signal capable of solving classification
    and regression tasks. Different types of readout layers can be specified
    via the ``readout_layer`` parameter. In particular, to use a multi-layer
    dense readout, the following parameters can be specified:

    - **type** (str): Typology of readout (it must be the class name of the readout as reported in darwin.layers.readout).
    - **hyper_params** (dict): Dictionary indicating the structure and parameters of dense layers used for the readout implementation.
    - **optimizer** (tensorflow.keras.optimizers): Function that belongs to the Keras optimizer library.
    - **loss** (tensorflow.keras.losses): Function that belongs to the Keras loss functions library.
    - **metrics** (tensorflow.keras.metrics): Function that belongs to the Keras metrics library.

    A example multi-layer dense readout is reported in the following:

    .. code-block:: python

        readout_layer= {
            'type': 'MultiLayerDense',
            'hyper_params': {
                'layers': [
                    {'units': 80, 'activation': quantized_relu},
                    {'units': 10, 'activation': 'softmax'}
                ],
                'optimizer': tf.keras.optimizers.SGD(lr=0.1, momentum=0.000005, nesterov=True),
                'loss': tf.keras.losses.categorical_crossentropy,
                'metrics': ['accuracy'],
            }
        }
    
    Alternatively, a single-layer dense readout can be instantiated
    as in the following example:

    .. code-block:: python

        readout_layer= {
            'type': 'SingleLayerDense'
        },

    In this case, no other parameters are accepted.

    Parameters
    ----------
    n_inputs : int
        The number of inputs.
    n_outputs : int
        The number of outputs.
    n_neurons : int
        Number of neurons in the hidden layer.
    sparsity : float
        Percentage of connections to be removed in the hidden layer.
    spectral_radius : float
        Adjusting factor for hidden neuronal connections.
    activation : darwin.activation
        One of the activation functions in darwin.activation to be used for
        neurons in the hidden layer.
    readout_layer : dict
        Description of the readout layer. Depending on the type of readout,
        different parameters can be included as a dictionary. See the documentation
        for more details.
    nbit_quant : int
        Number of bits used for the quantization. None indicates full
        precision will be used.
    feedforward : bool
        Wether to use a feedforward connection of the inputs to the readout layer.
    feedback : bool
        If True, feedback connections are used.
    continuation : bool
        If True, states are initialized to the latest state achieved
        during the training phase.
    dynamic_update : bool
        If True, states update at time t is performed using excitatory states at
        time t-1. Otherwise, the same states are used to generate any
        future states update.
    seed : int
        Initial value for the random number generator.
    '''

    def __init__(
        self,
        n_inputs,
        n_outputs,
        n_neurons,
        sparsity,
        spectral_radius,
        activation,
        readout_layer,
        nbit_quant=None,
        feedforward=False,
        feedback=False,
        continuation=False,
        dynamic_update=False,
        seed=0
    ) -> None:
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_neurons = n_neurons
        self.sparsity = sparsity
        self.spectral_radius = spectral_radius
        self.activation = activation
        self.nbit_quant = nbit_quant
        self.feedforward = feedforward
        self.feedback = feedback
        self.continuation = continuation
        self.dynamic_update = dynamic_update

        # Take note of the type of readout layer connected to the reservoir
        self.readout_type = readout_layer['type']

        # Set a value for the random number generator
        self.random_state = np.random.RandomState(seed)

        # Initialize hidden layer
        self.initialize_network()

        # Initialize readout layer
        feedforward_inputs_readout = 0
        if self.feedforward:
            feedforward_inputs_readout = self.n_inputs

        # Depending on the type selected by the user, initialize the readout layer
        if self.readout_type == MultiLayerDense.__name__:
            self.readout = MultiLayerDense(
                n_inputs=self.n_neurons + feedforward_inputs_readout,
                readout_layers=readout_layer['hyper_params']['layers'],
                readout_optimizer=readout_layer['hyper_params']['optimizer'],
                readout_loss=readout_layer['hyper_params']['loss'],
                readout_metrics=readout_layer['hyper_params']['metrics'],
                nbit_quant=nbit_quant,
                seed=seed
            )
        elif self.readout_type == SingleLayerDense.__name__:
            self.readout = SingleLayerDense(nbit_quant=self.nbit_quant)

    def initialize_network(self):
        '''Initializes the hidden layer with user's specified parameters.'''
        self.W_in = self.random_state.rand(self.n_neurons, self.n_inputs) * 2 - 1
        self.W = self.random_state.rand(self.n_neurons, self.n_neurons) - 0.5
        self.W[self.random_state.rand(*self.W.shape) < self.sparsity] = 0
        self.states = self.random_state.rand(self.n_neurons) * 2 - 1

        if self.spectral_radius is not None:
            self.W = self.W * (self.spectral_radius / np.max(np.abs(np.linalg.eigvals(self.W))))

        if self.nbit_quant is not None:
            self.quantize_apply()

        if self.feedback:
            self.W_feedb = self.random_state.rand(self.n_neurons, self.n_outputs) * 2 - 1

    def quantize_apply(self):
        '''Executes a symmetric quantization on the parameters of the hidden layer.'''
        self.W = fxp_quantization(self.W, self.nbit_quant)
        self.W_in = fxp_quantization(self.W_in, self.nbit_quant)
        self.states = fxp_quantization(self.states, self.nbit_quant)

        if self.feedback:
            self.W_feedb = fxp_quantization(self.W_feedb, self.nbit_quant)

    def fit(self, X_train, y_train, X_val, y_val, batch_size, epochs):
        '''Wrapper for readout fit function.

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
        tf.history or None
            A Tensorflow history object containing a record of training loss values
            and metrics values at successive epochs, as well as validation loss values
            and validation metrics values. None is returned when the selected readout
            does not provide a history object.
        '''

        partial_training = self.states_update(X_train)
        if X_val is not None:
            partial_validation = self.states_update(X_val)

        if self.readout_type == MultiLayerDense.__name__:
            history = self.readout.fit(X_train=partial_training, y_train=y_train, X_val=partial_validation, y_val=y_val, batch_size=batch_size, epochs=epochs)
        elif self.readout_type == SingleLayerDense.__name__:
            self.y_train = y_train
            self.readout.fit(partial_training, y_train)
            history = None

        return history

    def score(self, y_true, y_pred):
        '''Comptues the accuracy score of an RC Network.

        Parameters
        ----------
        y_true : numpy.ndarray
            Actual labels.
        y_pred : numpy.ndarray
            Predicted labels.

        Returns
        -------
        float
            Ratio between correctly classified samples and the over all
            number of samples.

        '''
        return accuracy_score(y_true, y_pred)

    def predict(self, X_val):
        '''Predicts the class or value associated to the test set.

        Parameters
        ----------
        X_val : numpy.ndarray
            Input values.

        Returns
        -------
        numpy.ndarray
            Predicted values.
        '''
        partial_validation = self.states_update(X_val)
        # print('Debug output: partial_validation')
        # print(partial_validation)
        return self.readout.predict(partial_validation)

    def states_update(self, X):
        '''Computes the intermediate states of the hidden neurons.

        Parameters
        ----------
        X : numpy.ndarray
            Input values (training or validation set).

        Returns
        -------
        numpy.ndarray
            Excitatory states of the hidden neurons.
        '''
        activation_quant = downcast
        excitatory_states = np.zeros((X.shape[0], self.n_neurons))

        # The first computation of excitatory states is the same for dynamic and non-dynamic configurations.
        pes = np.dot(self.W, self.states) + np.dot(self.W_in, X[0])
        excitatory_states[0, :] = activation_quant(self.activation(pes), self.nbit_quant)

        # print('Debug output: pes')
        # print(pes)

        for x in range(1, X.shape[0]):
            if not self.dynamic_update:
                # Do not use memory to update the network states.
                pes = np.dot(self.W, self.states) + np.dot(self.W_in, X[x])
            else:
                # Leverage previous states to compute the new ones.
                pes = np.dot(self.W, excitatory_states[x - 1]) + np.dot(self.W_in, X[x])
                if self.feedback:
                    pes = pes + np.dot(self.W_feedb, self.y_train[x - 1])

            excitatory_states[x, :] = activation_quant(self.activation(pes), self.nbit_quant)

        if self.continuation:
            self.states = excitatory_states[-1]

        if self.feedforward:
            excitatory_states = np.hstack((excitatory_states, X))

        return excitatory_states

    def summary(self):
        '''Helper function to print the structure of the model (inspired
        by and based upon TensorFlow's model.summary() function).
        '''
        positions = [.45, .85, 1.]

        print('Model: Echo State Network')
        print('_' * 80)
        self.print_summary_row(['Layer', 'Neurons #', "Params #"], positions)
        print('=' * 80)
        self.print_summary_row(['Reservoir', self.n_neurons, self.n_neurons - round(self.n_neurons * self.sparsity)], positions)
        print('_' * 80)
        print('\n')
        self.readout.readout.summary(line_length=80)

    def print_summary_row(self, fields, positions):
        '''Helper function to annotate the structure of the model, including both
        the hidden and the readout layers. Note that this function is borrowed from
        Tensortflow's model.summary() function.

        Parameters
        ----------
        fields : list
            Collection of items to be printed.
        positions : list
            Relative or absolute positions of elements in each line.
        '''
        positions = [int(80 * p) for p in positions]
        line = ''
        for i in range(len(fields)):
            if i > 0:
                line = line[:-1] + ' '
            line += str(fields[i])
            line = line[:positions[i]]
            line += ' ' * (positions[i] - len(line))
        print(line)