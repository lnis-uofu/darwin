'''
Model selection library to automatically determine the optimal Reservoir Computing network configuration.
'''

import itertools
import multiprocessing

from os import getpid
from multiprocessing import Pool
from darwin.networks import EchoStateNetwork

class GridSearchEchoStateNetwork():
    '''Architecture exploration by means of a grid search evaluation.

    Parameters
    ----------
    n_inputs : int
        Number of inputs.
    n_neurons : list
        Number of neurons in the hidden layer.
    sparsity : list
        Percentage of connections to be removed in the hidden layer.
    spectral_radius : list
        Adjusting factor for hidden neuronal connections.
    nbit_quant : list
        Number of bits used for the quantization. None indicates full precision will be used.
    readout_layers : dict
        Collection of dictionaries indicating the structure and parameters of dense layers used for the readout implementation.
    readout_optimizer : keras.optimizers
        Function that belongs to the Keras optimizer library.
    readout_loss : keras.losses
        Function that belongs to the Keras loss functions library.
    readout_metrics : keras.metrics
        Function that belongs to the Keras metrics library.
    X_train : numpy.ndarray
        Training set.
    y_train : numpy.ndarray
        Training labels.
    X_test : numpy.ndarray
        Test or validation set.
    y_test : numpy.ndarray
        Test or Validation labels.
    batch_size : int
        Size of the training batch.
    epochs : int
        Number of training epochs.
    seed : int, optional
        Initial value for the random number generator.

    Returns
    -------

    '''

    def __init__(
        self,
        n_inputs,
        n_neurons,
        sparsity,
        spectral_radius,
        activation,
        nbit_quant,
        readout_layers,
        readout_optimizer,
        readout_loss,
        readout_metrics,
        X_train,
        y_train,
        X_test,
        y_test,
        batch_size,
        epochs,
        seed
    ) -> None:
        self.n_inputs = n_inputs
        self.activation = activation
        self.readout_layers = readout_layers
        self.readout_optimizer = readout_optimizer
        self.readout_loss = readout_loss
        self.readout_metrics = readout_metrics
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.batch_size = batch_size
        self.epochs = epochs
        self.seed = seed

        # Permute hyperparameters and generate a list of dictionaries with all the permutations.
        hyper_names = ['n_neurons', 'sparsity', 'spectral_radius', 'nbit_quant']
        hp_perm = list(itertools.product(*[n_neurons, sparsity, spectral_radius, nbit_quant]))
        self.hyper_values = [dict(zip(hyper_names, x)) for x in hp_perm]

    def run(self):
        '''Grid search exploration on multiple processors.'''
        p = Pool(multiprocessing.cpu_count())

        grid_results = [p.apply_async(self.execute, args=(h,)) for h in self.hyper_values]

        p.close()
        p.join()

        # Select the best configuration (TODO: improve max selection).
        best_acc = 0
        best_conf = {}
        for l in [r.get() for r in grid_results]:
            if l['accuracy'] > best_acc:
                best_acc = l['accuracy']
                best_conf = l['hyper_values']

        return best_acc, best_conf

    def execute(self, hyper_values):
        '''Core function of the grid search analysis, where EchoStateNetwork
        objects are created and evaluated.
        Performances are annotated in order to determine the best
        hyper-parameter configuration.

        Parameters
        ----------
        hyper_values : dict
            Collection of permuted hyper-parameter values.

        Returns
        -------
        dict
            The best hyper-parameter configuration.

        '''
        print(f'Running {getpid()}')
        esn = EchoStateNetwork(
            n_inputs=self.n_inputs,
            n_neurons=hyper_values['n_neurons'],
            sparsity=hyper_values['sparsity'],
            spectral_radius=hyper_values['spectral_radius'],
            activation=self.activation,
            nbit_quant=hyper_values['nbit_quant'],
            readout_layers=self.readout_layers,
            readout_optimizer=self.readout_optimizer,
            readout_loss=self.readout_loss,
            readout_metrics=self.readout_metrics,
            seed=self.seed
        )
        history = esn.fit(
            X_train=self.X_train,
            y_train=self.y_train,
            X_val=self.X_test,
            y_val=self.y_test,
            batch_size=self.batch_size,
            epochs=self.epochs
        )
        print(f'Closing {getpid()}')
        return {
            'accuracy': history.history['accuracy'][0],
            'hyper_values': hyper_values
        }
