'''Utility library containing miscellaneaus methods.
'''

import os

from fxpmath import Fxp
from qkeras.utils import model_save_quantized_weights

def annotate_readout_layer_parameters(model, output_dir, ext='txt'):
    '''Annotates parameters of the readout layer.

    Parameters
    ----------
    model : EchoStateNetwork
        The model to be annotated
    output_dir : str
        Path of the destination output directory where files will be saved
    ext : str, optional
        Extension of the generated output files, by default 'txt'
    '''
    readout_dict = model_save_quantized_weights(model.readout.readout)
    for k in readout_dict:
        weights = readout_dict[k]['weights'][0]
        biases = model_save_quantized_weights(model.readout.readout)[k]['weights'][1]

        readout_param_file = os.path.join(output_dir, f'readout_{k}.' + ext)

        write_numpy_to_file(weights, model.nbit_quant, weights.shape[1], 0, readout_param_file)
        write_numpy_to_file(biases, model.nbit_quant, biases.shape[0], weights.shape[0], readout_param_file, append=True)

def annotate_hidden_layer_parameters(model, input_sample, output_dir, ext='txt'):
    '''Annotates the parameters of the specified model in dedicated output files.

    Parameters
    ----------
    model : EchoStateNetwork
        The model to be annotated
    output_dir : str
        Path of the destination output directory where files will be saved
    ext : str, optional
        Extension of the generated output files, by default 'txt'
    '''

    win_w_file = os.path.join(output_dir, 'win_w.' + ext)
    input_states_file = os.path.join(output_dir, 'input_states.' + ext)

    write_numpy_to_file(model.W_in, model.nbit_quant, model.W_in.shape[0], 0, win_w_file)
    write_numpy_to_file(model.W.T, model.nbit_quant, model.W.shape[0], model.W_in.shape[1], win_w_file, append=True)
    write_numpy_to_file(input_sample, model.nbit_quant, 1, 0, input_states_file)
    write_numpy_to_file(model.states, model.nbit_quant, 1, input_sample.shape[1], input_states_file, append=True)

def write_numpy_to_file(arr, nbit_quant, modulo, address_start, of_file, append=False):
    '''Annotates a numpy array to a text file, one value per line.

    Parameters
    ----------
    arr : numpy.ndarray
        The input array
    nbit_quant : int
        Size of the quantized word
    modulo : int
        Number of unique parameters for each neuron
    address_start : int
        Initial value for the address register
    of_file : str
        Complete path of the output file
    append : bool, optional
        If true, parameters are added to the specified file, by default False
    '''
    address_counter = address_start
    frame_counter = 0
    write_mode = 'w'

    if append:
        write_mode = 'a'

    with open(of_file, mode=write_mode, encoding='utf-8') as of:
        q_addr = Fxp(address_counter, signed=False, n_word=nbit_quant, n_frac=0)
        of.write(q_addr.hex() + '\n')
        for v in arr.flatten():
            if frame_counter == modulo:
                frame_counter = 0
                address_counter += 1
                q_addr = Fxp(address_counter, signed=False, n_word=nbit_quant, n_frac=0)
                of.write(q_addr.hex() + '\n')

            qv = Fxp(v, signed=True, n_word=nbit_quant, n_frac=nbit_quant - 1)
            of.write(qv.hex() + '\n')
            frame_counter += 1
