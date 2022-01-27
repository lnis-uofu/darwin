'''
Library of quantized and non-quantized activation functions.
'''

import numpy as np

def tanh_activation(x):
    '''Tanh activation function.

    Parameters
    ----------
    x : float
        Value of the input pre-activation.

    Returns
    -------
    float
        The corresponding Tanh(x) value.

    '''
    return np.tanh(x)

def sigmoid_activation(x):
    r'''Sigmoid activation function defined as:
    
    .. math::
        y = \frac{1}{1+e^{-x}}

    Parameters
    ----------
    x : float
        Value of the input pre-activation.

    Returns
    -------
    float
        The result of the Sigmoid function.
    '''
    return 1 / (1 + np.exp(-x))

def sigmoid_tanh_activation(x):
    r'''A Sigmoid activation defined as:
    
    .. math::
        y = \frac{\tanh\left(\frac{x}{2}+1\right)}{2}

    Parameters
    ----------
    x : float
        Value of the input pre-activation.

    Returns
    -------
    float
        The result of the Sigmoid function.
    '''
    return (np.tanh(x / 2) + 1) / 2

def fast_sigmoid_activation(x):
    r'''Fast Sigmoid activation function defined as:
    
    .. math::
        y = \frac{x}{1+\lvert x \rvert}
    
    It can be used to reduce the amount of hardware resources needed to implement a Sigmoid function.

    Parameters
    ----------
    x : float
        Value of the input pre-activation.

    Returns
    -------
    float
        The result of the fast Sigmoid.
    '''
    return x / (1 + np.abs(x))

def relu_activation(x):
    '''Rectifier Linear Unit activation function.

    Parameters
    ----------
    x : float
        Value of the input pre-activation.

    Returns
    -------
    float
        Returns ``x`` if ``x`` is larger or equal to 0. Otherwise 0 is returned.
    '''
    return x * (x > 0)

def approximate_sigmoid_activation(x, alpha=0.001):
    r'''Approximate version of the Fast Sigmoid activation function defined as:
    
    .. math::
        y = 0.5 \frac{x \cdot \alpha}{1+\lvert x \cdot \alpha \rvert} + 0.5

    Parameters
    ----------
    x : float
        Value of the input pre-activation
    alpha : float, optional
        Noise factor, defaults to 0.001

    Returns
    -------
    float
        The result of the approximate Sigmoid.
    '''
    return 0.5 * (x * alpha / (1 + np.abs(x * alpha))) + 0.5

def ramp_activation(x, bound=1):
    '''A modified ReLU activation where the maximum (minimum) value for the input is capped at a specified value.

    Parameters
    ----------
    x : numpy.ndarray
        The input tensor
    bound : int, optional
        Upper bound of the activation, by default 1

    Returns
    -------
    numpy.ndarray
        A tensor having the same shape of the input tensor, where values are the result of the ramp activation.
    '''
    return np.maximum(-bound, np.minimum(bound, x))
