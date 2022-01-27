'''
Library for quantization routines.
'''

import numpy as np
from fxpmath import Fxp

def fxp_quantization(fp, nbit_quant, bin_repr=False):
    '''Performs n-bit quantization leveraging the fxpmath library.

    Parameters
    ----------
    fp : float
        Full precision number to quantize.
    nbit_quant : int
        Number of word bits (first bit is reserved for the sign).
    bin_repr : bool, optional
        If True the binary two's complement representation is returned,
        otherwise the quantized number is returned as is, by default False.

    Returns
    -------
    float or str
        A float representing the quantized input number, or a string that
        represents the binary encoding of the quantized input number.
    '''

    qfp = Fxp(fp, signed=True, n_word=nbit_quant, n_frac=nbit_quant - 1)
    if bin_repr:
        return qfp.bin()
    else:
        return qfp.get_val()

def downcast(fp, nbit_quant):
    '''Applies a truncation on LSBs in order to downcast 32-bit words over
    the specified amount of bits.

    Parameters
    ----------
    fp : float
        The input number.
    nbit_quant : int
        Number of word bits (first bit is reserved for the sign).

    Returns
    -------
    float
        A float representing the quantized input number.
    '''
    qv = Fxp(fp, signed=True, n_word=nbit_quant, n_frac=nbit_quant - 1, rounding='trunc')
    return qv.get_val()

def symmetric_quantization(fp, nbit_quant, bound=1):
    '''Applies symmetric quantization on a given full-precision number.

    Parameters
    ----------
    fp : float
        Full precision number to quantize.
    bound : int
        Lower/Upper bound limit of the quantized interval. (Default value = 1)
    nbit_quant : int
        Number of word bits (first bit is reserved for the sign).

    Returns
    -------
    float
        The quantized representation of the input number.

    '''
    bound = bound - 2**-float(nbit_quant - 1)
    p = np.round(fp * (2**(nbit_quant - 1))) / 2**(nbit_quant - 1)
    return np.maximum(-bound, np.minimum(bound, p))

def lsb_truncation(fp, nbit_quant, bound=1):
    '''Applies truncate quantization on a given full-precision number.

    Parameters
    ----------
    fp :
        Full precision number to quantize.
    bound :
        Lower/Upper bound limit of the quantized interval. (Default value = 1)
    nbit_quant :
        

    Returns
    -------
    float
        The quantized representation of the input number.

    '''
    val = truncate_bin(fp, nbit_quant)
    bound = bound - 2**-float(nbit_quant - 1)
    p = val / 2**(nbit_quant - 1)
    return np.maximum(-bound, np.minimum(bound, p))

def truncate_bin(fp, nbit_quant):
    '''

    Parameters
    ----------
    fp :
        
    nbit_quant :
        

    Returns
    -------

    '''
    x = np.round(fp * (2**(32 - 1)))  # Convert fp in a 32 bit precision nb
    for i, b in enumerate(x):
        if b < 0:
            trunc = bin(int(b))[3:3 + nbit_quant - 1]
            val = -int(trunc, 2)
        else:
            trunc = bin(int(b))[2:2 + nbit_quant - 1]
            val = int(trunc, 2)
        x[i] = val
    return x