The Darwin Project Documentation
================================

.. toctree::
    :hidden:

    About <self>

The `Darwin` Python package allows to efficiently design, train, and analyze quantized *Reservoir Computing Networks* (RCNs). Current features include:

1. A grid-search-based hyper-parameter initialization routine, which allows the user to specify the search range for key hyper-parameters of the hidden and the readout layers;

2. Few-shot training routine on non-traditional readout layers to achieve improved generalization capabilities;

3. Routines for exporting quantized RCN's weights to text format, which can later be used for hardware simulations.

.. toctree::
    :hidden:
    :glob:
    :caption: Getting Started
    
    pages/installation
    pages/tutorials

.. toctree::
    :hidden:
    :glob:
    :caption: API Reference

    pages/api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
