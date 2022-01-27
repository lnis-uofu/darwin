# Darwin
A Design Automation Framework for Reservoir Computing Networks

## Overview
The `Darwin` Python package allows to efficiently design, train, and analyze quantized _Reservoir Computing Networks_ (RCNs). Current features include:
1. A grid-search-based hyper-parameter initialization routine, which allows the user to specify the search range for key hyper-parameters of the hidden and the readout layers;
2. Few-shot training routine on non-traditional readout layers to achieve improved generalization capabilities;
3. Routines for exporting quantized RCN's weights to text format, which can later be used for hardware simulations.

## Installation
First, create a local virtual environment via [Anaconda](https://www.anaconda.com/), [Conda](https://docs.conda.io/en/latest/) or [venv](https://docs.python.org/3/library/venv.html). For instance, the following instruction refers to `conda`, and it will create a virtual environment named `darwin-test` having Python 3.9:

    conda create --name darwin-test python=3.9

After setting-up the local virtual environment, clone the repository and `cd` into the root of the project, i.e.:

    git clone git@github.com:lnis-uofu/darwin.git
    cd darwin

At this point, it will suffice to run the following command to get `darwin` installed on your system:

    pip install -e .

The install process should automatically collect and install all required packages as well. At the end, the command

    pytest

can be used to make sure the installation was successful.

## Usage
The `darwin` framework can be used as a typical Python package. For instance, in a Python interpreter, the instructions

    >>> from darwin.activations import tanh_activation
    >>> tanh_activation(0.5)

will leverage the `tanh_activation` function declared in the `activations` module to generate the result of the `tanh` operation, e.g., 0.462117 in this specific example. In the same way, all other modules can be referenced in any other Python script.

For additional reference, the `examples` directory contains a few Jupyter Notebooks where RCNs are test on classification tasks on audio and image datasets.

## Documentation
The `docs` folder contains source files to build the documentation. To do so, just `cd` into `docs` and issue the following command:

    make html

A new `build` directory will be generated. It will contain the documentation static website, which can be examined by opening the `build/html/index.html` file with any web browser.