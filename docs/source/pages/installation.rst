Installation
============

First, create a local virtual environment via `Anaconda <https://www.anaconda.com/>`_, `Conda <https://docs.conda.io/en/latest/>`_ or `venv <https://docs.python.org/3/library/venv.html>`_. For instance, the following instruction refers to `conda`, and it will create a virtual environment named `darwin-test` having Python 3.9:

    .. code-block:: console

        conda create --name darwin-test python=3.9

After setting-up the local virtual environment, clone the repository and `cd` into the root of the project, i.e.:

    .. code-block:: console

        git clone git@github.com:lnis-uofu/darwin.git
        cd darwin

At this point, it will suffice to run the following command to get `darwin` installed on your system:

    .. code-block:: console
        
        pip install -e .

The install process should automatically collect and install all required packages as well. At the end, the command

    .. code-block:: console

        pytest

can be used to make sure the installation was successful.

Requirements
------------

The complete list of requirements can be found in the `requirements.txt` file available in the project root.