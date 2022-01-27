from distutils.core import setup

setup(
    name='Darwin',
    version='1.0',
    description='A Design Automation Framework for Reservoir Computing Networks',
    author='Laboratory for NanoIntegrated Systems (LNIS) - The University of Utah',
    author_email='valerio.tenace@utah.edu',
    packages=['darwin',
              'darwin.layers'],
    install_requires=[
        'numpy>=1.22.1',
        'tensorflow>=2.7.0',
        'fxpmath>=0.4.5',
        'scikit_learn>=1.0.2',
        'pandas>=1.3.5',
        'QKeras>=0.9.0',
        'pytest>=6.2.5',
        'ipykernel>=6.7.0',
        'matplotlib>=3.5.1',
        'sphinx>=4.4.0',
        'sphinx-rtd-theme>=1.0.0',
        'nbsphinx>=0.8.8',
        'nbsphinx_link>=1.3.0'
    ]
)