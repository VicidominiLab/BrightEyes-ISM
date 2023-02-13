Description
===============================

BrightEyes-ISM is a Python package for analysing and simulating Image Scanning Microscopy (ISM) datasets.

The analysis module contains libraries for:

    + Adaptive Pixel Reassignment (https://doi.org/10.1038/s41592-018-0291-9)
    + Focus-ISM (https://doi.org/10.1038/s41467-022-35333-y)
    + Image Deconvolution (https://doi.org/10.48550/arXiv.2211.12510)
    + Fourier Ring Correlation (https://doi.org/10.1038/s41467-019-11024-z)

The simulation module contains libraries for:

    + Generation of ISM point spread functions (https://doi.org/10.1016/j.cpc.2022.108315)
    + Generation of tubulin phantom samples

The dataio module contains libraries for

    + Reading the data and metadata from the MCS software (https://github.com/VicidominiLab/BrightEyes-MCS)


Note that all the image processing functions assume that the detector array has a squared geometry.
Datasets acquired with a non-cartesian arrangement (e.g. AiryScan) might require additional pre-processing.

Installation
===============================

The source code can be found at https://github.com/VicidominiLab/BrightEyes-ISM. The latest version of the package can be installed directly from GitHub:

.. code-block:: python

   pip install git+https://github.com/VicidominiLab/BrightEyes-ISM


Or the most recent stable version can be installed using pip:

.. code-block:: python

   pip install brighteyes-ism

We also provide most of the package's functionalities via a Napari plugin, which can be used as a graphical user interface.
The plugin is called Napari-ISM and can be found at https://github.com/VicidominiLab/napari-ISM. It can be installed using pip:

.. code-block:: python

   pip install napari-ism

Dependencies
============

BrightEyes-ISM requires Python >= 3.7 and the following packages:

.. code-block:: python

    numpy
    scipy
    matplotlib
    scikit-image
    scikit-learn
    poppy
    PyCustomFocus
    h5py
    tqdm
    statsmodels


Importing
============

The modules can be imported as in the following example

.. code-block:: python

   import brighteyes_ism.analysis.APR_lib as apr
   import brighteyes_ism.simulation.PSF_sim as psf
   import brighteyes_ism.dataio.mcs as mcs
   
Note that the installation requires an hyphen symbol (brighteyes-ism), but the import requires an underscore symbol (brighteyes_ism).

Example
============

An example of usage of BrightEyes-ISM can be found in the following notebook: https://github.com/VicidominiLab/BrightEyes-ISM/blob/main/examples/BrightEyes_ISM_demo.ipynb

Citing
============

BrightEyes-ISM can be cited as 

.. code-block:: python

    

Napari-ISM can be cited as

.. code-block:: python

    