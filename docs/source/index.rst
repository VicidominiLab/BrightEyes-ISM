.. BrightEyes-ISM documentation master file, created by
   sphinx-quickstart on Mon Feb  6 14:30:58 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

BrightEyes-ISM's documentation
==========================================

BrightEyes-ISM is a Python package for analysing and simulating Image Scanning Microscopy (ISM) datasets.

.. image:: ../img/BrightEyesISMlogo.jpg
  :width: 400
  :alt: BrightEyes-ISM Logo
  :align: center

Read about BrightEyes-ISM here:

   Zunino, A., Slenders, E., Fersini, F. et al. Open-source tools enable accessible and advanced image scanning microscopy data analysis. Nat. Photon. (2023). https://doi.org/10.1038/s41566-023-01216-x


BrightEyes-ISM contains the following modules.

The analysis module contains libraries for:

    + Adaptive Pixel Reassignment (https://doi.org/10.1038/s41592-018-0291-9)
    + Focus-ISM (https://doi.org/10.1038/s41467-022-35333-y)
    + Image Deconvolution (https://doi.org/10.1088/1361-6420/accdc5)
    + Fourier Ring Correlation (https://doi.org/10.1038/s41467-019-11024-z)

The simulation module contains libraries for:

    + Generation of ISM point spread functions (https://doi.org/10.1016/j.cpc.2022.108315)
    + Generation of tubulin phantom samples

The dataio module contains libraries for

    + Reading the data and metadata from the MCS software (https://github.com/VicidominiLab/BrightEyes-MCS)


Note that all the image processing functions assume that the detector array has a squared geometry.
Datasets acquired with a non-cartesian arrangement (e.g. AiryScan) might require additional pre-processing.


.. toctree::
   :caption: Usage
   :maxdepth: 8
   
   installation

.. toctree::
   :caption: Modules
   :maxdepth: 8

   brighteyes_ism.analysis
   brighteyes_ism.simulation
   brighteyes_ism.dataio

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
