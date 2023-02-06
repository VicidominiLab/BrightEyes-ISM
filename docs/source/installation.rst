Installation
===============================

The source code can be found at https://github.com/VicidominiLab/BrightEyes-ISM. The package can be installed directly from GitHub:

.. code-block:: python

   pip install git+https://github.com/VicidominiLab/BrightEyes-ISM


Or it can be installed using pip:

.. code-block:: python

   pip install brighteyes-ism


We also provide most of the package's functionalities via a Napari plugin, which can be used as a graphical user interface.
The plugin is called Napari-ISM and can be found at https://github.com/VicidominiLab/napari-ISM. It can be installed using pip:

.. code-block:: python

   pip install napari-ism

Dependencies
============

BrightEyes-ISM requires Python >= 3.10. The package also needs the following packages:

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
   
Note that the installation require an hyphen symbol, but the import requires an underscore symbol.