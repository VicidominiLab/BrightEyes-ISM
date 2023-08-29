# BrightEyes-ISM

[![License](https://img.shields.io/pypi/l/brighteyes-ism.svg?color=green)](https://github.com/VicidominiLab/BrightEyes-ISM/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/brighteyes-ism.svg?color=green)](https://pypi.org/project/brighteyes-ism/)
[![Python Version](https://img.shields.io/pypi/pyversions/brighteyes-ism.svg?color=green)](https://python.org)
<!--
[![tests](https://github.com/VicidominiLab/napari-ISM/workflows/tests/badge.svg)](https://github.com/VicidominiLab/napari-ISM/actions)
[![codecov](https://codecov.io/gh/VicidominiLab/napari-ISM/branch/main/graph/badge.svg)](https://codecov.io/gh/VicidominiLab/napari-ISM)
-->


A toolbox for analysing and simulating Image Scanning Microscopy (ISM) datasets.
The analysis module contains libraries for:

* Adaptive Pixel Reassignment (https://doi.org/10.1038/s41592-018-0291-9)
* Focus-ISM (https://doi.org/10.1038/s41467-022-35333-y)
* Image Deconvolution (https://doi.org/10.1088/1361-6420/accdc5)
* Fourier Ring Correlation (https://doi.org/10.1038/s41467-019-11024-z)

The simulation module contains libraries for:

* Generation of ISM point spread functions (https://doi.org/10.1016/j.cpc.2022.108315)
* Generation of tubulin phantom samples

The dataio module contains libraries for:

* Reading the data and metadata from the MCS software (https://github.com/VicidominiLab/BrightEyes-MCS)

----------------------------------

## Installation

You can install `brighteyes-ism` via [pip] directly from GitHub:

    pip install git+https://github.com/VicidominiLab/BrightEyes-ISM

or using the version on [PyPI]:

    pip install brighteyes-ism

It requires the following Python packages

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
	matplotlib-scalebar

## Documentation

You can find some examples of usage here:

https://github.com/VicidominiLab/BrightEyes-ISM/tree/main/examples

You can read the manual of this package on Read the Docs:

https://brighteyes-ism.readthedocs.io

## Citation

If you find BrightEyes-ISM useful for your research, please cite it as:

_Zunino, A., Slenders, E., Fersini, F. et al. Open-source tools enable accessible and advanced image scanning microscopy data analysis. Nat. Photon. (2023). https://doi.org/10.1038/s41566-023-01216-x_


## Contributing

Contributions are very welcome. Tests can be run with [tox], please ensure
the coverage at least stays the same before you submit a pull request.

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"BrightEyes-ISM" is free and open source software

## Issues

If you encounter any problems, please [file an issue] along with a detailed description.

[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt

[file an issue]: https://github.com/VicidominiLab/brighteyes-ism/issues

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/project/brighteyes-ism/
