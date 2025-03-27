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
* Image and ISM datasets visualization
* Miscellaneous tools

The simulation module contains libraries for:

* Generation of ISM point spread functions (https://doi.org/10.48550/arXiv.2502.03170)
* Generation of tubulin phantom samples

The dataio module contains libraries for:

* Reading the data and metadata from the MCS software (https://doi.org/10.21105/joss.07125)

----------------------------------

## Installation

You can install `brighteyes-ism` via [pip] directly from GitHub:

    pip install git+https://github.com/VicidominiLab/BrightEyes-ISM

or using the version on [PyPI]:

    pip install brighteyes-ism

It requires the following Python packages

    numpy
    scipy
    scikit-image
    scikit-learn
    matplotlib
    joblib
    tqdm
    h5py
    statsmodels
    matplotlib-scalebar
    torch
    torchvision
    zernikepy
    psf-generator

## Documentation

You can find some examples of usage here:

https://github.com/VicidominiLab/BrightEyes-ISM/tree/main/examples

You can read the manual of this package on Read the Docs:

https://brighteyes-ism.readthedocs.io

## Citation

If you find BrightEyes-ISM useful for your research, please cite it as:

_Zunino, A., Slenders, E., Fersini, F. et al. Open-source tools enable accessible and advanced image scanning microscopy data analysis. Nat. Photon. (2023). https://doi.org/10.1038/s41566-023-01216-x_

## License

Distributed under the terms of the [GNU GPL v3.0] license,
"BrightEyes-ISM" is free and open source software

## Contributing

You want to contribute? Great!
Contributing works best if you creat a pull request with your changes.

1. Fork the project.
2. Create a branch for your feature: `git checkout -b cool-new-feature`
3. Commit your changes: `git commit -am 'My new feature'`
4. Push to the branch: `git push origin cool-new-feature`
5. Submit a pull request!

If you are unfamilar with pull requests, you find more information on pull requests in the
 [github help](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-pull-requests)

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
