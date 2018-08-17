# GPdoemd
[![Build Status](https://travis-ci.org/cog-imperial/GPdoemd.svg?branch=master)](https://travis-ci.org/cog-imperial/GPdoemd/branches) [![codecov](https://codecov.io/gh/cog-imperial/GPdoemd/branch/master/graph/badge.svg)](https://codecov.io/gh/cog-imperial/GPdoemd/branch/master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[**Design of experiments for model discrimination using Gaussian process surrogate models.**](http://proceedings.mlr.press/v80/olofsson18a.html)

## Installation
The GPdoemd package has been tested and validated on OSX and Ubuntu. No guarantees are provided that GPdoemd works on Windows-based systems.

##### Requirements
Python 3.4+
* numpy >= 1.7
* scipy >= 0.17
* [GPy](https://github.com/SheffieldML/GPy)

##### Creating a virtual environment
We recommend installing GPdoemd in a virtual environment. To set up a new virtual environment called myenv (example name), run the command
```
python3 -m venv myenv
```
in the folder where you want to store the virtual environment. After the virtual environment has been created, activate it as follows
```
source myenv/bin/activate
```
It is recommended that you update the pip installation in the virtual environment
```
pip install --upgrade pip
```

##### Installing GPdoemd
To install GPdoemd, first install all required packages. They are listed above and in the file requirements.txt. Then run the following
```
pip install git+https://github.com/cog-imperial/GPdoemd
```
This should also take care of installing the required packages. 

It is also possible to clone into the GPdoemd git repository and install it using setup.py, but this is not recommended for most users.

##### Uninstalling GPdoemd
The GPdoemd package can be uninstalled by running
```
pip uninstall GPdoemd
```
Alternatively, the folder containing the virtual environment can be deleted. This will remove the entire virtual environment in one go. There is no need to uninstall all packages in the virtual environment prior to deleting it.

If setup.py has been used to install GPdoemd, then good luck!

## Authors
* **[Simon Olofsson](https://www.doc.ic.ac.uk/~so2015/)** ([scwolof](https://github.com/scwolof)) - Imperial College London

## License
The GPdoemd package is released under the MIT License. Please refer to the [LICENSE](https://github.com/cog-imperial/GPdoemd/blob/master/LICENSE) file for details.

## Acknowledgements
This work has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no.675251.

