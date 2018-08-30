# GPdoemd
[![Build Status](https://travis-ci.org/cog-imperial/GPdoemd.svg?branch=master)](https://travis-ci.org/cog-imperial/GPdoemd/branches) [![codecov](https://codecov.io/gh/cog-imperial/GPdoemd/branch/master/graph/badge.svg)](https://codecov.io/gh/cog-imperial/GPdoemd/branch/master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

Python package based on the following paper presented at ICML 2018  
[**Design of experiments for model discrimination using Gaussian process surrogate models.**](http://proceedings.mlr.press/v80/olofsson18a.html)
```
@inproceedings{Olofsson2018a,
  author    = {Simon Olofsson and Marc Peter Deisenroth and Ruth Misener},
  title     = {Design of experiments for model discrimination hybridising analytical and data-driven approaches},
  booktitle = {ICML '18: Proceedings of the International Conference on Machine Learning},
  address   = {Stockholm, Sweden},
  year      = {2018},
}
```

## Installation
The GPdoemd package has been tested and validated on OSX and Ubuntu.  
No guarantees are provided that GPdoemd works on Windows-based systems.

##### Requirements
Python 3.4+
* numpy >= 1.7
* scipy >= 0.17
* [GPy](https://github.com/SheffieldML/GPy)

##### Creating a virtual environment
We recommend installing GPdoemd in a virtual environment.  
To set up a new virtual environment called myenv (example name), run the command
```
python3 -m venv myenv
```
in the folder where you want to store the virtual environment.  
After the virtual environment has been created, activate it as follows
```
source myenv/bin/activate
```
It is recommended that you update the pip installation in the virtual environment
```
pip install --upgrade pip
```

##### Installing GPdoemd
First install all required packages. They are listed above and in the file [requirements.txt](https://github.com/cog-imperial/GPdoemd/blob/master/requirements.txt).  
To install GPdoemd, run the following in the virtual environment
```
pip install git+https://github.com/cog-imperial/GPdoemd
```
It is also possible to clone into/download the GPdoemd git repository and install it using setup.py, but this is not recommended for most users.

##### Uninstalling GPdoemd
The GPdoemd package can be uninstalled by running
```
pip uninstall GPdoemd
```
Alternatively, the folder containing the virtual environment can be deleted.

## Authors
* **[Simon Olofsson](https://www.doc.ic.ac.uk/~so2015/)** ([scwolof](https://github.com/scwolof)) - Imperial College London

## License
The GPdoemd package is released under the MIT License. Please refer to the [LICENSE](https://github.com/cog-imperial/GPdoemd/blob/master/LICENSE) file for details.

## Acknowledgements
This work has received funding from the European Union's Horizon 2020 research and innovation programme under the Marie Skłodowska-Curie grant agreement no.675251.

