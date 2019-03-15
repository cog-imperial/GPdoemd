# GPdoemd
[![Build Status](https://travis-ci.org/cog-imperial/GPdoemd.svg?branch=master)](https://travis-ci.org/cog-imperial/GPdoemd/branches) [![codecov](https://codecov.io/gh/cog-imperial/GPdoemd/branch/master/graph/badge.svg)](https://codecov.io/gh/cog-imperial/GPdoemd/branch/master) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python package based on the following papers  
[**Design of experiments for model discrimination using Gaussian process surrogate models**](http://proceedings.mlr.press/v80/olofsson18a.html)  
[**GPdoemd: A Python package for design of experiments for model discrimination**](https://www.sciencedirect.com/science/article/pii/S0098135419300468)  
Please reference this software package as
```
@article{GPdoemd,
  author  = {Simon Olofsson and Lukas Hebing and Sebastian Niedenf\"uhr and Marc Peter Deisenroth and Ruth Misener},
  title   = {{GP}doemd: A {P}ython package for design of experiments for model discrimination},
  journal = {Computers \& Chemical Engineering},
  volume  = {125},
  pages   = {54--70},
  year    = {2019},
}
```

## Background
Here we provide a brief introduction to design of experiments for model discrimination, and the method used in the GPdoemd package. For more information, we refer to the paper referenced above.

##### Design of experiments for model discrimination
We are interested in some system _g_ (e.g. the human body, a bioreactor, or a chemical reaction), from which we can generate data _D_ constisting of noisy observations **y**=_g_(**x**) given experimental designs **x**. To predict the behaviour of the system, the engineers and researchers come up with several competing models _f_<sub>i</sub>, i=1,...,M. The models produce predictions _f_<sub>i</sub>(**x**, **p**<sub>i</sub>) given some model parameters **p**<sub>i</sub>. In a classical setting, these model parameters are tuned to make the model predictions fit the observed data.

If we are in a setting where we have multiple rival models and insufficient data to discriminate between them (i.e. to discard inaccurate models), we need to design additional experiments **x** to collect more data. The goal is to find the experimental design **x**\* that yields the expected maximally informative observations for discriminating between the models. Simply put, we want to find the experiment for which the model predictions differ the most. However, because the model parameters are estimated using noisy data, there is uncertainty in the model parameters that needs to be accounted for. To this end, we wish to compute the marginal predictive distributions _p_(_f_<sub>i</sub>(**x**)|_D_) where the model parameters **p**<sub>i</sub> have been marginalised out. Computing the marginal predictive distributions is typically intractable, so we rely on approximations.

There are existing methods to approximate the marginal predictive distributions. Roughly, these can be dividied into analytic methods (computationally cheap, but limited to certain models) and data-driven methods (Monte Carlo-based, flexible but often computationally expensive). In the paper references above, and in this software package, the idea is to use an approach that hybridises analytic and data-driven methods, using analytic surrogate models learnt from sampled data. This way we can extend the computationally cheap analytic method to a wider range of models.

## Installation
The following instructions work for OSX and Ubuntu systems.  
For installation on a Windows system, please refer to the file [windows_install.md](https://github.com/cog-imperial/GPdoemd/blob/dev/windows_install.md). 

##### Requirements
Python 3.4+
* numpy >= 1.7
* scipy >= 0.17
* [GPy](https://github.com/SheffieldML/GPy)

##### Optional
* gp_grief ([forked repository](https://github.com/scwolof/gp_grief)): GP-GRIEF surrogate models

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
First install all required packages in the virtual environment.  
The required packages are listed above and in the file [requirements.txt](https://github.com/cog-imperial/GPdoemd/blob/master/requirements.txt).  
```
pip install numpy scipy six paramz matplotlib
pip install GPy
```
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

