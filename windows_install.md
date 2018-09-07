
## Installation
The GPdoemd package has been tested and validated on OSX and Ubuntu.  
No guarantees are provided that GPdoemd works on Windows-based systems.

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
python -m venv myenv
```
in the folder where you want to store the virtual environment.  
After the virtual environment has been created, activate it as follows
```
myenv\Scripts\activate
```
It is recommended that you update the pip installation in the virtual environment
```
python -m pip install --upgrade pip
```

##### Installing GPdoemd
First install all required packages in the virtual environment.  
The required packages are listed above and in the file [requirements.txt](https://github.com/cog-imperial/GPdoemd/blob/master/requirements.txt).  
```
pip install numpy scipy six paramz matplotlib
```
Try running
```
pip install GPy
```
If you are lucky, this works. In our experience, the last command will fail - GPy will need to be built from source.
Download the files from the [GPy GitHub repository](https://github.com/SheffieldML/GPy). Enter the downloaded GPy folder and run
```
python setup.py
```
If the setup is successful, skip to the installation of GPdoemd.
If the previous command generates cython-related errors, try to comment out the lines in setup.py related to ext_mods, and add
```
ext_mods = []
```
After this, try running `python setup.py` again.

To install GPdoemd, run the following in the virtual environment
```
pip install git+https://github.com/cog-imperial/GPdoemd
```
Alternatively, if you do not have git installed, download the files from the GitHub repository and install using `python setup.py`.

##### Uninstalling GPdoemd
The GPdoemd package can be uninstalled by running
```
pip uninstall GPdoemd
```
Alternatively, the folder containing the virtual environment can be deleted.

