# XAIMol

## Installation

BBOMol depends on [EvoMol](https://doi.org/10.1186/s13321-020-00458-z). Follow first the installation steps described 
on <a href='https://github.com/jules-leguy/evomol'>EvoMol repository</a>. 

XAIMol installation

Finally, type the following commands to install [ChemDesc](https://github.com/jules-leguy/ChemDesc), a dependency that 
is required to compute the molecular descriptors.

```shell script
$ cd ..                                                   # Go back to the previous directory if you are still in the BBOMol installation directory
$ git clone https://github.com/jules-leguy/ChemDesc.git   # Clone ChemDesc
$ cd chemdesc                                             # Move into ChemDesc directory
$ conda activate evomolenv                                # Activate evomolenv environment
$ python -m pip install .                                 # Install ChemDesc
```