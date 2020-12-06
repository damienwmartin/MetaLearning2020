# MetaLearning2020

This repository contains the code used for our MIT 6.883 Meta-Learning Fall 2020 course project.

It includes a python-only implementation of the ReBeL algorithm, along with our proposed ReBeL 2.0.
The intention was to create a clean and easy-to-modify implementation for
research purposes.

## Installation

### Requirements
This code base was run and tested on **Ubuntu 20.04**. MacOS and Windows are not
officially supported yet.

The code requires the following packages.
```
pytorch
jupyterlab
numpy
matplotlib
networkx
unittest
```

### Installation instructions
To install:

1. Clone this repository to your local machine and.
```
 $ git clone https://github.com/damienwmartin/MetaLearning2020.git
```

2. Follow the instructions to install `pytorch` listed on their <a href='https://pytorch.org/get-started/locally/' target='blank'>website</a>.

3. Install other dependencies using

pip:
```
 $ pip install -r requirements.txt
```

or conda:
```
 $ conda install -c conda-forge -c anaconda --file requirements.txt
```

## File structure
The main implementation code is found in the `Rebel/` folder.

The `experiments/` folder contains jupyter notebooks for replicating the experiments in our project report.

The `tests\` folder contains test cases for confirming successful installation and that the implementation is working correctly.

More details about the implementation can be found [here](https://github.com/damienwmartin/MetaLearning2020/tree/main/Rebel#python-rebel-implementation)

## References
Original rebel paper can be found [here](https://arxiv.org/abs/2007.13544) and repository [here](https://github.com/facebookresearch/rebel)
