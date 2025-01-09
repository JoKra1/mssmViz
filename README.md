# mssmViz

![Docs](https://github.com/jokra1/mssm_tutorials/actions/workflows/documentation.yml/badge.svg?branch=main)

## Description

Plotting functions for the Mixed Sparse Smooth Models ([mssm](https://github.com/JoKra1/mssm)) toolbox. ``mssm`` is a toolbox to estimate Generalized Additive Mixed Models (GAMMs), Generalized Additive Mixed Models of Location Scale and Shape (GAMMLSS), and even more general smooth models in the sense defined by [Wood, Pya, & Säfken (2016)](https://doi.org/10.1080/01621459.2016.1180986). In addition, a tutorial for ``mssm`` is provided with this repository. **Documentation** for ``mssmViz``  is hosted [here](https://jokra1.github.io/mssm_tutorials/index.html).

## Installation

To install ``mssm`` simply run:

```
conda create -n mssm_env python=3.11
conda activate mssm_env
pip install mssm
```

Subsequently, ``mssmViz`` can be installed by running:

```
pip install mssmViz
```

Alternatively, you can clone the repository into a folder of your choice:

```
git clone https://github.com/JoKra1/mssm_tutorials.git
```

After navigating to the folder into which you cloned this repository, you can then install `mssmViz` plot functions
by running:

```
pip install -e .
```

The -e flag will ensure that any new changes you pull from this repository will be reflected when you use the plot functions.