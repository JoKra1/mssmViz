# mssmViz

## Description

Plotting functions for the Massive Smooth Models ([mssm](https://github.com/JoKra1/mssm)) toolbox. ``mssm`` is a toolbox to estimate Generalized Additive Mixed Models (GAMMs) and Generalized Additive Mixed Models of Location Scale and Shape (GAMMLSS). In addition, a tutorial for ``mssm`` is provided with this repository.

## Installation

To install ``mssm`` simply run:

```
conda create -n mssm_env python=3.11
conda activate mssm_env
pip install mssm
pip install matplotlib # Needed for tutorials
```

Subsequently, clone this tutorial repository into a folder of your choice:

```
git clone https://github.com/JoKra1/mssm_tutorials.git
```

After selecting the conda environment you just created as kernel and navigating to the folder into which you cloned this repository, you can install `mssmViz` plot functions
by running

```
pip install -e .
```

The -e flag will ensure that any new changes you pull from this repository will be reflected when you use the plot functions.