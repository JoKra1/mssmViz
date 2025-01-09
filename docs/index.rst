.. mssmViz documentation master file, created by
   sphinx-quickstart on Thu Jan  9 18:52:26 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mssmViz documentation
=====================

This is the documentation of ``mssmViz``. The entire source code is available for inspection on `GitHub <https://github.com/JoKra1/mssm_tutorials>`_.
``mssmViz`` contains code to extract information from (:mod:`mssmViz.src.extract`), to visualize, & and to validate (:mod:`mssmViz.src.plot`) smooth models estimated via the `mssm <https://github.com/JoKra1/mssm>`_ Python toolbox for estimating Generalized Additive Mixed Models (GAMMs), Generalized Additive Mixed Models of Location Scale
and Shape (GAMMLSS), and more general (mixed) smooth models in the sense defined by `Wood, Pya, & Säfken (2016) <https://doi.org/10.1080/01621459.2016.1180986>`_.

Approximate estimation (and automatic regularization) of the latter only requires users to provide the (gradient of) the log-likelihood.
Furthermore, ``mssm`` is an excellent choice for the modeling of multi-level time-series data, often estimating additive models with separate smooths for thousands of levels in a couple of minutes.

Use the side-bar on the left to navigate through the document tree.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules