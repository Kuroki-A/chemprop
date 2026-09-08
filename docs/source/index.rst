Chemprop
========

.. note::
   These pages accompany the Kuroki-maintained Chemprop v1 compatibility
   branch. The package version is ``1.7.1+kuroki.1`` and the source repository
   is `<https://github.com/Kuroki-A/chemprop>`_.

.. note:: The `fork README <https://github.com/Kuroki-A/chemprop/blob/master/README.md>`_ and `args.py <https://github.com/Kuroki-A/chemprop/blob/master/chemprop/args.py>`_ are the authoritative references for maintenance-only additions.

`Chemprop <https://github.com/Kuroki-A/chemprop>`_ is a message passing neural network for molecular property prediction.

At its core, Chemprop contains a directed message passing neural network (D-MPNN), which was first presented in `Analyzing Learned Molecular Representations for Property Prediction <https://pubs.acs.org/doi/abs/10.1021/acs.jcim.9b00237>`_. The Chemprop D-MPNN shows strong molecular property prediction capabilities across a range of properties, from quantum mechanical energy to human toxicity.

Chemprop was later used in the paper `A Deep Learning Approach to Antibiotic Discovery <https://www.cell.com/cell/fulltext/S0092-8674(20)30102-1>`_ to discover promising new antibiotics by predicting the likelihood that a molecule would inhibit the growth of *E. coli*.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   requirements
   installation
   tutorial
   web
   data
   features
   models
   train
   hyperopt
   interpret
   args
   nn_utils
   utils
   sklearn
   scripts


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
