.. _requirements:

Requirements
============

For small datasets (~1000 molecules), it is possible to train models within a
few minutes on a standard laptop with CPUs only. However, for larger datasets
and larger Chemprop models, we recommend using a GPU for significantly faster
training.

This maintained v1 fork targets Python 3.10 and PyTorch 2.6.0. The committed
:code:`environment.yml` installs the official PyTorch CUDA 12.4 wheel; an
NVIDIA driver capable of CUDA 12.5 is backward compatible with that runtime.
The wheel supplies its CUDA user-space runtime, so a separate CUDA 12.4 toolkit
installation is not required for normal Chemprop use.

All models are built with `PyTorch <https://pytorch.org/>`_. See
:ref:`installation` for the reproducible installation procedure and GPU check.
