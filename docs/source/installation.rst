.. _installation:

Installation
============

Overview
--------

Install this maintained v1 build directly from the fork's source tree. The
package named ``chemprop`` on PyPI is the upstream project and does not contain
the maintenance fixes and feature generators documented here.

.. note::
   The Kuroki-maintained v1 build is available from this fork's source tree,
   not from the upstream ``chemprop`` package on PyPI.

Conda
-----

Both options require conda, so first install Miniconda from `<https://conda.io/miniconda.html>`_.

The committed environment targets Python 3.10 and installs the official
PyTorch 2.6.0 CUDA 12.4 wheel. A CUDA 12.5-capable NVIDIA driver is backward
compatible with this cu124 runtime.

Installing from source
----------------------

1. :code:`git clone https://github.com/Kuroki-A/chemprop.git`
2. :code:`cd chemprop`
3. :code:`conda env create -f environment.yml`
4. :code:`conda activate chemprop310-cu124`
5. :code:`python -m pip check`
6. :code:`python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`

The expected GPU check is :code:`2.6.0+cu124`, :code:`12.4`, and
:code:`True`. The editable Chemprop installation is already performed by the
environment file.

Docker
------

Chemprop can also be installed with Docker. Docker makes it possible to isolate the Chemprop code and environment. To install and run our code in a Docker container, follow these steps:

1. :code:`git clone https://github.com/Kuroki-A/chemprop.git`
2. :code:`cd chemprop`
3. Install Docker from `<https://docs.docker.com/install/>`_
4. :code:`docker build -t chemprop .`
5. :code:`docker run -it chemprop:latest`

Note that you will need to run the latter command with nvidia-docker if you are on a GPU machine in order to be able to access the GPUs.
Alternatively, with Docker 19.03+, you can specify the :code:`--gpus` command line option instead.

The container CUDA runtime must be compatible with the host NVIDIA driver.
This repository uses the PyTorch cu124 wheel rather than a conda
:code:`cudatoolkit` dependency.
