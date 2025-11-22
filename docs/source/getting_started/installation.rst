Installation
============

Basic Installation
------------------

Install pyFANTOM in editable mode (recommended for development):

.. code-block:: bash

   pip install -e .

Or install as a regular package:

.. code-block:: bash

   pip install .

CUDA Support (Optional)
------------------------

To install with CUDA support for GPU acceleration, include the ``cuda`` extra:

.. code-block:: bash

   pip install -e .[cuda]

Note: You may need to install a specific CuPy version for your CUDA toolkit (e.g., ``cupy-cuda12x``). See the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for details.

MKL-Optimized Builds (Advanced)
--------------------------------

For better performance on Intel CPUs, we recommend using MKL-compiled wheels. If you already have CHOLMOD compiled with MKL, you can run:

.. code-block:: bash

   bash env_setup.sh

For Linux users, we also provide precompiled wheels with MKL-enabled CHOLMOD:

.. code-block:: bash

   bash env_setup_from_wheel.sh
