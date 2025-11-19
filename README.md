# pyFANTOM: Fast, Efficient, GPU and CPU ready general topology optimization

**FANTOM**: **Finite-element ANalysis and TOpology Methods**

[![Documentation Status](https://readthedocs.org/projects/pyfantom/badge/?version=latest)](https://pyfantom.readthedocs.io/en/latest/?badge=latest)

pyFANTOM is a general package for topology optimization built for Finite-Elemente (FE) based topology optimizaiton. All features are built for general purpose use cases with object oriented setup enablig customization and adapting to different problems.

📖 **Documentation**: https://pyfantom.readthedocs.io/

The package by default includes physiocs for linear elasticity, with future releases planned to include other physics. The package also comes with the minimum compliance problem predefined, however, optimizers, meshes, and FE features are all independant of this and can be cutomized sperately as needed.

## Installation

### Basic Installation

Install pyFANTOM in editable mode (recommended for development):

```bash
pip install -e .
```

Or install as a regular package:

```bash
pip install .
```

### CUDA Support (Optional)

To install with CUDA support, include the `cuda` extra:

```bash
pip install -e .[cuda]
```

Note: You may need to install a specific CuPy version for your CUDA toolkit (e.g., `cupy-cuda12x`). See [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) for details.

### Alternative: Manual Dependency Installation

You can also install dependencies manually from `requirements.txt`:

```bash
pip install -r requirements.txt
```

### MKL-Optimized Builds (Advanced)

We recommend installing packages with MKL compiled wheels if you are going to be using Intel CPUs. This makes a notable difference in performance. If you already have CHOLMOD compiled with MKL you can simply run the `env_setup.sh` script to install the packages with MKL wheels. Please refer to SuiteSparse documentation for compiling CHOLMOD with MKL.

```bash
bash env_setup.sh
```

We also provide precompiled wheels of scikit-sparse with MKL enabled CHOLMOD. This is only a many-linux wheel however, so on windows or mac one still need to compile with MKL compiled CHOLMOD. If you are on linux and want to use this wheel you can setup the packages using `env_setup_from_wheel.sh`.

```bash
bash env_setup_from_wheel.sh
```