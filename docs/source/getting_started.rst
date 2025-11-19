Getting Started
===============

Welcome to pyFANTOM! This guide will help you get started with topology optimization using pyFANTOM.

Installation
------------

Basic Installation
~~~~~~~~~~~~~~~~~~

Install pyFANTOM in editable mode (recommended for development):

.. code-block:: bash

   pip install -e .

Or install as a regular package:

.. code-block:: bash

   pip install .

CUDA Support (Optional)
~~~~~~~~~~~~~~~~~~~~~~~

To install with CUDA support for GPU acceleration, include the ``cuda`` extra:

.. code-block:: bash

   pip install -e .[cuda]

Note: You may need to install a specific CuPy version for your CUDA toolkit (e.g., ``cupy-cuda12x``). See the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for details.

MKL-Optimized Builds (Advanced)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For better performance on Intel CPUs, we recommend using MKL-compiled wheels. If you already have CHOLMOD compiled with MKL, you can run:

.. code-block:: bash

   bash env_setup.sh

For Linux users, we also provide precompiled wheels with MKL-enabled CHOLMOD:

.. code-block:: bash

   bash env_setup_from_wheel.sh

Quick Start Example
-------------------

Here's a simple example to get you started with a 2D minimum compliance topology optimization problem:

.. code-block:: python

   import numpy as np
   from pyFANTOM.CPU import (
       StructuredMesh2D,
       StructuredStiffnessKernel,
       CHOLMOD,
       FiniteElement,
       StructuredFilter2D,
       MinimumCompliance,
       PGD
   )
   from matplotlib import pyplot as plt
   from tqdm.auto import trange
   from pyFANTOM.Physics import LinearElasticity

   # 1. Setup physics model
   physics = LinearElasticity(E=1.0, nu=1/3, thickness=1.0, type='PlaneStress')

   # 2. Create mesh
   mesh = StructuredMesh2D(nx=256, ny=64, lx=4.0, ly=1.0, physics=physics)

   # 3. Setup stiffness kernel and solver
   kernel = StructuredStiffnessKernel(mesh=mesh)
   solver = CHOLMOD(kernel=kernel)
   FE = FiniteElement(mesh=mesh, kernel=kernel, solver=solver)

   # 4. Apply boundary conditions (MBB beam)
   # Fix left edge
   left_nodes = np.where(mesh.nodes[:, 0] < 1e-6)[0]
   FE.add_dirichlet_boundary_condition(
       node_ids=left_nodes,
       dofs=np.array([[1, 0]]),  # Fix in x
       rhs=0.0
   )

   # fix bottom right corner
   bottom_right = np.where(
       np.logical_and(
           np.abs(mesh.nodes[:, 0] - 4.0) < 1e-6,
           np.abs(mesh.nodes[:, 1] - 0.0) < 1e-6
       )
   )[0]
   FE.add_dirichlet_boundary_condition(
       node_ids=bottom_right,
       dofs=np.array([[0, 1]]),  # Fix in y
       rhs=0.0
   )

   # Apply point load at top-left corner
   top_left = np.where(
       np.logical_and(
           np.abs(mesh.nodes[:, 0] - 0.0) < 1e-6,
           np.abs(mesh.nodes[:, 1] - 1.0) < 1e-6
       )
   )[0]
   FE.add_point_forces(
       node_ids=top_left,
       forces=np.array([[0.0, -1.0]])  # Downward force
   )

   # 5. Setup density filter
   filter = StructuredFilter2D(mesh=mesh, r_min=1.5)

   # 6. Create optimization problem
   problem = MinimumCompliance(
       FE=FE,
       filter=filter,
       volume_fraction=[0.4],
       penalty=3.0,
       penalty_schedule=lambda p, i: (
           1 if i < 40 else
           1 + (p - 1)/2 if i < 60 else
           p
       ),
   )

   # 7. Setup optimizer and run
   optimizer = PGD(problem=problem, change_tol=1e-4, fun_tol=1e-6)

   # Run optimization
   progress = trange(200)
   for i in progress:
       optimizer.iter()
       logs = optimizer.logs()
       
       progress.set_postfix(logs)

       if optimizer.converged():
           print("Optimization converged!")
           break

   # 8. Visualize result
   plt.figure(figsize=(10, 8))
   problem.visualize_solution()
   plt.axis('off')

Core Concepts
-------------

pyFANTOM follows an object-oriented design with the following main components:

1. **Physics Models** (:doc:`Physics Models <api/pyFANTOM.Physics>`)
   - Define material behavior and element-level computations
   - Examples: ``LinearElasticity``, ``NLElasticity``, ``SteadyHeatTransfer``

2. **Meshes** (:doc:`CPU Backend <api/pyFANTOM.CPU>` or :doc:`CUDA Backend <api/pyFANTOM.CUDA>`)
   - Define geometry and discretization
   - Examples: ``StructuredMesh2D``, ``StructuredMesh3D``, ``GeneralMesh``

3. **Finite Element Analysis (FEA)**
   - ``FiniteElement``: Manages boundary conditions, forces, and solution
   - ``StiffnessKernel``: Assembles stiffness matrices
   - ``Solver``: Solves linear systems (e.g., ``CHOLMOD``, ``CG``, ``MultiGrid``)

4. **Density Filters**
   - Ensure minimum feature sizes
   - Examples: ``StructuredFilter2D``, ``StructuredFilter3D``, ``GeneralFilter``

5. **Optimization Problems**
   - Define objective and constraints
   - Examples: ``MinimumCompliance``, ``ComplianceConstrainedMinimumVolume``

6. **Optimizers**
   - Solve optimization problems
   - Examples: ``MMA``, ``OC``, ``PGD``

Workflow Overview
-----------------

A typical topology optimization workflow in pyFANTOM follows these steps:

1. **Setup Physics**: Choose a physics model (e.g., linear elasticity)
2. **Create Mesh**: Define the geometry and discretization
3. **Setup FEA**: Configure stiffness kernel, solver, and finite element handler
4. **Apply BCs**: Set boundary conditions and loads
5. **Setup Filter**: Configure density filtering for manufacturability
6. **Define Problem**: Create an optimization problem (e.g., minimum compliance)
7. **Initialize**: Set initial design variables
8. **Optimize**: Run an optimizer (e.g., MMA) until convergence
9. **Visualize**: Plot and analyze the optimized design

CPU vs CUDA Backend
-------------------

pyFANTOM provides two backends with identical APIs:

- **CPU Backend** (``pyFANTOM.CPU``): Pure CPU implementation, works everywhere
- **CUDA Backend** (``pyFANTOM.CUDA``): GPU-accelerated, requires CUDA-capable GPU

To switch between backends, simply change your imports:

.. code-block:: python

   # CPU backend
   from pyFANTOM.CPU import StructuredMesh2D, FiniteElement, MinimumCompliance
   
   # CUDA backend (same API!)
   from pyFANTOM.CUDA import StructuredMesh2D, FiniteElement, MinimumCompliance

Next Steps
----------

- Check out the :doc:`Examples <examples>` for more detailed tutorials
- Explore the :doc:`API Reference <api/index>` for detailed class documentation
- See the :doc:`CPU Backend <api/pyFANTOM.CPU>` and :doc:`CUDA Backend <api/pyFANTOM.CUDA>` documentation for available classes
