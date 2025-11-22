.. pyFANTOM documentation master file

Welcome to pyFANTOM!
====================

**FANTOM**: **Finite-element ANalysis and Topology Optimization Module**

pyFANTOM is a fast, efficient, GPU and CPU-ready general topology optimization package. Built with an object-oriented design, pyFANTOM enables you to perform finite-element based topology optimization with ease and flexibility.

What is pyFANTOM?
-----------------

pyFANTOM is a general-purpose topology optimization framework designed for flexibility and performance. The package provides:

- **Dual Backend Support**: Run your optimization problems on both CPU and GPU (CUDA) with the same API
- **Multiple Mesh Types**: Support for structured 2D/3D meshes and unstructured meshes
- **Efficient Solvers**: Including CHOLMOD, CG, GMRES, and a custom MultiGrid solver optimized for topology optimization
- **Scalability**: Solve mega-voxel problems (8M+ elements) on GPUs with 16GB VRAM thanks to matrix-free multi-grid implementation
- **Flexible Physics**: Built-in linear elasticity support with extensible physics models
- **Interactive Visualization**: 3D interactive visualizations using K3D for exploring your optimized topologies

Key Features
------------

**High Performance**
  - GPU acceleration via CUDA backend
  - Matrix-free multi-grid solver for large-scale problems
  - Efficient sparse solvers (CHOLMOD, CG, GMRES, MultiGrid)
  - Solve mega-voxel problems (8M+ elements) on 16GB GPUs

**Flexible & Extensible**
  - Object-oriented design for easy customization
  - Support for custom physics models
  - Multiple optimization algorithms (PGD, MMA, OC)
  - Independent components (meshes, solvers, optimizers)

**Multiple Problem Types**
  - Structured 2D and 3D meshes
  - Unstructured meshes via pygmsh
  - CPU and GPU backends with the same API
  - Interactive 3D visualizations

Quick Start
-----------

Ready to get started? Check out our examples:

- :doc:`2D Topology Optimization <examples/2d_topology_optimization>` - Classic 2D structured meshes with CPU and GPU examples
- :doc:`3D Topology Optimization <examples/3d_topology_optimization>` - 3D structured and unstructured mesh examples
- :doc:`Mega Voxel Optimization <examples/mega_voxel_topology_optimization>` - Solve massive 8M element problems on GPU

Or jump straight to the :doc:`Installation <getting_started/installation>` guide and :doc:`Introduction Tutorial <getting_started/introduction>` for a complete walkthrough.

A Simple Example
----------------

Here's a quick taste of what pyFANTOM can do:

.. code-block:: python

   from pyFANTOM.CPU import (
       StructuredMesh2D, StructuredStiffnessKernel, CHOLMOD,
       FiniteElement, StructuredFilter2D, MinimumCompliance, PGD
   )
   from pyFANTOM.Physics import LinearElasticity

   # Setup physics and mesh
   physics = LinearElasticity(E=1.0, nu=1/3, thickness=1.0, type='PlaneStress')
   mesh = StructuredMesh2D(nx=256, ny=64, lx=4.0, ly=1.0, physics=physics)

   # Setup solver and finite element analysis
   kernel = StructuredStiffnessKernel(mesh=mesh)
   solver = CHOLMOD(kernel=kernel)
   FE = FiniteElement(mesh=mesh, kernel=kernel, solver=solver)

   # Apply boundary conditions and setup optimization problem
   filter = StructuredFilter2D(mesh=mesh, r_min=1.5)
   problem = MinimumCompliance(FE=FE, filter=filter, volume_fraction=[0.4], penalty=3.0)
   optimizer = PGD(problem=problem, change_tol=1e-4, fun_tol=1e-6)

   # Run optimization
   for i in range(200):
       optimizer.iter()
       if optimizer.converged():
           break

   # Visualize result
   problem.visualize_solution()

See the :doc:`Getting Started <getting_started/introduction>` guide for the complete example with boundary conditions and detailed explanations.


Documentation Table of Contents
--------------------------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/installation
   getting_started/introduction

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/2d_topology_optimization
   examples/3d_topology_optimization
   examples/mega_voxel_topology_optimization

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   api/pyFANTOM.CPU
   api/pyFANTOM.CUDA
   api/pyFANTOM.Physics

.. toctree::
   :maxdepth: 1
   :caption: Other Sections

   citing
   contributing