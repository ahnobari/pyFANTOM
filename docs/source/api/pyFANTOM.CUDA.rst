CUDA Backend
============

The CUDA backend provides GPU-accelerated implementations of all core functionality for topology optimization.

.. automodule:: pyFANTOM.CUDA
   :noindex:

Mesh Classes
------------

* :class:`pyFANTOM.geom.CUDA._mesh.CuStructuredMesh2D` - CUDA 2D structured mesh
* :class:`pyFANTOM.geom.CUDA._mesh.CuStructuredMesh3D` - CUDA 3D structured mesh
* :class:`pyFANTOM.geom.CUDA._mesh.CuGeneralMesh` - CUDA general unstructured mesh

Filter Classes
--------------

* :class:`pyFANTOM.geom.CUDA._filters.CuStructuredFilter2D` - CUDA 2D structured density filter
* :class:`pyFANTOM.geom.CUDA._filters.CuStructuredFilter3D` - CUDA 3D structured density filter
* :class:`pyFANTOM.geom.CUDA._filters.CuGeneralFilter` - CUDA general density filter

Finite Element Classes
----------------------

* :class:`pyFANTOM.FiniteElement.CUDA.FiniteElement.FiniteElement` - CUDA finite element analysis engine

Stiffness Kernel Classes
------------------------

* :class:`pyFANTOM.stiffness.CUDA._FEA.StructuredStiffnessKernel` - CUDA structured mesh stiffness kernel
* :class:`pyFANTOM.stiffness.CUDA._FEA.GeneralStiffnessKernel` - CUDA general mesh stiffness kernel
* :class:`pyFANTOM.stiffness.CUDA._FEA.UniformStiffnessKernel` - CUDA uniform element stiffness kernel

Solver Classes
--------------

* :class:`pyFANTOM.solvers.CUDA._solvers.CG` - CUDA Conjugate Gradient iterative solver
* :class:`pyFANTOM.solvers.CUDA._solvers.GMRES` - CUDA GMRES iterative solver
* :class:`pyFANTOM.solvers.CUDA._solvers.SPSOLVE` - CUDA sparse direct solver
* :class:`pyFANTOM.solvers.CUDA._solvers.MultiGrid` - CUDA multigrid iterative solver

Optimizer Classes
-----------------

* :class:`pyFANTOM.Optimizers.CUDA.MMA.MMA` - CUDA Method of Moving Asymptotes optimizer
* :class:`pyFANTOM.Optimizers.CUDA.OC.OC` - CUDA Optimality Criteria optimizer
* :class:`pyFANTOM.Optimizers.CUDA.PGD.PGD` - CUDA Projected Gradient Descent optimizer

Problem Classes
---------------

* :class:`pyFANTOM.Problem.CUDA.MinimumCompliance.MinimumCompliance` - CUDA minimum compliance problem
* :class:`pyFANTOM.Problem.CUDA.ComplianceConstrainedMinimumVolume.ComplianceConstrainedMinimumVolume` - CUDA volume-constrained compliance problem
* :class:`pyFANTOM.Problem.CUDA.WeightDistributionMinimumCompliance.WeightDistributionMinimumCompliance` - CUDA weight distribution problem

Detailed Documentation
----------------------

.. autoclass:: pyFANTOM.geom.CUDA._mesh.CuStructuredMesh2D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CUDA._mesh.CuStructuredMesh3D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CUDA._mesh.CuGeneralMesh
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CUDA._filters.CuStructuredFilter2D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CUDA._filters.CuStructuredFilter3D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CUDA._filters.CuGeneralFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.FiniteElement.CUDA.FiniteElement.FiniteElement
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.stiffness.CUDA._FEA.StructuredStiffnessKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.stiffness.CUDA._FEA.GeneralStiffnessKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.stiffness.CUDA._FEA.UniformStiffnessKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CUDA._solvers.CG
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CUDA._solvers.GMRES
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CUDA._solvers.SPSOLVE
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CUDA._solvers.MultiGrid
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Optimizers.CUDA.MMA.MMA
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Optimizers.CUDA.OC.OC
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Optimizers.CUDA.PGD.PGD
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Problem.CUDA.MinimumCompliance.MinimumCompliance
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Problem.CUDA.ComplianceConstrainedMinimumVolume.ComplianceConstrainedMinimumVolume
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Problem.CUDA.WeightDistributionMinimumCompliance.WeightDistributionMinimumCompliance
   :members:
   :undoc-members:
   :show-inheritance:
