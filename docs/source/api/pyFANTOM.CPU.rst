CPU Backend
===========

The CPU backend provides all core functionality for topology optimization using CPU-based computations.

.. automodule:: pyFANTOM.CPU
   :noindex:

Mesh Classes
------------

* :class:`pyFANTOM.geom.CPU._mesh.StructuredMesh2D` - 2D structured mesh
* :class:`pyFANTOM.geom.CPU._mesh.StructuredMesh3D` - 3D structured mesh
* :class:`pyFANTOM.geom.CPU._mesh.GeneralMesh` - General unstructured mesh

Filter Classes
--------------

* :class:`pyFANTOM.geom.CPU._filters.StructuredFilter2D` - 2D structured density filter
* :class:`pyFANTOM.geom.CPU._filters.StructuredFilter3D` - 3D structured density filter
* :class:`pyFANTOM.geom.CPU._filters.GeneralFilter` - General density filter

Finite Element Classes
----------------------

* :class:`pyFANTOM.FiniteElement.CPU.FiniteElement.FiniteElement` - Finite element analysis engine
* :class:`pyFANTOM.FiniteElement.CPU.NLFiniteElement.NLFiniteElement` - Nonlinear finite element analysis engine

Stiffness Kernel Classes
------------------------

* :class:`pyFANTOM.stiffness.CPU._FEA.StructuredStiffnessKernel` - Structured mesh stiffness kernel
* :class:`pyFANTOM.stiffness.CPU._FEA.GeneralStiffnessKernel` - General mesh stiffness kernel
* :class:`pyFANTOM.stiffness.CPU._FEA.UniformStiffnessKernel` - Uniform element stiffness kernel

Solver Classes
--------------

* :class:`pyFANTOM.solvers.CPU._solvers.CHOLMOD` - CHOLMOD direct solver
* :class:`pyFANTOM.solvers.CPU._solvers.CG` - Conjugate Gradient iterative solver
* :class:`pyFANTOM.solvers.CPU._solvers.BiCGSTAB` - BiCGSTAB iterative solver
* :class:`pyFANTOM.solvers.CPU._solvers.GMRES` - GMRES iterative solver
* :class:`pyFANTOM.solvers.CPU._solvers.SPLU` - Sparse LU direct solver
* :class:`pyFANTOM.solvers.CPU._solvers.SPSOLVE` - Sparse direct solver
* :class:`pyFANTOM.solvers.CPU._solvers.MultiGrid` - Multigrid iterative solver

Optimizer Classes
-----------------

* :class:`pyFANTOM.Optimizers.CPU.MMA.MMA` - Method of Moving Asymptotes optimizer
* :class:`pyFANTOM.Optimizers.CPU.OC.OC` - Optimality Criteria optimizer
* :class:`pyFANTOM.Optimizers.CPU.PGD.PGD` - Projected Gradient Descent optimizer

Problem Classes
---------------

* :class:`pyFANTOM.Problem.CPU.MinimumCompliance.MinimumCompliance` - Minimum compliance problem
* :class:`pyFANTOM.Problem.CPU.MinimumComplianceNL.MinimumComplianceNL` - Nonlinear minimum compliance problem
* :class:`pyFANTOM.Problem.CPU.ComplianceConstrainedMinimumVolume.ComplianceConstrainedMinimumVolume` - Volume-constrained compliance problem
* :class:`pyFANTOM.Problem.CPU.WeightDistributionMinimumCompliance.WeightDistributionMinimumCompliance` - Weight distribution problem

Detailed Documentation
----------------------

.. autoclass:: pyFANTOM.geom.CPU._mesh.StructuredMesh2D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CPU._mesh.StructuredMesh3D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CPU._mesh.GeneralMesh
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CPU._filters.StructuredFilter2D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CPU._filters.StructuredFilter3D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.geom.CPU._filters.GeneralFilter
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.FiniteElement.CPU.FiniteElement.FiniteElement
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.FiniteElement.CPU.NLFiniteElement.NLFiniteElement
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.stiffness.CPU._FEA.StructuredStiffnessKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.stiffness.CPU._FEA.GeneralStiffnessKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.stiffness.CPU._FEA.UniformStiffnessKernel
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CPU._solvers.CHOLMOD
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CPU._solvers.CG
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CPU._solvers.BiCGSTAB
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CPU._solvers.GMRES
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CPU._solvers.SPLU
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CPU._solvers.SPSOLVE
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.solvers.CPU._solvers.MultiGrid
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Optimizers.CPU.MMA.MMA
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Optimizers.CPU.OC.OC
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Optimizers.CPU.PGD.PGD
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Problem.CPU.MinimumCompliance.MinimumCompliance
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Problem.CPU.MinimumComplianceNL.MinimumComplianceNL
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Problem.CPU.ComplianceConstrainedMinimumVolume.ComplianceConstrainedMinimumVolume
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyFANTOM.Problem.CPU.WeightDistributionMinimumCompliance.WeightDistributionMinimumCompliance
   :members:
   :undoc-members:
   :show-inheritance:
