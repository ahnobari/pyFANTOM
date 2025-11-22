Mega Voxel Topology Optimization Example
========================================

Solving Massive Mega-Voxel Problems On The GPU
-----------------------------------------------

Here we demonstrate how you can use the multi-grid solver in FANTOM to solve mega-voxel TO problems in 3D. We will use the bridge problem again and use a structured mesh with 8M elements!

You can run this with 16GB of GPU VRAM since FANTOM's multi-grid is matrix free at the highest level!

=============================
Importing The Necessary Tools
=============================

First we import the tools from pyFANTOM we will be using in this example.

.. code-block:: python

   from pyFANTOM.CUDA import(
       StructuredMesh3D, # Mesh Object for 3D Structured Meshes
       StructuredFilter3D, # Filter kernel for 3D Structured Meshes
       StructuredStiffnessKernel, # Stiffness Kernel for 3D Structured Meshes
       FiniteElement, # Finite Element Object to Setup Boundary Conditions and RHS
       CG, GMRES, MultiGrid, SPSOLVE, # Sparse Solvers
       MinimumCompliance, # TO Problem Definition
       PGD, MMA, OC # Nonlinear Optimizers
   )

   from pyFANTOM.Physics import LinearElasticity # Physics Model For FEA

   # Other Imports
   import numpy as np
   import matplotlib.pyplot as plt
   from tqdm import trange

===============================
Setting Up Physics and Geometry
===============================

The first step in using pyFANTOM is defining the domain of the problem and setting up the physics for FEA.

.. code-block:: python

   length = 1.0
   width = 0.25
   height = 0.25

   # Create a physics model
   physics = LinearElasticity(E=1.0, nu=1/3, thickness=1.0, type='PlaneStress')

   # Create a mesh with 8M elements (512 x 128 x 128)
   mesh = StructuredMesh3D(nx=512, ny=128, nz=128, lx=length, ly=height, lz=width, physics=physics)

====================================
Setting Up The Finite Element Solver
====================================

The next step in using pyFANTOM is setting up a stiffness kernel and finite-element boundary conditions. For mega-voxel problems, we use the MultiGrid solver with multiple levels to efficiently solve the large linear systems.

.. code-block:: python

   # Create a stiffness kernel
   stiffness_kernel = StructuredStiffnessKernel(mesh = mesh)

   # Create a solver
   # For mega-voxel problems, MultiGrid with multiple levels is essential
   solver = MultiGrid(
       mesh=mesh,
       kernel=stiffness_kernel,
       maxiter=100,
       tol=1e-4,
       n_level=6,  # 6 levels for the multi-grid hierarchy
       cycle='W',  # W-cycle for better convergence
       w_level=[2,3],  # Number of W-cycles at each level
       omega_boost=1.05  # Relaxation parameter boost
   )

   # Create a finite-element object
   fe = FiniteElement(mesh=mesh, kernel=stiffness_kernel, solver=solver)

   left_edge_nodes = np.where(mesh.nodes[:, 0] < 1e-12)[0]
   right_edge_nodes = np.where(mesh.nodes[:, 0] > length - 1e-12)[0]

   bottom_edge_nodes = np.where(mesh.nodes[:, 1] < 1e-12)[0]

   fe.reset_dirichlet_boundary_conditions()
   fe.reset_forces()
   # Apply force to the bottom edge
   fe.add_point_forces(
       node_ids=bottom_edge_nodes,
       positions=None,
       forces=np.array([[0.0, -1.0/len(bottom_edge_nodes), 0.0]])
   )

   # Apply zero displacement to the left edge and the right edge
   fe.add_dirichlet_boundary_condition(
       node_ids=np.concatenate([left_edge_nodes, right_edge_nodes]), # Node IDs to apply the boundary condition to
       positions=None, # Alternative way to specify the boundary condition is to pass positions in space
       dofs=np.array([[1, 1, 1]]), # Degrees of freedom to apply the boundary condition to
       rhs=0.0 # Value to apply the boundary condition to
   )

====================================
Setting Up TO Problem And Optimizer
====================================

Once the finite element is setup we can define the problem and optimizer to perform TO.

.. code-block:: python

   # Define the filter kernel for TO
   filter_kernel = StructuredFilter3D(mesh=mesh, r_min=1.5)

   # Define the TO problem
   to_problem = MinimumCompliance(
       FE=fe,
       filter=filter_kernel,
       E_mul=[1.0], # You can pass a list of values to perform multi-material TO
       volume_fraction=[0.05], # You can pass a list of values to perform volume fraction control for each material
       void=1e-9, # You can pass a value to set void modulus
       penalty=3, # You can pass a value to set penalty factor
       # penalty_schedule = lambda p, i: (p-2)*np.round(3 * min(50, i) / 50)/3 + 2, # You can pass a function to set penalty schedule here 
       heavyside= True, # You can pass a boolean to use heavyside or not
       eta=0.5, # You can pass a value to set eta - only used for heavyside
       beta=2.0, # You can pass a value to set beta, or a function of iteration to set beta schedule - only used for heavyside
   )

   # Define the optimizer
   optimizer = PGD( # You can use any of the optimizers in pyFANTOM: OC, MMA, PGD
       problem=to_problem,
       change_tol=np.inf, # No change tolerance
       fun_tol=1e-4, # Function tolerance (convergence criterion for the optimizer)
   )

====================================
Running The Optimization Loop
====================================

Now we can run the optimization loop and perform TO on the mega-voxel problem.

.. code-block:: python

   maximum_iterations = 300

   Progressbar = trange(maximum_iterations, desc='Optimizer Iterations', leave=True)
   objective_history = []
   for i in Progressbar:
       optimizer.iter()
       
       Progressbar.set_postfix(
           optimizer.logs()
       )
       
       objective_history.append(optimizer.logs()['objective'])
       
       if optimizer.converged():
           print(f'Converged in {i} iterations')
           break

The mega-voxel optimization converged in 73 iterations with the following output:

::

   Optimizer Iterations:  24%|██▍       | 73/300 [23:49<1:14:04, 19.58s/it, objective=244, variable change=5.01, function change=6.24e-6, iteration=75, residual=0.0135] Converged in 73 iterations

Now we can visualize the resulting topology:

.. code-block:: python

   to_problem.visualize_solution()

.. raw:: html

   <iframe src="/_static/MegaVoxelExample/8MBridge.html" width="100%" height="600px" style="border: none;"></iframe>

We can also verify the volume fraction:

.. code-block:: python

   (to_problem.desvars>0.5).sum()/to_problem.desvars.size

This returns approximately 0.05 (5%), confirming that the volume fraction constraint is satisfied.

Notes on Mega-Voxel Problems
----------------------------

- **Memory Efficiency**: FANTOM's multi-grid solver is matrix-free at the highest level, allowing you to solve problems with 8M+ elements on GPUs with 16GB VRAM.

- **Performance**: The multi-grid solver with W-cycles provides excellent convergence rates even for very large problems, making it practical to solve mega-voxel topology optimization problems.

- **Scalability**: The solver scales well with the number of grid levels. For problems with 8M elements, using 6 levels provides a good balance between memory usage and convergence speed.

