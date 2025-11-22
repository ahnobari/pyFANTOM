3D Topology Optimization Example
================================

Classic 3D Structured Meshes
-----------------------------

In this example we will show how you can use pyFANTOM to perform minimum compliance TO on **structured meshes** in 3D. For this example we will perform TO on the bridge problem.

Below we detail how you can do this in a few lines of code!


=============================
Importing The Necessary Tools
=============================

First we import the tools from pyFANTOM we will be using in this example.

.. code-block:: python

   from pyFANTOM.CPU import(
       StructuredMesh3D, # Mesh Object for 3D Structured Meshes
       StructuredFilter3D, # Filter kernel for 3D Structured Meshes
       StructuredStiffnessKernel, # Stiffness Kernel for 3D Structured Meshes
       FiniteElement, # Finite Element Object to Setup Boundary Conditions and RHS
       CHOLMOD, CG, GMRES, MultiGrid, SPLU, SPSOLVE, # Sparse Solvers
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
   height = 0.25
   width = 0.25

   # Create a physics model
   physics = LinearElasticity(E=1.0, nu=1/3, thickness=1.0, type='PlaneStress')

   # Create a mesh
   mesh = StructuredMesh3D(nx=64, ny=32, nz=32, lx=length, ly=height, lz=width, physics=physics)

====================================
Setting Up The Finite Element Solver
====================================

The next step in using pyFANTOM is setting up a stiffness kernel and finite-element boundary conditions.

.. code-block:: python

   # Create a stiffness kernel
   stiffness_kernel = StructuredStiffnessKernel(mesh=mesh)

   # Create a solver
   # This can be any of the sparse solvers in pyFANTOM but in 3D CHOLMOD quickly becomes infeasible
   solver = MultiGrid(mesh=mesh, kernel=stiffness_kernel, n_level=4, cycle='W', w_level=1, maxiter=200, tol=1e-4) 

   # Create a finite-element object
   fe = FiniteElement(mesh=mesh, kernel=stiffness_kernel, solver=solver)

   # Visualize the problem
   fe.visualize_problem()

.. raw:: html

   <iframe src="../_static/3DExample/1.html" width="100%" height="600px" style="border: none;"></iframe>

As seen above we do not have loads or boundary conditions. So now we will set this up for the bridge problem:

.. code-block:: python

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

   # Visualize the problem
   fe.visualize_problem()

.. raw:: html

   <iframe src="../_static/3DExample/2.html" width="100%" height="600px" style="border: none;"></iframe>

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
       volume_fraction=[0.1], # You can pass a list of values to perform volume fraction control for each material
       void=1e-9, # You can pass a value to set void modulus
       penalty=3, # You can pass a value to set penalty factor
       #penalty_schedule = lambda p, i: (p-1)*np.round(4 * min(100, i) / 100)/4 + 1, # You can pass a function to set penalty schedule here 
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

Now we can run the optimization loop and perform TO.

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

The optimization converged in 46 iterations with the following output:

::

   Optimizer Iterations:  15%|█▌        | 46/300 [02:12<12:11,  2.88s/it, objective=132, variable change=2.34, function change=7.19e-5, iteration=48, residual=8.09e-5] Converged in 46 iterations

Now we can visualize the resulting topology:

.. code-block:: python

   to_problem.visualize_solution()

.. raw:: html

   <iframe src="../_static/3DExample/3.html" width="100%" height="600px" style="border: none;"></iframe>

Running on GPU
--------------

The exact same problem can be run on GPU by just switching the backend. 

A few notes on CUDA back-ends:
- In general the only exact solver available for CUDA is SPSOLVE, but in general for structured meshes the Multi-Grid solver with our custom CUDA kernels is the fastest solver on GPU.
- The CUDA backend uses cupy, however, the inputs to the FE class can be numpy arrays.

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


   length = 1.0
   width = 0.25
   height = 0.25

   # Create a physics model
   physics = LinearElasticity(E=1.0, nu=1/3, thickness=1.0, type='PlaneStress')

   # Create a mesh
   mesh = StructuredMesh3D(nx=64, ny=32, nz=32, lx=length, ly=height, lz=width, physics=physics)

   # Create a stiffness kernel
   stiffness_kernel = StructuredStiffnessKernel(mesh = mesh)

   # Create a solver
   solver = MultiGrid(
       mesh=mesh,
       kernel=stiffness_kernel,
       maxiter=200,
       tol=1e-4,
       n_level=4,
       cycle='W',
       w_level=1
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

   # Define the filter kernel for TO
   filter_kernel = StructuredFilter3D(mesh=mesh, r_min=1.5)

   # Define the TO problem
   to_problem = MinimumCompliance(
       FE=fe,
       filter=filter_kernel,
       E_mul=[1.0], # You can pass a list of values to perform multi-material TO
       volume_fraction=[0.1], # You can pass a list of values to perform volume fraction control for each material
       void=1e-9, # You can pass a value to set void modulus
       penalty=3, # You can pass a value to set penalty factor
       #penalty_schedule = lambda p, i: (p-1)*np.round(4 * min(100, i) / 100)/4 + 1, # You can pass a function to set penalty schedule here 
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

   # Visualize the problem
   fe.visualize_problem()

.. raw:: html

   <iframe src="../_static/3DExample/4.html" width="100%" height="600px" style="border: none;"></iframe>

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

The GPU optimization converged in 59 iterations:

::

   Optimizer Iterations:  20%|█▉        | 59/300 [00:09<00:38,  6.32it/s, objective=131, variable change=1.97, function change=6.37e-5, iteration=61, residual=9.35e-5] Converged in 59 iterations

.. code-block:: python

   to_problem.visualize_solution()

.. raw:: html

   <iframe src="../_static/3DExample/5.html" width="100%" height="600px" style="border: none;"></iframe>

Unstructured Meshes
-------------------

pyFANTOM also supports unstructured meshes for topology optimization. Below we will use pygmesh to mesh a 3D model and perform TO on it. 

**NOTE:**
- This is a toy problem to represent the capabilities of the package not a real-world problem.
- The Multi-Grid solver which is the most efficient way to solve the FEA problem is only available for structured meshes. So you may want to consider voxelizing geometry in some cases. 
- The code below uses GPU since solving these larger 3D problems on CPU can be slow even with all of the efficient kernels that FANTOM ships with.

.. code-block:: python

   from pyFANTOM.CUDA import(
       GeneralMesh, # Mesh Object for Unstructured Meshes
       GeneralFilter, # Filter kernel for Unstructured Meshes
       GeneralStiffnessKernel, UniformStiffnessKernel, # Stiffness Kernel for Unstructured Meshes
       FiniteElement, # Finite Element Object to Setup Boundary Conditions and RHS
       CG, GMRES, MultiGrid, SPSOLVE, # Sparse Solvers
       MinimumCompliance, # TO Problem Definition
       PGD, MMA, OC # Nonlinear Optimizers
   )

   from pyFANTOM.Physics import LinearElasticity # Physics Model For FEA

   import pygmsh

   # Other Imports
   import numpy as np
   import cupy as cp
   import matplotlib.pyplot as plt
   from tqdm import trange

=====================
Create A Pac-Man Mesh
=====================

Below we will make a toy problem mesh using pygmesh.

.. code-block:: python

   # Create a geometry object
   with pygmsh.occ.Geometry([
           '-setnumber', 'Mesh.Algorithm', '8',
           '-setnumber', 'Mesh.SubdivisionAlgorithm', '1',
           '-setnumber', 'Mesh.RecombinationAlgorithm', '2',
           '-setnumber', 'Mesh.RecombineAll', '0',
           '-setnumber', 'Mesh.SaveWithoutOrphans', '1'
       ]) as geom:
       # Load the STEP file
       geom.import_shapes("PacMan.step")
       geom.characteristic_length_max = 10
       geom.characteristic_length_min = 10
       mesh_uniform = geom.generate_mesh()

       mesh_uniform = [mesh_uniform.points[:,0:3], mesh_uniform.cells[2].data.astype(int).tolist()]

   # Create a physics model
   physics = LinearElasticity(E=1.0, nu=1/3, thickness=1.0, type='PlaneStress')

   # Create a mesh
   mesh = GeneralMesh(np.array(mesh_uniform[0])/1000, np.array(mesh_uniform[1]), physics=physics)

   # Create a stiffness kernel
   if mesh.is_uniform:
       stiffness_kernel = UniformStiffnessKernel(mesh=mesh)
   else:
       stiffness_kernel = GeneralStiffnessKernel(mesh=mesh) 

   # Create a solver
   solver = CG(kernel=stiffness_kernel, maxiter=5000, tol=1e-4)

   # Create a finite-element object
   fe = FiniteElement(mesh=mesh, kernel=stiffness_kernel, solver=solver)

   # Visualize the mesh
   fe.visualize_problem()

.. raw:: html

   <iframe src="../_static/3DExample/6.html" width="100%" height="600px" style="border: none;"></iframe>

Now we will setup boundary conditions for this toy problem:

.. code-block:: python

   fe.reset_dirichlet_boundary_conditions()
   fe.reset_forces()

   # find nodes on the boundary of the packman circle
   bc_nodes = np.isclose(np.linalg.norm(mesh.nodes[:,0:2].get() - np.array([0.5,0.5]), axis=1),0.5)
   bc_nodes = np.logical_and(bc_nodes, mesh.nodes[:,0].get() < 0.25)
   bc_nodes = np.where(bc_nodes)[0]

   fe.add_dirichlet_boundary_condition(
       node_ids=bc_nodes,
       positions=None,
       dofs=np.array([[1, 1, 1]]),
       rhs=0.0
   )

   # apply load at the mouth of the packman
   upper_mouth = np.logical_and(np.isclose(np.abs(np.dot(mesh.nodes.get(),np.array([-1,1, 0]))),0), mesh.nodes[:,0].get() > 0.5)
   force_node = np.where(upper_mouth)[0]
   force_nodes = force_node[np.isin(force_node, bc_nodes, invert=True)]
   force = np.array([[1,-1, 0]])/force_nodes.shape[0]/4 # Broadcastable, If needed you can provide one for each point

   fe.add_point_forces(
       node_ids=force_nodes,
       positions=None,
       forces=force
   )

   lower_mouth = np.logical_and(np.isclose(np.abs(np.dot(mesh.nodes.get(),np.array([1,1, 0]))),1.0), mesh.nodes[:,0].get() > 0.5)
   force_node = np.where(lower_mouth)[0]
   force_nodes = force_node[np.isin(force_node, bc_nodes, invert=True)]
   force = np.array([[1,1, 0]])/force_nodes.shape[0]/4 # Broadcastable, If needed you can provide one for each point

   fe.add_point_forces(
       node_ids=force_nodes,
       positions=None,
       forces=force
   )

   # Visualize the problem
   fe.visualize_problem()

.. raw:: html

   <iframe src="../_static/3DExample/7.html" width="100%" height="600px" style="border: none;"></iframe>

Now we can just setup the problem and optimizer like before:

.. code-block:: python

   # Define the filter kernel for TO This may take a while to compute for a large unstructured mesh
   filter_kernel = GeneralFilter(mesh=mesh, r_min=0.01) # In general cases r_min is mesh space, not scaled by mesh size be cassreful!

   # Define the TO problem
   to_problem = MinimumCompliance(
       FE=fe,
       filter=filter_kernel,
       E_mul=[1.0], # You can pass a list of values to perform multi-material TO
       volume_fraction=[0.05], # You can pass a list of values to perform volume fraction control for each material
       void=1e-9, # You can pass a value to set void modulus
       penalty=3, # You can pass a value to set penalty factor
       #penalty_schedule = lambda p, i: (p-1)*np.round(4 * min(100, i) / 100)/4 + 1, # You can pass a function to set penalty schedule here 
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

The unstructured mesh optimization converged in 148 iterations:

::

   Optimizer Iterations:  49%|████▉     | 148/300 [03:25<03:31,  1.39s/it, objective=50.6, variable change=0.726, function change=9.5e-5, iteration=150, residual=0.00177] Converged in 148 iterations

.. code-block:: python

   # Visualize the solution
   to_problem.visualize_solution()

.. raw:: html

   <iframe src="../_static/3DExample/8.html" width="100%" height="600px" style="border: none;"></iframe>

As you can see this geometry is a bit of a mess. For best results you should work with as uniform of a hex mesh as possible instead of this simple tetra mesh.

