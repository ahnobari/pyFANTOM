from ..FiniteElement import FiniteElement as FE
from ...geom.CPU._mesh import StructuredMesh, GeneralMesh, StructuredMesh2D, StructuredMesh3D
from ...stiffness.CPU._FEA import StructuredStiffnessKernel, GeneralStiffnessKernel, UniformStiffnessKernel
from ...solvers.CPU._solvers import CHOLMOD, CG, BiCGSTAB, GMRES, SPLU, SPSOLVE, MultiGrid
from ...visualizers._2d import plot_mesh_2D, plot_problem_2D, plot_field_2D, plot_problem_2D_wVoid
from ...visualizers._3d import plot_problem_3D, plot_mesh_3D, plot_field_3D
from typing import Optional, Union, List
from scipy.spatial import KDTree
import numpy as np

### Champ added
from ...physics.NLElasticity import NLElasticity

class NLFiniteElement(FE):
    """
    Nonlinear finite element analysis engine for geometrically nonlinear problems.
    
    Extends FiniteElement with Newton-Raphson solver for large deformation analysis.
    Handles geometrically nonlinear elasticity where element stiffness matrices depend
    on current deformation state. Requires NLElasticity physics and NLUniformStiffnessKernel.
    
    Parameters
    ----------
    mesh : StructuredMesh2D, StructuredMesh3D, or GeneralMesh
        Finite element mesh defining geometry and physics
    kernel : NLUniformStiffnessKernel
        Nonlinear stiffness assembly kernel (must support set_Ktan())
    solver : CHOLMOD, CG, BiCGSTAB, GMRES, SPLU, SPSOLVE, or MultiGrid
        Linear system solver for Newton-Raphson iterations
    physics : NLElasticity
        Nonlinear elasticity physics model
        
    Attributes
    ----------
    mesh : Mesh
        Associated mesh
    kernel : NLUniformStiffnessKernel
        Nonlinear stiffness assembly kernel
    solver : Solver
        Linear solver
    rhs : ndarray
        Right-hand side force vector, shape (n_nodes * dof,)
    dof : int
        Degrees of freedom per node (2 for 2D, 3 for 3D)
    is_3D : bool
        True for 3D problems
    state : NLState
        Nonlinear state manager tracking deformation and internal forces
    lagrangeMult : ndarray
        Lagrange multipliers from adjoint solve (for sensitivity analysis)
    last_dR_Drho : ndarray
        Derivative of residual w.r.t. density (for sensitivity analysis)
        
    Methods
    -------
    add_dirichlet_boundary_condition(node_ids=None, positions=None, dofs=None, rhs=0)
        Apply fixed displacement boundary conditions
    add_point_forces(forces, node_ids=None, positions=None)
        Apply point loads
    reset_forces()
        Clear all forces
    reset_dirichlet_boundary_conditions()
        Remove all boundary conditions
    solveNL(rho=None)
        Solve nonlinear system using Newton-Raphson, returns (U, residual)
    visualize_problem(**kwargs)
        Plot mesh with BCs and loads
    visualize_density(rho, **kwargs)
        Plot optimization result
    visualize_deformed_mesh(rho=None, **kwargs)
        Plot deformed mesh configuration
    visualize_field(field, **kwargs)
        Plot displacement or stress field
        
    Notes
    -----
    **Nonlinear Analysis:**
    - Uses Newton-Raphson method with load stepping
    - Tangent stiffness matrix updated each iteration
    - Load stepping: 4 load steps by default for convergence
    - Convergence based on out-of-balance work
    
    **Sensitivity Analysis:**
    - Solves adjoint system after equilibrium
    - Computes dR/drho for gradient computation
    - More expensive than linear analysis
    
    **Use Cases:**
    - Large deformation problems
    - Buckling-sensitive structures
    - Compliant mechanisms
    - Soft robotics
    
    Examples
    --------
    >>> from pyFANTOM.CPU import *
    >>> from pyFANTOM import NLElasticity
    >>> # Setup nonlinear FEA
    >>> physics = NLElasticity(E=1.0, nu=0.3)
    >>> mesh = StructuredMesh2D(nx=64, ny=32, lx=2.0, ly=1.0, physics=physics)
    >>> kernel = NLUniformStiffnessKernel(mesh=mesh)
    >>> solver = CG(kernel=kernel)
    >>> FE = NLFiniteElement(mesh=mesh, kernel=kernel, solver=solver, physics=physics)
    >>> 
    >>> # Apply BCs and loads
    >>> FE.add_dirichlet_boundary_condition(node_ids=fixed_nodes, rhs=0)
    >>> FE.add_point_forces(node_ids=load_nodes, forces=np.array([[0, -1.0]]))
    >>> 
    >>> # Solve nonlinear system
    >>> U, residual = FE.solveNL(rho=np.ones(len(mesh.elements)) * 0.5)
    >>> print(f"Residual: {residual:.2e}")
    """
    def __init__(self, 
                 mesh: Union[StructuredMesh2D, StructuredMesh3D, GeneralMesh],
                 kernel: Union[StructuredStiffnessKernel, GeneralStiffnessKernel, UniformStiffnessKernel],
                 solver: Union[CHOLMOD, CG, BiCGSTAB, GMRES, SPLU, SPSOLVE, MultiGrid],
                 physics: NLElasticity):
        super().__init__()
        
        self.mesh = mesh
        self.kernel = kernel
        self.solver = solver
        self.dtype = mesh.dtype
        
        self.rhs = np.zeros([self.kernel.shape[0]], dtype=self.dtype)
        self.d_rhs = np.zeros([self.kernel.shape[0]], dtype=self.dtype) + np.nan
        self.KDTree = None
        self.nel = len(self.mesh.elements)
        
        self.is_3D = self.mesh.nodes.shape[1] == 3
        self.dof = self.mesh.dof

        # Champ added below
        self.n_nodes = self.mesh.nodes.shape[0]
        self.n_dofTotal = self.n_nodes * self.dof
        self.state = NLState(mesh=mesh, physics=physics)

        # Below is for gradients
        self.lagrangeMult = None
        self.last_B_scaled = None
        self.last_B = None
        self.last_S_hat_wo_rho_w_gamma = None
        self.last_S_hat = None
        self.last_S_hat_w_rho_wo_gamma = None
        self.last_dGammaDrho = None
        

    def add_dirichlet_boundary_condition(self,
                                        node_ids: Optional[np.ndarray] = None,
                                        positions: Optional[np.ndarray] = None,
                                        dofs: Optional[np.ndarray] = None,
                                        rhs: Union[float,np.ndarray] = 0):
        """
        Apply Dirichlet (fixed displacement) boundary conditions.
        
        Parameters
        ----------
        node_ids : ndarray, optional
            Node indices to constrain, shape (n_nodes,)
        positions : ndarray, optional
            Physical coordinates to constrain (uses KDTree search), shape (n_nodes, spatial_dim)
        dofs : ndarray, optional
            DOF mask per node, shape (n_nodes, dof) with 1=constrained, 0=free.
            If None, all DOFs at specified nodes are constrained
        rhs : float or ndarray, optional
            Prescribed displacement values. Scalar for zero displacement, or array shape (n_nodes, dof)
            
        Notes
        -----
        - Provide either node_ids OR positions, not both
        - dofs array: [[1,0]] constrains only x-displacement in 2D
        - Multiple calls accumulate constraints
        - Modifies kernel.constraints boolean array
        - Same interface as linear FiniteElement
        """
        if node_ids is None and positions is None:
            raise ValueError("Either node_ids or positions must be provided.")
        if node_ids is not None and positions is not None:
            raise ValueError("Only one of node_ids or positions should be provided.")
        
        N_con = node_ids.shape[0] if node_ids is not None else positions.shape[0]
        
        if isinstance(rhs, np.ndarray) and rhs.shape[0] != N_con:
            raise ValueError("rhs must have the same length as node_ids or positions.")
        
        if node_ids is not None:
            if dofs is None:
                # assume all dofs are being set
                for i in range(self.mesh.dof):
                    cons = node_ids * self.mesh.dof + i
                    self.kernel.add_constraints(cons)
                    self.d_rhs[cons] = rhs[:, i] if isinstance(rhs, np.ndarray) else rhs + np.nan_to_num(self.d_rhs[cons], nan=0)
            else:
                if dofs.shape[0] != node_ids.shape[0] and dofs.shape[0] != 1:
                    raise ValueError("dofs must have the same length as node_ids.")
                elif dofs.shape[0] == 1:
                    dofs = np.tile(dofs, (node_ids.shape[0], 1))
                    
                for i in range(self.mesh.dof):
                    cons = node_ids[dofs[:, i]==1] * self.mesh.dof + i
                    self.kernel.add_constraints(cons)
                    
                    self.d_rhs[cons] = rhs[dofs[:, i]==1, i] if isinstance(rhs, np.ndarray) else rhs + np.nan_to_num(self.d_rhs[cons], nan=0)
                    
        else:
            if self.KDTree is None:
                self.KDTree = KDTree(self.mesh.nodes)
                
            _, node_ids = self.KDTree.query(positions)
            
            self.add_dirichlet_boundary_condition(node_ids=node_ids, dofs=dofs, rhs=rhs)
            
    def add_neumann_boundary_condition(self, **kwargs):
        """
        Apply Neumann boundary conditions (not implemented).
        
        Raises
        ------
        NotImplementedError
            Always raised. Neumann boundary conditions not yet implemented for nonlinear FEA.
        """
        raise NotImplementedError("Neumann boundary conditions are not implemented in this version of FiniteElement.")
    
    def add_point_forces(self, 
                         forces: np.ndarray,
                         node_ids: Optional[np.ndarray] = None,
                         positions: Optional[np.ndarray] = None):
        """
        Apply point loads to specified nodes.
        
        Parameters
        ----------
        forces : ndarray
            Force vectors, shape (n_forces, dof) where dof is 2 for 2D or 3 for 3D.
            For 2D: [[fx, fy]], for 3D: [[fx, fy, fz]]
        node_ids : ndarray, optional
            Node indices for force application, shape (n_forces,)
        positions : ndarray, optional
            Physical coordinates for force application (uses KDTree search)
            
        Notes
        -----
        - Provide either node_ids OR positions
        - Multiple calls accumulate forces
        - Forces are automatically set to prescribed values at Dirichlet nodes
        - Units should match physics model (e.g., Newtons for SI units)
        - Same interface as linear FiniteElement
        """
        if node_ids is None and positions is None:
            raise ValueError("Either node_ids or positions must be provided.")
        if node_ids is not None and positions is not None:
            raise ValueError("Only one of node_ids or positions should be provided.")

        N_forces = node_ids.shape[0] if node_ids is not None else positions.shape[0]
        
        if (forces.shape[0] != N_forces and forces.shape[0] != 1) or forces.shape[1] != self.mesh.dof:
            raise ValueError("forces must have shape (N_forces, mesh.dof).")
        
        if node_ids is not None:
            if forces.shape[0] == 1:
                forces = np.tile(forces, (node_ids.shape[0], 1))
                
            for i in range(self.mesh.dof):
                self.rhs[node_ids * self.mesh.dof + i] += forces[:, i]
                
            # set dirichlet rhs
            dirichlet_dofs = np.logical_not(np.isnan(self.d_rhs))
            self.rhs[dirichlet_dofs] = self.d_rhs[dirichlet_dofs]
            
        else:
            if self.KDTree is None:
                self.KDTree = KDTree(self.mesh.nodes)
                
            _, node_ids = self.KDTree.query(positions)
            
            self.add_point_forces(forces=forces, node_ids=node_ids)
            
    def reset_forces(self):
        """
        Clear all applied forces.
        
        Sets the right-hand side force vector to zero. Useful when reconfiguring
        loading conditions or starting a new analysis.
        """
        self.rhs[:] = 0
    
    def reset_dirichlet_boundary_conditions(self):
        """
        Remove all Dirichlet boundary conditions.
        
        Clears all fixed displacement constraints and resets the kernel's constraint
        state. Useful when reconfiguring boundary conditions.
        """
        self.kernel.set_constraints([])
        self.kernel.has_cons = False
        
    def visualize_problem(self, ax=None, **kwargs):
        """
        Visualize mesh with boundary conditions and loads.
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on (2D only). If None, creates new figure
        **kwargs
            Additional arguments passed to plot_problem_2D/3D
            
        Returns
        -------
        matplotlib.axes.Axes or k3d.Plot
            Plot object (2D: matplotlib axes, 3D: k3d plot)
            
        Notes
        -----
        Shows:
        - Mesh elements
        - Fixed nodes (Dirichlet BCs) as markers
        - Applied forces as arrows/glyphs
        - For 3D: Interactive k3d plot
        - For 2D: Static matplotlib plot
        """
        if self.is_3D:
            return plot_problem_3D(
                self.mesh.nodes,
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                **kwargs)
        else:
            return plot_problem_2D(
                self.mesh.nodes,
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                ax=ax,
                **kwargs)
    
    def visualize_solution(self, ax=None, **kwargs):
        """
        Visualize solution (mesh with BCs and loads).
        
        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on (2D only). If None, creates new figure
        **kwargs
            Additional arguments passed to plot_problem_2D
            
        Returns
        -------
        matplotlib.axes.Axes
            Plot object (2D only, 3D raises NotImplementedError)
            
        Notes
        -----
        Currently only supports 2D visualization. For 3D, use visualize_problem().
        """
        if self.is_3D:
            raise NotImplementedError("3D solution visualization is not implemented yet.")
            return
        else:
            return plot_problem_2D(
                self.mesh.nodes,
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                ax=ax,
                **kwargs)
            
    def visualize_density(self, rho, ax=None, **kwargs):
        """
        Visualize density distribution (optimization result).
        
        Parameters
        ----------
        rho : ndarray
            Element densities, shape (n_elements,) or (n_elements, n_materials)
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on (2D only). If None, creates new figure
        **kwargs
            Additional arguments passed to plot_problem_2D/3D
            
        Returns
        -------
        matplotlib.axes.Axes or k3d.Plot
            Plot object (2D: matplotlib axes, 3D: k3d plot)
            
        Notes
        -----
        Color-codes elements by density value (0=void, 1=solid).
        For multi-material problems, pass rho with shape (n_elements, n_materials).
        """
        if self.is_3D:
            return plot_problem_3D(
                self.mesh.nodes,
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                rho = rho,
                **kwargs)
        else:
            return plot_problem_2D(
                self.mesh.nodes,
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                ax=ax,
                rho = rho,
                **kwargs)
        
    def visualize_deformed_mesh(self, rho = None, ax=None, **kwargs):
        """
        Visualize deformed mesh configuration.
        
        Parameters
        ----------
        rho : ndarray, optional
            Element densities for masking void regions, shape (n_elements,).
            If None, uses full density (rho=1.0)
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on (2D only). If None, creates new figure
        **kwargs
            Additional arguments passed to plot_problem_2D_wVoid
            
        Returns
        -------
        matplotlib.axes.Axes
            Plot object (2D only, 3D raises NotImplementedError)
            
        Notes
        -----
        Shows mesh in deformed configuration (nodes + displacements from solveNL()).
        Useful for visualizing large deformations in nonlinear analysis.
        Currently only supports 2D visualization.
        """
        if rho is None:
            rho = np.ones(self.nel, dtype=self.dtype)

        if self.is_3D:
            raise NotImplementedError("3D deformed mesh visualization is not implemented yet.")
        else:
            return plot_problem_2D_wVoid(
                self.mesh.nodes + self.state._get_tU_global().reshape(-1,2),
                self.mesh.elements,
                f = self.rhs.reshape(-1, self.mesh.dof),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof),
                ax=ax,
                rho = rho,
                **kwargs)
            
    def visualize_field(self, field, ax=None, rho=None, **kwargs):
        """
        Visualize scalar or vector field (displacement, stress, strain, etc.).
        
        Parameters
        ----------
        field : ndarray
            Field values to plot. Shape depends on field type:
            - Scalar: (n_elements,) or (n_nodes,)
            - Vector: (n_nodes, dof) for displacement-like fields
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on (2D only). If None, creates new figure
        rho : ndarray, optional
            Density mask to hide void regions, shape (n_elements,)
        **kwargs
            Additional arguments passed to plot_field_2D/3D
            
        Returns
        -------
        matplotlib.axes.Axes or k3d.Plot
            Plot object (2D: matplotlib axes, 3D: k3d plot)
            
        Notes
        -----
        Common fields:
        - Displacement: U.reshape(-1, dof)
        - Von Mises stress: from Problem.FEA() output
        - Strain energy: from Problem.FEA() output
        - If rho provided, elements with rho < 0.5 are hidden
        """
        if self.is_3D:
            return plot_field_3D(
                self.mesh.nodes,
                self.mesh.elements,
                field,
                rho=rho,
                **kwargs)
        else:
            return plot_field_2D(
                self.mesh.nodes,
                self.mesh.elements,
                field,
                rho=rho,
                ax=ax,
                **kwargs)
    
    def solve(self, rho=None):
        """
        Solve linear system (for compatibility, calls solveNL).
        
        Parameters
        ----------
        rho : ndarray, optional
            Element density variables, shape (n_elements,). If None, uses rho=1 (full density)
            
        Returns
        -------
        U : ndarray
            Displacement vector, shape (n_nodes * dof,)
        residual : float
            Normalized residual ||K@U - F|| / ||F||
            
        Notes
        -----
        For compatibility with linear FiniteElement interface. In practice,
        use solveNL() for nonlinear analysis. This method may not work correctly
        for large deformations.
        """
        if rho is not None and rho.shape[0] != self.nel:
            raise ValueError("rho must have the same length as the number of elements in the mesh.")
        
        if rho is None:
            rho = np.ones(self.nel, dtype=self.dtype)
        
        self.kernel.set_rho(rho)
        U, residual = self.solver.solve(self.rhs, use_last=True)
        
        return U, residual
    
    def solveNL(self, rho=None):
        """
        Solve nonlinear system using Newton-Raphson method with load stepping.
        
        Parameters
        ----------
        rho : ndarray, optional
            Element density variables, shape (n_elements,). If None, uses rho=1 (full density)
            
        Returns
        -------
        U : ndarray
            Total displacement vector after convergence, shape (n_nodes * dof,)
        residual : float
            Final normalized residual ||K@U - F|| / ||F||
            
        Notes
        -----
        **Newton-Raphson Method:**
        - Uses load stepping: 4 load steps by default
        - Each load step applies incremental force: F_step = (step+1)/nLoadSteps * F_total
        - Up to 50 Newton iterations per load step
        - Convergence: |OoBWork_current / OoBWork_initial| < 5e-5
        
        **Out-of-Balance Work:**
        - OoBWork = K@U - F (residual force vector)
        - Measures equilibrium error
        - Constrained DOFs excluded from convergence check
        
        **Tangent Stiffness:**
        - Updated each Newton iteration via state._KTan()
        - Computed from current deformation state
        - Passed to kernel via set_Ktan()
        
        **Adjoint Preparation:**
        - After equilibrium, solves for Lagrange multipliers
        - Stores dR/drho for sensitivity analysis
        - Required for gradient computation in optimization
        
        **State Management:**
        - Updates state.tDeltatUk (incremental displacement)
        - Updates state.tU_global (total displacement)
        - Tracks element stress and deformation
        
        Raises
        ------
        ValueError
            If rho has wrong shape or solver doesn't converge within max iterations
            
        Examples
        --------
        >>> U, residual = FE.solveNL(rho=np.ones(nel) * 0.5)
        >>> print(f"Converged with residual: {residual:.2e}")
        >>> # Access final displacement
        >>> U_reshaped = U.reshape(-1, 2)  # For 2D: (n_nodes, 2)
        """
        if rho is not None and rho.shape[0] != self.nel:
            raise ValueError("rho must have the same length as the number of elements in the mesh.")
        
        if rho is None:
            rho = np.ones(self.nel, dtype=self.dtype)

        ittGlobal = 0

        self.state._init_tDeltatUk()
        if ittGlobal == 0:
            self.state._init_tU_global()
        else:
            if self.state._get_tU_global() is not None:
                uPreviousSolve = self.state._get_tU_global()
                self.state._update_tDeltatUk(uPreviousSolve)
                self.state._init_tU_global()
                self.state._update_tU_global(uPreviousSolve)
            else:
                self.state._init_tU_global()
        
        self.kernel.set_rho(rho) 

        # Use this to divide the global force vector
        nLoadSteps = 4
        maxSolvesPerLoadStep = 50

        alpha = 1/nLoadSteps
        
        # Get Dirichlet nodes from kernel. #Only works for dirichlet BC for now
        constrainedDofIdx = np.where(self.kernel.constraints)

        while ittGlobal < nLoadSteps:
            solve_has_converged = False

            rhsLoadStep = (ittGlobal + 1) * alpha * self.rhs
            self.state._set_FextTarget(rhsLoadStep)
            
            previousOoBWork = None

            ittWithinLoadStep = 0 
            while ittWithinLoadStep < maxSolvesPerLoadStep and not solve_has_converged:

                KTanCurrent = self.state._KTan(rho, self.state._get_tDeltatUk()).copy()

                if ittWithinLoadStep == 0:
                    OoBWork_0 = self.state._get_OoBwork(rho, self.state._get_tDeltatUk())
                    OoBWork_0[constrainedDofIdx] = 0 # make constrainted dof 0 for OoBWork
                    compliance_ref = OoBWork_0.dot(rhsLoadStep)
                    previousOoBWork = OoBWork_0.copy()


                # Solve system
                self.kernel.set_Ktan(KTanCurrent)
                Uincr, residual = self.solver.solve(previousOoBWork, use_last=True)


                self.state._update_tDeltatUk(Uincr)
                self.state._update_tU_global(Uincr)
                
                
                OoBWork_opt = self.state._get_OoBwork(rho, self.state._get_tDeltatUk())
                OoBWork_opt[constrainedDofIdx] = 0

                compliance_current = OoBWork_opt.dot(rhsLoadStep)
                previousOoBWork = OoBWork_opt.copy()


                if np.abs(compliance_current/compliance_ref) < 5e-5: # this is a bit relaxed
                    solve_has_converged = True

                if ittWithinLoadStep > maxSolvesPerLoadStep:
                    raise ValueError("Non-linear solver did not converge within the maximum number of iterations per load step.")
                
                ittWithinLoadStep += 1
                

            ittGlobal += 1

        # Solve one last time for lagrange multipliers
        KTan, t0F, dR_Drho = self.state.lastIterationMatrices(rho)

        self.kernel.set_Ktan(KTan)
        lagrangeMult, residual = self.solver.solve(self.rhs, use_last=False)

        self.lagrangeMult = (self.state._convertGlobalUVect2ElementalArray(lagrangeMult)).reshape(self.state.n_elements, -1)
        self.last_dR_Drho = dR_Drho

        return self.state._get_tU_global(), residual




class NLState():
    """
    Nonlinear state manager for tracking deformation and internal forces.
    
    Internal helper class for NLFiniteElement that manages state variables during
    Newton-Raphson iterations. Tracks incremental and total displacements, internal
    forces, and element-level deformation states.
    
    Parameters
    ----------
    mesh : GeneralMesh
        Finite element mesh
    physics : NLElasticity
        Nonlinear elasticity physics model
        
    Attributes
    ----------
    mesh : GeneralMesh
        Associated mesh
    physics : NLElasticity
        Nonlinear elasticity physics
    nodes : ndarray
        Node coordinates, shape (n_nodes, spatial_dim)
    elements : ndarray
        Element connectivity, shape (n_elements, nodes_per_element)
    x0s : ndarray
        Initial element node coordinates, shape (n_elements, nodes_per_element, spatial_dim)
    dof : int
        Degrees of freedom per node (2 for 2D, 3 for 3D)
    n_nodes : int
        Total number of nodes
    n_dofTotal : int
        Total number of DOFs (n_nodes * dof)
    n_elements : int
        Total number of elements
    stressLastSolved : ndarray
        Stress state from last converged solution, shape (3, n_elements, 2, 2)
    stressCurrent : ndarray
        Current stress state during iteration, shape (3, n_elements, 2, 2)
    xsLastSolved : ndarray
        Element node coordinates from last converged solution
    tDeltatUk : ndarray
        Incremental displacement per element, shape (n_elements, nodes_per_element, spatial_dim)
    tU_global : ndarray
        Total global displacement vector, shape (n_dofTotal,)
    conArr : ndarray
        Connectivity array mapping element DOFs to global DOFs, shape (n_elements, 8) for quads
    FextTarget : ndarray
        Target external force vector for current load step, shape (n_dofTotal,)
    """
    def __init__(self,
                 mesh: GeneralMesh,
                 physics: NLElasticity):
        """
        Initialize nonlinear state manager.
        
        Parameters
        ----------
        mesh : GeneralMesh
            Finite element mesh
        physics : NLElasticity
            Nonlinear elasticity physics model
        """
        self.mesh = mesh
        self.physics = physics

        self.nodes = mesh.nodes
        self.elements = mesh.elements
        self.x0s = self.nodes[self.elements]

        self.dof = self.mesh.dof
        self.n_nodes = self.mesh.nodes.shape[0]
        self.n_dofTotal = self.n_nodes * self.dof
        self.n_elements = self.x0s.shape[0]

        self.stressLastSolved = np.zeros((3, self.n_elements, 2, 2)) # the last two dimensions should be equivalent to the number of gauss points in each dimension
        self.stressCurrent = np.zeros((3, self.n_elements, 2, 2)) 
        self.xsLastSolved = self.x0s.copy()
        self.tDeltatUk = None


        self.tU_global = None 

        # the connectivity array is hardcoded for quad elements for now
        self.conArr = np.array((2 * self.elements[:,0], 2 * self.elements[:,0] + 1, 
                           2 * self.elements[:,1], 2 * self.elements[:,1] + 1, 
                           2 * self.elements[:,2], 2 * self.elements[:,2] + 1, 
                           2 * self.elements[:,3], 2 * self.elements[:,3] + 1)).transpose()

        self.FextTarget = None

    def _KTan(self, rho, tDeltatUk):
        """
        Compute tangent stiffness matrix for current deformation state.
        
        Parameters
        ----------
        rho : ndarray
            Element densities, shape (n_elements,)
        tDeltatUk : ndarray
            Incremental displacement per element, shape (n_elements, nodes_per_element, spatial_dim)
            
        Returns
        -------
        KTan : ndarray
            Tangent stiffness matrix, shape (n_dofTotal, n_dofTotal)
            
        Notes
        -----
        Computes KTan from current deformed configuration (x0s + tDeltatUk).
        Used in Newton-Raphson iterations to linearize the nonlinear system.
        """
        xs = self.x0s + tDeltatUk
        KTan, t0F = self.physics.KTan(self.x0s, xs, rho)[0:2] 
        return KTan
    
    def _t0F(self, rho, tDeltatUk):
        """
        Compute internal force vector for current deformation state.
        
        Parameters
        ----------
        rho : ndarray
            Element densities, shape (n_elements,)
        tDeltatUk : ndarray
            Incremental displacement per element, shape (n_elements, nodes_per_element, spatial_dim)
            
        Returns
        -------
        t0F : ndarray
            Internal force vector per element, shape (n_elements, element_dof)
            
        Notes
        -----
        Computes internal forces from current deformed configuration.
        Used to compute out-of-balance work (residual) in Newton-Raphson.
        """
        xs = self.x0s + tDeltatUk
        KTan, t0F = self.physics.KTan(self.x0s, xs, rho)[0:2]
        return t0F

    def lastIterationMatrices(self, rho):
        """
        Get matrices from last converged iteration for adjoint solve.
        
        Parameters
        ----------
        rho : ndarray
            Element densities, shape (n_elements,)
            
        Returns
        -------
        KTan : ndarray
            Tangent stiffness matrix at convergence, shape (n_dofTotal, n_dofTotal)
        t0F : ndarray
            Internal force vector at convergence, shape (n_elements, element_dof)
        dR_Drho : ndarray
            Derivative of residual w.r.t. density, shape (n_dofTotal, n_elements)
            
        Notes
        -----
        Called after Newton-Raphson convergence to prepare for adjoint sensitivity
        analysis. dR_Drho is needed for gradient computation in optimization.
        """
        xs = self.x0s + self.tDeltatUk
        KTan, t0F, dR_Drho = self.physics.KTan(self.x0s, xs, rho)
        return KTan, t0F, dR_Drho

    def _get_tU_global(self):
        """
        Get total global displacement vector.
        
        Returns
        -------
        tU_global : ndarray or None
            Total displacement vector, shape (n_dofTotal,), or None if not initialized
            
        Notes
        -----
        Returns a copy to prevent external modification of internal state.
        """
        if self.tU_global is None: 
            return None
        else:
            return self.tU_global.copy()
    
    def _init_tU_global(self) -> None:
        """
        Initialize total global displacement vector to zero.
        
        Notes
        -----
        Called at the start of each load step to reset total displacement.
        """
        self.tU_global = np.zeros(self.n_dofTotal)

    def _update_tDeltatUk(self, Uincr) -> None:
        """
        Update incremental displacement from Newton-Raphson iteration.
        
        Parameters
        ----------
        Uincr : ndarray
            Incremental displacement from linear solve, shape (n_dofTotal,)
            
        Notes
        -----
        Converts global displacement increment to elemental format and accumulates
        into tDeltatUk. Called each Newton iteration.
        """
        self.tDeltatUk += self._convertGlobalUVect2ElementalArray(Uincr)

    def _update_tU_global(self, Uincr) -> None:
        """
        Update total global displacement vector.
        
        Parameters
        ----------
        Uincr : ndarray
            Incremental displacement from linear solve, shape (n_dofTotal,)
            
        Notes
        -----
        Accumulates displacement increments into total displacement.
        Called each Newton iteration to track total deformation.
        """
        self.tU_global += Uincr

    def _get_tDeltatUk(self):
        """
        Get incremental displacement per element.
        
        Returns
        -------
        tDeltatUk : ndarray
            Incremental displacement per element, shape (n_elements, nodes_per_element, spatial_dim)
            
        Notes
        -----
        Returns a copy to prevent external modification of internal state.
        """
        return self.tDeltatUk.copy()
    
    def _set_FextTarget(self, FextTarget) -> None:
        """
        Set target external force vector for current load step.
        
        Parameters
        ----------
        FextTarget : ndarray
            Target external force vector, shape (n_dofTotal,)
            
        Notes
        -----
        Called at the start of each load step to set the target force level.
        Used to compute out-of-balance work (residual).
        """
        self.FextTarget = FextTarget

    def _get_FextTarget(self):
        """
        Get target external force vector.
        
        Returns
        -------
        FextTarget : ndarray
            Target external force vector, shape (n_dofTotal,)
            
        Notes
        -----
        Returns a copy to prevent external modification of internal state.
        """
        return self.FextTarget.copy()

    def _init_tDeltatUk(self) -> None:
        """
        Initialize incremental displacement per element to zero.
        
        Notes
        -----
        Called at the start of each load step to reset incremental displacement.
        Shape matches element node coordinates (hardcoded for 4-noded quad elements).
        """
        # hard coded for 4-noded quad element
        self.tDeltatUk = np.zeros(self.xsLastSolved.shape)

    def _get_internalwork(self, rho, tDeltatUk):
        """
        Get internal force vector (internal work) for current state.
        
        Parameters
        ----------
        rho : ndarray
            Element densities, shape (n_elements,)
        tDeltatUk : ndarray
            Incremental displacement per element, shape (n_elements, nodes_per_element, spatial_dim)
            
        Returns
        -------
        t0F : ndarray
            Internal force vector per element, shape (n_elements, element_dof)
            
        Notes
        -----
        Wrapper around _t0F() for consistency in naming. Internal work refers to
        the internal forces that resist deformation.
        """
        return self._t0F(rho, tDeltatUk)

    def _get_OoBwork(self, rho, tDeltatUk):
        """
        Compute out-of-balance work (residual force vector).
        
        Parameters
        ----------
        rho : ndarray
            Element densities, shape (n_elements,)
        tDeltatUk : ndarray
            Incremental displacement per element, shape (n_elements, nodes_per_element, spatial_dim)
            
        Returns
        -------
        OoBwork : ndarray
            Out-of-balance work (residual), shape (n_dofTotal,)
            
        Notes
        -----
        OoBwork = F_ext - F_int, measures equilibrium error.
        Used for Newton-Raphson convergence check. Should approach zero at equilibrium.
        """
        return self._get_FextTarget() - self._assemInternalWork(self._get_internalwork(rho, tDeltatUk))

    def _assemInternalWork(self, t0F):
        """
        Assemble global internal force vector from elemental forces.
        
        Parameters
        ----------
        t0F : ndarray
            Internal force vector per element, shape (n_elements, element_dof)
            
        Returns
        -------
        internalWork : ndarray
            Assembled global internal force vector, shape (n_dofTotal,)
            
        Raises
        ------
        ValueError
            If number of elements doesn't match t0F shape
            
        Notes
        -----
        Scatters elemental forces to global DOFs using connectivity array.
        Standard finite element assembly operation.
        """
        if self.n_elements != t0F.shape[0]:
            raise ValueError("The number of elements and the number of stress entries do not match.")
        
        
        internalWork = np.zeros(self.n_dofTotal)
        for i in range(self.n_elements):
            internalWork[self.conArr[i,:]] += t0F[i,:]

        return internalWork
    
    def _convertGlobalUVect2ElementalArray(self, globalVect):
        """
        Convert global displacement vector to elemental array format.
        
        Parameters
        ----------
        globalVect : ndarray
            Global displacement vector, shape (n_dofTotal,)
            
        Returns
        -------
        stackedElementalArray : ndarray
            Displacement per element, shape (n_elements, nodes_per_element, spatial_dim)
            
        Notes
        -----
        Gathers global DOFs to element-level format using connectivity array.
        Used to convert linear solver output to element coordinates for physics evaluation.
        Hardcoded for 2D (spatial_dim=2) in reshape.
        """
        stackedElementalArray = globalVect[self.conArr].reshape(self.n_elements, -1, 2)
        return stackedElementalArray