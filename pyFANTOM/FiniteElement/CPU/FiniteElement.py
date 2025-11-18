from ..FiniteElement import FiniteElement as FE
from ...geom.CPU._mesh import StructuredMesh, GeneralMesh, StructuredMesh2D, StructuredMesh3D
from ...stiffness.CPU._FEA import StructuredStiffnessKernel, GeneralStiffnessKernel, UniformStiffnessKernel
from ...solvers.CPU._solvers import CHOLMOD, CG, BiCGSTAB, GMRES, SPLU, SPSOLVE, MultiGrid
from ...visualizers._2d import plot_mesh_2D, plot_problem_2D, plot_field_2D
from ...visualizers._3d import plot_problem_3D, plot_mesh_3D, plot_field_3D
from typing import Optional, Union, List
from scipy.spatial import KDTree
import numpy as np


class FiniteElement(FE):
    """
    Finite element analysis engine managing boundary conditions, forces, and solution.
    
    Central class coordinating mesh, stiffness assembly kernel, and linear solver for FEA.
    Provides convenient methods for applying boundary conditions, forces, solving, and visualization.
    
    Parameters
    ----------
    mesh : StructuredMesh2D, StructuredMesh3D, or GeneralMesh
        Finite element mesh defining geometry and physics
    kernel : StructuredStiffnessKernel, GeneralStiffnessKernel, or UniformStiffnessKernel
        Stiffness matrix assembly kernel
    solver : CHOLMOD, CG, BiCGSTAB, GMRES, SPLU, SPSOLVE, or MultiGrid
        Linear system solver
        
    Attributes
    ----------
    mesh : Mesh
        Associated mesh
    kernel : StiffnessKernel
        Stiffness assembly kernel
    solver : Solver
        Linear solver
    rhs : ndarray
        Right-hand side force vector, shape (n_nodes * dof,)
    dof : int
        Degrees of freedom per node (2 for 2D, 3 for 3D)
    is_3D : bool
        True for 3D problems
        
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
    solve(rho=None)
        Solve linear system K(rho) @ U = F, returns (U, residual)
    visualize_problem(**kwargs)
        Plot mesh with BCs and loads
    visualize_density(rho, **kwargs)
        Plot optimization result
    visualize_field(field, **kwargs)
        Plot displacement or stress field
        
    Notes
    -----
    - Boundary conditions modify kernel.constraints boolean array
    - Can specify nodes by index (node_ids) or coordinates (positions with KDTree search)
    - Dirichlet BC handling: rows/columns zeroed, diagonal set to 1
    - solve() returns residual: ||K@U - F|| / ||F|| for validation
    
    Examples
    --------
    >>> from pyFANTOM.CPU import *
    >>> mesh = StructuredMesh2D(nx=64, ny=32, lx=2.0, ly=1.0)
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> solver = CHOLMOD(kernel=kernel)
    >>> FE = FiniteElement(mesh=mesh, kernel=kernel, solver=solver)
    >>> 
    >>> # Fix left edge
    >>> left_nodes = np.where(mesh.nodes[:, 0] < 1e-6)[0]
    >>> FE.add_dirichlet_boundary_condition(node_ids=left_nodes, dofs=np.array([[1,1]]), rhs=0)
    >>> 
    >>> # Apply downward load on right edge
    >>> right_nodes = np.where(np.abs(mesh.nodes[:, 0] - 2.0) < 1e-6)[0]
    >>> FE.add_point_forces(node_ids=right_nodes, forces=np.array([[0, -1.0]]))
    >>> 
    >>> # Solve
    >>> U, residual = FE.solve(rho=np.ones(len(mesh.elements)) * 0.5)
    >>> print(f\"Residual: {residual:.2e}\")
    """
    def __init__(self, 
                 mesh: Union[StructuredMesh2D, StructuredMesh3D, GeneralMesh],
                 kernel: Union[StructuredStiffnessKernel, GeneralStiffnessKernel, UniformStiffnessKernel],
                 solver: Union[CHOLMOD, CG, BiCGSTAB, GMRES, SPLU, SPSOLVE, MultiGrid]):
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
        
        Examples
        --------
        >>> # Fix all DOFs at left edge
        >>> left_nodes = np.where(mesh.nodes[:, 0] < 1e-6)[0]
        >>> FE.add_dirichlet_boundary_condition(node_ids=left_nodes, rhs=0)
        >>> 
        >>> # Fix only y-displacement at bottom
        >>> bottom_nodes = np.where(mesh.nodes[:, 1] < 1e-6)[0]
        >>> FE.add_dirichlet_boundary_condition(node_ids=bottom_nodes, dofs=np.array([[0,1]]), rhs=0)
        >>> 
        >>> # Prescribe non-zero displacement
        >>> FE.add_dirichlet_boundary_condition(node_ids=[10], dofs=np.array([[1,0]]), rhs=np.array([[0.1, 0]]))
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
        
        Examples
        --------
        >>> # Apply downward force at right edge
        >>> right_nodes = np.where(np.abs(mesh.nodes[:, 0] - 2.0) < 1e-6)[0]
        >>> FE.add_point_forces(node_ids=right_nodes, forces=np.array([[0, -1.0]]))
        >>> 
        >>> # Apply force at specific location
        >>> FE.add_point_forces(positions=np.array([[1.0, 0.5]]), forces=np.array([[10.0, 0]]))
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
        self.d_rhs[:] = np.nan
        
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
        Solve finite element system: K(rho) @ U = F.
        
        Parameters
        ----------
        rho : ndarray, optional
            Element density variables, shape (n_elements,). If None, uses rho=1 (full density)
            
        Returns
        -------
        U : ndarray
            Displacement vector, shape (n_nodes * dof,). Interleaved DOFs: [ux0, uy0, ux1, uy1, ...]
        residual : float
            Normalized residual ||K@U - F|| / ||F|| for solution validation
            
        Notes
        -----
        - residual < 1e-5 indicates accurate solution
        - residual > 1e-2 indicates ill-conditioned system (check Problem.ill_conditioned())
        - Solver reuses factorization if available (e.g., CHOLMOD)
        - For topology optimization, rho represents material distribution
        
        Examples
        --------
        >>> rho = np.ones(len(mesh.elements)) * 0.5  # 50% density everywhere
        >>> U, residual = FE.solve(rho=rho)
        >>> if residual > 1e-3:
        >>>     print("Warning: Poor convergence!")
        >>> 
        >>> # Extract displacements
        >>> U_xy = U.reshape(-1, 2)  # Shape: (n_nodes, 2) for 2D
        >>> u_x = U_xy[:, 0]
        >>> u_y = U_xy[:, 1]
        """
        
        if rho is not None and rho.shape[0] != self.nel:
            raise ValueError("rho must have the same length as the number of elements in the mesh.")
        
        if rho is None:
            rho = np.ones(self.nel, dtype=self.dtype)
        
        self.kernel.set_rho(rho)
        U, residual = self.solver.solve(self.rhs, use_last=True)
        
        return U, residual