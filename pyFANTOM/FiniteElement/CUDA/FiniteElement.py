from ..FiniteElement import FiniteElement as FE
from ...geom.CUDA._mesh import (
    CuGeneralMesh as GeneralMesh,
    CuStructuredMesh2D as StructuredMesh2D,
    CuStructuredMesh3D as StructuredMesh3D)
from ...stiffness.CUDA._FEA import StructuredStiffnessKernel, GeneralStiffnessKernel, UniformStiffnessKernel
from ...solvers.CUDA._solvers import CG, GMRES, SPSOLVE, MultiGrid
from ...visualizers._2d import plot_mesh_2D, plot_problem_2D, plot_field_2D
from ...visualizers._3d import plot_problem_3D, plot_mesh_3D, plot_field_3D
from typing import Optional, Union, List
from scipy.spatial import KDTree
import cupy as np


class FiniteElement(FE):
    """
    CUDA-accelerated finite element analysis engine.
    
    GPU version of FiniteElement using CuPy for all computations. Provides identical
    API to CPU version but with 5-10x performance improvement for large problems.
    All arrays stored on GPU memory.
    
    Parameters
    ----------
    mesh : StructuredMesh2D, StructuredMesh3D, or GeneralMesh
        CUDA mesh defining geometry and physics
    kernel : StructuredStiffnessKernel, GeneralStiffnessKernel, or UniformStiffnessKernel
        CUDA stiffness assembly kernel
    solver : CG, GMRES, SPSOLVE, or MultiGrid
        CUDA linear system solver
        
    Attributes
    ----------
    mesh : Mesh
        Associated CUDA mesh
    kernel : StiffnessKernel
        Stiffness assembly kernel
    solver : Solver
        Linear solver
    rhs : cupy.ndarray
        Right-hand side force vector on GPU, shape (n_nodes * dof,)
    d_rhs : cupy.ndarray
        Prescribed displacement values for Dirichlet BCs, shape (n_nodes * dof,)
    dof : int
        Degrees of freedom per node (2 for 2D, 3 for 3D)
    is_3D : bool
        True for 3D problems
    nel : int
        Number of elements
        
    Notes
    -----
    - All arrays stored as CuPy arrays on GPU
    - KDTree searches performed on CPU (temporary transfer)
    - Visualization methods transfer data to CPU automatically
    - Requires CUDA-capable GPU and CuPy
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import *
    >>> mesh = StructuredMesh2D(nx=256, ny=128, lx=2.0, ly=1.0)
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> solver = CG(kernel=kernel)
    >>> FE = FiniteElement(mesh=mesh, kernel=kernel, solver=solver)
    >>> 
    >>> # Apply BCs and loads
    >>> FE.add_dirichlet_boundary_condition(node_ids=fixed_nodes, rhs=0)
    >>> FE.add_point_forces(node_ids=load_nodes, forces=cp.array([[0, -1.0]]))
    >>> 
    >>> # Solve
    >>> U, residual = FE.solve(rho=cp.ones(nel) * 0.5)
    """
    def __init__(self, 
                 mesh: Union[StructuredMesh2D, StructuredMesh3D, GeneralMesh],
                 kernel: Union[StructuredStiffnessKernel, GeneralStiffnessKernel, UniformStiffnessKernel],
                 solver: Union[CG, GMRES, SPSOLVE, MultiGrid]):
        """
        Initialize CUDA finite element analysis engine.
        
        Parameters
        ----------
        mesh : StructuredMesh2D, StructuredMesh3D, or GeneralMesh
            CUDA mesh defining geometry and physics
        kernel : StructuredStiffnessKernel, GeneralStiffnessKernel, or UniformStiffnessKernel
            CUDA stiffness assembly kernel
        solver : CG, GMRES, SPSOLVE, or MultiGrid
            CUDA linear system solver
            
        Notes
        -----
        Initializes force vector (rhs) and Dirichlet BC storage (d_rhs) on GPU.
        All arrays remain on GPU throughout computation.
        """
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
                                        rhs: Union[float,np.ndarray] = 0.):
        """
        Apply Dirichlet (fixed displacement) boundary conditions.
        
        Parameters
        ----------
        node_ids : cp.ndarray, optional
            Node indices to constrain on GPU, shape (n_nodes,)
        positions : cp.ndarray, optional
            Physical coordinates to constrain (uses KDTree search on CPU), shape (n_nodes, spatial_dim)
        dofs : cp.ndarray, optional
            DOF mask per node, shape (n_nodes, dof) with 1=constrained, 0=free.
            If None, all DOFs at specified nodes are constrained
        rhs : float or cp.ndarray, optional
            Prescribed displacement values. Scalar for zero displacement, or array shape (n_nodes, dof)
            
        Notes
        -----
        - Provide either node_ids OR positions, not both
        - dofs array: [[1,0]] constrains only x-displacement in 2D
        - Multiple calls accumulate constraints
        - Modifies kernel.constraints boolean array on GPU
        - KDTree search performed on CPU (nodes transferred temporarily)
        """
        
        if not node_ids is None:
            node_ids = np.array(node_ids)
        
        if not positions is None:
            positions = np.array(positions)
            
        if dofs is not None:
            dofs = np.array(dofs)
            
        if not isinstance(rhs, float):
            rhs = np.array(rhs, dtype=self.dtype)
        
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
                self.KDTree = KDTree(self.mesh.nodes.get())
                
            _, node_ids = self.KDTree.query(positions.get())
            
            self.add_dirichlet_boundary_condition(node_ids=node_ids, dofs=dofs, rhs=rhs)
            
    def add_neumann_boundary_condition(self, **kwargs):
        """
        Apply Neumann boundary conditions (not implemented).
        
        Raises
        ------
        NotImplementedError
            Always raised. Neumann boundary conditions not yet implemented for CUDA FEA.
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
        forces : cp.ndarray
            Force vectors on GPU, shape (n_forces, dof) where dof is 2 for 2D or 3 for 3D.
            For 2D: [[fx, fy]], for 3D: [[fx, fy, fz]]
        node_ids : cp.ndarray, optional
            Node indices for force application on GPU, shape (n_forces,)
        positions : cp.ndarray, optional
            Physical coordinates for force application (uses KDTree search on CPU)
            
        Notes
        -----
        - Provide either node_ids OR positions
        - Multiple calls accumulate forces
        - Forces are automatically set to prescribed values at Dirichlet nodes
        - Units should match physics model (e.g., Newtons for SI units)
        - KDTree search performed on CPU (nodes transferred temporarily)
        """
        
        if not node_ids is None:
            node_ids = np.array(node_ids)
        
        if not positions is None:
            positions = np.array(positions)
            
        forces = np.array(forces)
        
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
                self.KDTree = KDTree(self.mesh.nodes.get())
                
            _, node_ids = self.KDTree.query(positions.get())
            
            self.add_point_forces(forces=forces, node_ids=node_ids)
            
    def reset_forces(self):
        """
        Clear all applied forces.
        
        Sets the right-hand side force vector to zero on GPU. Useful when reconfiguring
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
        - GPU arrays are transferred to CPU for visualization (.get() calls)
        - Shows mesh elements, fixed nodes, and applied forces
        """
        if self.is_3D:
            return plot_problem_3D(
                self.mesh.nodes.get(),
                self.mesh.elements.get(),
                f = self.rhs.reshape(-1, self.mesh.dof).get(),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof).get(),
                **kwargs)
        else:
            return plot_problem_2D(
                self.mesh.nodes.get(),
                self.mesh.elements.get(),
                f = self.rhs.reshape(-1, self.mesh.dof).get(),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof).get(),
                ax=ax,
                **kwargs)
            
    def visualize_density(self, rho, ax=None, **kwargs):
        """
        Visualize density distribution (optimization result).
        
        Parameters
        ----------
        rho : cp.ndarray or ndarray
            Element densities on GPU or CPU, shape (n_elements,) or (n_elements, n_materials)
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
        - GPU arrays are automatically transferred to CPU for visualization
        - Color-codes elements by density value (0=void, 1=solid)
        """
        if self.is_3D:
            return plot_problem_3D(
                self.mesh.nodes.get(),
                self.mesh.elements.get(),
                f = self.rhs.reshape(-1, self.mesh.dof).get(),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof).get(),
                rho = rho.get() if isinstance(rho, np.ndarray) else rho,
                **kwargs)
        else:
            return plot_problem_2D(
                self.mesh.nodes.get(),
                self.mesh.elements.get(),
                f = self.rhs.reshape(-1, self.mesh.dof).get(),
                c = self.kernel.constraints.reshape(-1, self.mesh.dof).get(),
                ax=ax,
                rho = rho.get() if isinstance(rho, np.ndarray) else rho,
                **kwargs)
            
    def visualize_field(self, field, ax=None, rho=None, **kwargs):
        """
        Visualize scalar or vector field (displacement, stress, strain, etc.).
        
        Parameters
        ----------
        field : cp.ndarray or ndarray
            Field values to plot on GPU or CPU. Shape depends on field type:
            - Scalar: (n_elements,) or (n_nodes,)
            - Vector: (n_nodes, dof) for displacement-like fields
        ax : matplotlib.axes.Axes, optional
            Existing axes to plot on (2D only). If None, creates new figure
        rho : cp.ndarray or ndarray, optional
            Density mask to hide void regions, shape (n_elements,)
        **kwargs
            Additional arguments passed to plot_field_2D/3D
            
        Returns
        -------
        matplotlib.axes.Axes or k3d.Plot
            Plot object (2D: matplotlib axes, 3D: k3d plot)
            
        Notes
        -----
        - GPU arrays are automatically transferred to CPU for visualization
        - Common fields: displacement, Von Mises stress, strain energy
        - If rho provided, elements with rho < 0.5 are hidden
        """
        if self.is_3D:
            return plot_field_3D(
                self.mesh.nodes.get(),
                self.mesh.elements.get(),
                field.get() if isinstance(field, np.ndarray) else field,
                rho=rho.get() if isinstance(rho, np.ndarray) else rho,
                **kwargs)
        else:
            return plot_field_2D(
                self.mesh.nodes.get(),
                self.mesh.elements.get(),
                field.get() if isinstance(field, np.ndarray) else field,
                rho=rho.get() if isinstance(rho, np.ndarray) else rho,
                ax=ax,
                **kwargs)
    
    def solve(self, rho=None):
        """
        Solve finite element system: K(rho) @ U = F on GPU.
        
        Parameters
        ----------
        rho : cp.ndarray, optional
            Element density variables on GPU, shape (n_elements,). If None, uses rho=1 (full density)
            
        Returns
        -------
        U : cp.ndarray
            Displacement vector on GPU, shape (n_nodes * dof,). Interleaved DOFs: [ux0, uy0, ux1, uy1, ...]
        residual : float
            Normalized residual ||K@U - F|| / ||F|| for solution validation
            
        Notes
        -----
        - residual < 1e-5 indicates accurate solution
        - residual > 1e-2 indicates ill-conditioned system
        - All operations performed on GPU for maximum performance
        - Solver reuses factorization if available
        - For topology optimization, rho represents material distribution
        """
        if rho is not None and rho.shape[0] != self.nel:
            raise ValueError("rho must have the same length as the number of elements in the mesh.")
        
        if rho is None:
            rho = np.ones(self.nel, dtype=self.dtype)
        
        self.kernel.set_rho(rho)
        U, residual = self.solver.solve(self.rhs, use_last=True)
        
        return U, residual