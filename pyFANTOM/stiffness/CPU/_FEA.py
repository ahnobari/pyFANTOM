from ...core.CPU._ops import (
    get_diagonal_node_basis,
    get_diagonal_node_basis_full,
    get_diagonal_node_basis_flat,
    process_dk,
    process_dk_full,
    process_dk_flat,
    mat_vec_node_basis_parallel,
    mat_vec_node_basis_parallel_full,
    mat_vec_node_basis_parallel_flat,
    mat_vec_node_basis_parallel_wcon,
    mat_vec_node_basis_parallel_full_wcon,
    mat_vec_node_basis_parallel_flat_wcon,
    matmat_node_basis_prallel,
    matmat_node_basis_prallel_,
    matmat_node_basis_full_prallel,
    matmat_node_basis_full_prallel_,
    matmat_node_basis_flat_prallel,
    matmat_node_basis_flat_prallel_,
    matmat_node_basis_flat_prallel_)
from ...geom.CPU._mesh import StructuredMesh, GeneralMesh
import numpy as np
from scipy.sparse import csr_matrix, eye, issparse

class StiffnessKernel:
    """
    Base class for stiffness matrix assembly kernels.
    
    Provides matrix-free and explicit matrix operations for finite element stiffness
    matrices. Supports design-dependent stiffness via density variables (rho).
    
    Attributes
    ----------
    shape : tuple
        Global stiffness matrix dimensions (n_dof, n_dof)
    has_rho : bool
        True if design variables have been set
    rho : ndarray
        Current design variables (densities), shape (n_elements,)
    constraints : ndarray (bool)
        Boolean array marking constrained DOFs
    has_cons : bool
        True if boundary conditions have been applied
        
    Methods
    -------
    set_rho(rho)
        Set design variables for matrix-free operations
    dot(rhs)
        Matrix-vector product K(rho) @ rhs
    construct(rho)
        Build explicit CSR matrix representation
    diagonal(rho=None)
        Extract diagonal of stiffness matrix
    add_constraints(dof_indices)
        Apply Dirichlet boundary conditions
    process_grad(U)
        Compute element-wise sensitivities dK/drho : U
    reset()
        Clear all state (rho, constraints, factorizations)
        
    Notes
    -----
    - Matrix-free mode: Use dot() after set_rho() for O(n) memory
    - Explicit mode: Use construct() to build CSR matrix for direct solvers
    - Boundary conditions: Modify constraints array, zero rows/columns in matrix
    - Subclasses implement _matvec(), _matmat() for specific assembly strategies
    
    Examples
    --------
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> kernel.set_rho(rho)
    >>> u = kernel.dot(f)  # Matrix-free K @ f
    >>> K = kernel.construct(rho)  # Explicit matrix
    """
    def __init__(self):
        self.has_rho = False
        self.rho = None
        self.shape = None
        self.matvec = self.dot
        self.rmatvec = self.dot
        self.matmat = self.dot
        self.rmatmat = self.dot
        
    def construct(self, rho):
        """
        Build explicit CSR sparse matrix representation.
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
            
        Returns
        -------
        csr_matrix
            Sparse stiffness matrix in CSR format
            
        Notes
        -----
        - Memory-intensive: O(nnz) storage
        - Required for direct solvers (CHOLMOD, SPLU)
        - Reuses sparsity pattern on subsequent calls (faster)
        """
        raise NotImplementedError("construct method must be implemented in subclasses.")
        
    def _matvec(self, rho, vec):
        """
        Matrix-vector product K(rho) @ vec (internal).
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
        vec : ndarray
            Input vector, shape (n_dof,)
            
        Returns
        -------
        ndarray
            Output vector, shape (n_dof,)
            
        Notes
        -----
        Must be implemented in subclasses.
        """
        raise NotImplementedError("_matvec method must be implemented in subclasses.")
        
    def _rmatvec(self, rho, vec):
        """
        Transpose matrix-vector product K(rho)^T @ vec (internal).
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
        vec : ndarray
            Input vector, shape (n_dof,)
            
        Returns
        -------
        ndarray
            Output vector, shape (n_dof,)
            
        Notes
        -----
        For symmetric K, same as _matvec. Must be implemented in subclasses.
        """
        raise NotImplementedError("_rmatvec method must be implemented in subclasses.")
        
    def _matmat(self, rho, mat):
        """
        Matrix-matrix product K(rho) @ mat (internal).
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
        mat : csr_matrix or ndarray
            Input matrix, shape (n_dof, n_cols)
            
        Returns
        -------
        csr_matrix or ndarray
            Output matrix, shape (n_dof, n_cols)
            
        Notes
        -----
        Must be implemented in subclasses.
        """
        raise NotImplementedError("_matmat method must be implemented in subclasses.")
        
    def _rmatmat(self, rho, mat):
        """
        Transpose matrix-matrix product K(rho)^T @ mat (internal).
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
        mat : csr_matrix or ndarray
            Input matrix, shape (n_dof, n_cols)
            
        Returns
        -------
        csr_matrix or ndarray
            Output matrix, shape (n_dof, n_cols)
            
        Notes
        -----
        For symmetric K, same as _matmat. Must be implemented in subclasses.
        """
        raise NotImplementedError("_rmatmat method must be implemented in subclasses.")
        
    def set_rho(self, rho):
        """
        Set design variables for subsequent matrix-free operations.
        
        Parameters
        ----------
        rho : ndarray
            Design variables (densities), shape (n_elements,)
            
        Notes
        -----
        After calling set_rho(), dot() can be used for matrix-free operations
        without passing rho each time.
        """
        self.rho = rho
        self.has_rho = True
        
    def dot(self, rhs):
        """
        Matrix-vector or matrix-matrix product K(rho) @ rhs.
        
        Parameters
        ----------
        rhs : ndarray or csr_matrix
            Right-hand side vector or matrix
            - Vector: shape (n_dof,)
            - Matrix: shape (n_dof, n_cols)
            
        Returns
        -------
        ndarray or csr_matrix
            Result of K @ rhs, same shape as rhs
            
        Raises
        ------
        ValueError
            If rho has not been set (call set_rho() first)
        ValueError
            If rhs shape doesn't match matrix dimensions
            
        Notes
        -----
        Requires rho to be set via set_rho() or passed to construct().
        """
        raise NotImplementedError("dot method must be implemented in subclasses.")
        
    def diagonal(self, rho=None):
        """
        Extract diagonal of stiffness matrix.
        
        Parameters
        ----------
        rho : ndarray, optional
            Design variables. If None, uses self.rho (must be set)
            
        Returns
        -------
        ndarray
            Diagonal entries, shape (n_dof,)
            
        Raises
        ------
        ValueError
            If rho is None and has_rho is False
            
        Notes
        -----
        Useful for Jacobi preconditioning or diagonal scaling.
        """
        raise NotImplementedError("diagonal method must be implemented in subclasses.")
        
    def reset(self):
        """
        Reset kernel state.
        
        Clears design variables, constraints, and any cached matrices or factorizations.
        Useful when starting a new optimization or changing problem setup.
        """
        self.has_rho = False
        self.rho = None
        self.CSR = None
        self.ptr = None
        self.has_been_constructed = False
        self.has_cons = False
        if hasattr(self, 'constraints'):
            self.constraints[:] = False
    
    def __matmul__(self, rhs):
        """Convenience: kernel @ rhs calls dot(rhs)."""
        return self.dot(rhs)

class StructuredStiffnessKernel(StiffnessKernel):
    """
    Optimized stiffness matrix assembly for structured meshes.
    
    Uses node-basis assembly with a single precomputed element stiffness matrix for maximum efficiency
    on uniform grid meshes. Supports matrix-free operations (matvec) and explicit CSR construction.
    
    Parameters
    ----------
    mesh : StructuredMesh
        StructuredMesh2D or StructuredMesh3D with uniform elements
        
    Attributes
    ----------
    K_single : ndarray
        Single element stiffness matrix used for all elements
    shape : tuple
        Global stiffness matrix dimensions (n_dof, n_dof)
    constraints : ndarray (bool)
        Boolean array marking constrained DOFs
    has_cons : bool
        True if boundary conditions have been applied
    non_con_map : ndarray
        Index map to non-constrained DOFs for reduced system
        
    Methods
    -------
    set_rho(rho)
        Set design variables for subsequent matrix-free operations
    dot(rhs)
        Matrix-vector product K @ rhs (requires rho to be set)
    construct(rho)
        Build explicit CSR matrix representation
    diagonal(rho=None)
        Extract diagonal of stiffness matrix
    add_constraints(dof_indices)
        Apply Dirichlet boundary conditions
    process_grad(U)
        Compute element-wise sensitivities dK/drho : U for adjoint method
        
    Notes
    -----
    - Fastest kernel for structured meshes (2-3x faster than UniformStiffnessKernel)
    - Automatically handles constrained DOFs by zeroing rows/columns
    - Matrix-free dot() operation uses optimized numba kernels
    - Use construct() only when explicit matrix is needed (e.g., for direct solvers)
    
    Examples
    --------
    >>> from pyFANTOM.CPU import StructuredMesh2D, StructuredStiffnessKernel
    >>> mesh = StructuredMesh2D(nx=64, ny=32, lx=2.0, ly=1.0)
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> kernel.add_constraints(np.array([0, 1, 2]))  # Fix first node
    >>> rho = np.ones(len(mesh.elements)) * 0.5
    >>> kernel.set_rho(rho)
    >>> u = kernel.dot(f)  # Matrix-free K @ f
    """
    def __init__(self, mesh: StructuredMesh):
        super().__init__()
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        self.K_single = mesh.K_single
        
        self.dtype = mesh.dtype
        
        self.elements_flat = self.elements.flatten()
        self.el_ids = np.arange(self.elements.shape[0], dtype=np.int32).repeat(self.elements.shape[1])
        self.sorter = np.argsort(self.elements_flat).astype(np.int32)
        self.node_ids = np.searchsorted(self.elements_flat, np.arange(self.nodes.shape[0],dtype=np.int32), sorter=self.sorter, side='left').astype(np.int32)
        self.elements_size = self.elements.shape[1]
        self.n_nodes = self.nodes.shape[0]
        
        self.has_been_constructed = False
        self.ptr = None
        
        self.max_con_count = np.unique(self.elements_flat, return_counts=True)[1].max() * self.dof * self.elements_size
        
        self.constraints = np.zeros(self.n_nodes*self.dof, dtype=np.bool_)
        
        self.shape = (self.n_nodes*self.dof, self.n_nodes*self.dof)
        
        self.has_cons = False
        self.idx_map = np.arange(self.n_nodes*self.dof, dtype=np.int32)
        self.mat_vec = np.zeros(self.n_nodes*self.dof, dtype=self.dtype)
    
    def diagonal(self, rho=None):
        """
        Extract diagonal of stiffness matrix.
        
        Parameters
        ----------
        rho : ndarray, optional
            Design variables. If None, uses self.rho (must be set via set_rho())
            
        Returns
        -------
        ndarray
            Diagonal entries, shape (n_dof,)
            
        Raises
        ------
        ValueError
            If rho is None and has_rho is False
        """
        if rho is None and not self.has_rho:
            raise ValueError("Rho has not been set. diagonal works only after setting rho or if rho is provided.")
        elif rho is None:
            return get_diagonal_node_basis(self.K_single, self.elements_flat, self.el_ids, self.rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
        else:
            return get_diagonal_node_basis(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
                
    def set_constraints(self, constraints):
        """
        Set Dirichlet boundary conditions (replaces existing constraints).
        
        Parameters
        ----------
        constraints : ndarray
            DOF indices to constrain, shape (n_constrained_dofs,)
            
        Notes
        -----
        Clears all existing constraints and sets only the specified DOFs.
        Use add_constraints() to add constraints incrementally.
        """
        self.constraints[:] = False
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def add_constraints(self, constraints):
        """
        Add Dirichlet boundary conditions (accumulates with existing constraints).
        
        Parameters
        ----------
        constraints : ndarray
            DOF indices to constrain, shape (n_constrained_dofs,)
            
        Notes
        -----
        Adds to existing constraints. Use set_constraints() to replace all constraints.
        """
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def process_grad(self, U):
        """
        Compute element-wise sensitivities for adjoint method.
        
        Parameters
        ----------
        U : ndarray
            Displacement solution vector, shape (n_dof,)
            
        Returns
        -------
        ndarray
            Element-wise sensitivities dK/drho : U, shape (n_elements,)
            
        Notes
        -----
        Computes -U^T @ dK/drho @ U for each element, used in compliance
        sensitivity analysis. Result is per-element contribution to objective gradient.
        """
        return process_dk(self.K_single, self.elements, U, self.dof, self.elements_size)
    
    def construct(self, rho):
        size = self.n_nodes*self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), parallel=True)
            self.ptr = np.copy(self.CSR.indptr)
        else:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), self.ptr, parallel=True)

        self.has_been_constructed = True
        
        return self.CSR

    def _matvec(self, rho, vec):
        if not self.has_cons:
            mat_vec_node_basis_parallel(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.mat_vec, self.dof, self.elements_size)
        else:
            mat_vec_node_basis_parallel_wcon(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.constraints, self.mat_vec, self.dof, self.elements_size)
        return self.mat_vec
    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)
    
    def _matmat(self, rho, mat, Cp=None, parallel=True):
        if not Cp is None:
            if not self.has_cons:
                return matmat_node_basis_prallel_(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp)
            else:
                out = matmat_node_basis_prallel_(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp, self.constraints)
                out.indices[out.indices < 0] = 0
                return out
            
        if not self.has_cons:
            return matmat_node_basis_prallel(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count)
        else:
            return matmat_node_basis_prallel(self.K_single, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count, self.constraints)
            
    def _rmatmat(self, rho, mat, Cp=None, parallel=True):
        return self._matmat(rho, mat, Cp, parallel)
    
    def dot(self, rhs):
        if self.has_rho:
            if isinstance(rhs, np.ndarray):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError("Shape of the input vector does not match the number of nodes and dof.")
            elif issparse(rhs):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    if isinstance(rhs, csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError("Shape of the input matrix does not match the number of nodes and dof.")
            else:
                raise NotImplementedError("Only numpy arrays as vectors and scipy sparse matrices are supported.")
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")

class UniformStiffnessKernel(StiffnessKernel):
    """
    Stiffness matrix assembly for general uniform meshes.
    
    Handles unstructured meshes where all elements have the same number of nodes and identical geometry,
    using element-specific stiffness matrices. More general than StructuredStiffnessKernel but less efficient.
    
    Parameters
    ----------
    mesh : GeneralMesh
        GeneralMesh with uniform element topology (mesh.is_uniform must be True)
        
    Attributes
    ----------
    Ks : ndarray
        Element stiffness matrices, shape (n_elements, dof_per_element, dof_per_element)
    shape : tuple
        Global stiffness matrix dimensions
    constraints : ndarray (bool)
        Boolean array for constrained DOFs
        
    Methods
    -------
    Same interface as StructuredStiffnessKernel
        
    Notes
    -----
    - Requires mesh.is_uniform == True (all elements same size)
    - Stores per-element stiffness matrices (more memory than StructuredStiffnessKernel)
    - Use for unstructured uniform meshes (e.g., triangular or tetrahedral)
    - For structured grids, prefer StructuredStiffnessKernel for better performance
    
    Examples
    --------
    >>> from pyFANTOM.CPU import GeneralMesh, UniformStiffnessKernel
    >>> # Triangular mesh
    >>> nodes = np.array([[0,0], [1,0], [0,1], [1,1]])
    >>> elements = np.array([[0,1,2], [1,3,2]])
    >>> mesh = GeneralMesh(nodes, elements)
    >>> kernel = UniformStiffnessKernel(mesh=mesh)
    """
    def __init__(self, mesh: GeneralMesh):
        super().__init__()
        
        if not mesh.is_uniform:
            raise ValueError("The mesh is not uniform, you should use GeneralStiffnessKernel instead.")
        
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        self.elements_flat = mesh.elements_flat
        self.el_ids = np.arange(self.elements.shape[0], dtype=np.int32).repeat(self.elements.shape[1])
        self.sorter = np.argsort(self.elements_flat).astype(np.int32)
        self.node_ids = np.searchsorted(self.elements_flat, np.arange(self.nodes.shape[0]), sorter=self.sorter, side='left').astype(np.int32)
        self.elements_size = self.elements.shape[1]
        self.n_nodes = self.nodes.shape[0]
        
        self.dtype = mesh.dtype
        
        self.Ks = mesh.Ks
        
        self.has_been_constructed = False
        self.ptr = None
        
        self.max_con_count = np.unique(self.elements_flat, return_counts=True)[1].max() * self.dof * self.elements_size
        
        self.constraints = np.zeros(self.n_nodes*self.dof, dtype=np.bool_)
        
        self.shape = (self.n_nodes*self.dof, self.n_nodes*self.dof)
        
        self.has_cons = False
        self.idx_map = np.arange(self.n_nodes*self.dof, dtype=np.int32)
        self.mat_vec = np.zeros(self.n_nodes*self.dof, dtype=self.dtype)
    
    def diagonal(self, rho=None):
        """
        Extract diagonal of stiffness matrix.
        
        Parameters
        ----------
        rho : ndarray, optional
            Design variables. If None, uses self.rho (must be set via set_rho())
            
        Returns
        -------
        ndarray
            Diagonal entries, shape (n_dof,)
            
        Raises
        ------
        ValueError
            If rho is None and has_rho is False
        """
        if rho is None and not self.has_rho:
            raise ValueError("Rho has not been set. diagonal works only after setting rho or if rho is provided.")
        elif rho is None:
            return get_diagonal_node_basis_full(self.Ks, self.elements_flat, self.el_ids, self.rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
        else:
            return get_diagonal_node_basis_full(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
    
    def set_constraints(self, constraints):
        """
        Set Dirichlet boundary conditions (replaces existing constraints).
        
        Parameters
        ----------
        constraints : ndarray
            DOF indices to constrain, shape (n_constrained_dofs,)
            
        Notes
        -----
        Clears all existing constraints and sets only the specified DOFs.
        Use add_constraints() to add constraints incrementally.
        """
        self.constraints[:] = False
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def add_constraints(self, constraints):
        """
        Add Dirichlet boundary conditions (accumulates with existing constraints).
        
        Parameters
        ----------
        constraints : ndarray
            DOF indices to constrain, shape (n_constrained_dofs,)
            
        Notes
        -----
        Adds to existing constraints. Use set_constraints() to replace all constraints.
        """
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def process_grad(self, U):
        """
        Compute element-wise sensitivities for adjoint method.
        
        Parameters
        ----------
        U : ndarray
            Displacement solution vector, shape (n_dof,)
            
        Returns
        -------
        ndarray
            Element-wise sensitivities dK/drho : U, shape (n_elements,)
            
        Notes
        -----
        Computes -U^T @ dK/drho @ U for each element, used in compliance
        sensitivity analysis. Result is per-element contribution to objective gradient.
        """
        return process_dk_full(self.Ks, self.elements, U, self.dof, self.elements_size)
    
    def construct(self, rho):
        size = self.n_nodes*self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), parallel=True)
            self.ptr = np.copy(self.CSR.indptr)
        else:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), self.ptr, parallel=True)

        self.has_been_constructed = True
        
        return self.CSR
    
    def _matvec(self, rho, vec):
        if not self.has_cons:
            mat_vec_node_basis_parallel_full(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.mat_vec, self.dof, self.elements_size)
        else:
            mat_vec_node_basis_parallel_full_wcon(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.constraints, self.mat_vec, self.dof, self.elements_size)

        return self.mat_vec
    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)
        
    def _matmat(self, rho, mat, Cp=None, parallel=True):
        if not Cp is None:
            if not self.has_cons:
                return matmat_node_basis_full_prallel_(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp)
            else:
                out = matmat_node_basis_full_prallel_(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp, self.constraints)
                out.indices[out.indices < 0] = 0
                return out
            
        if not self.has_cons:
            return matmat_node_basis_full_prallel(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count)
        else:
            out = matmat_node_basis_full_prallel(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count, self.constraints)
            out.indices[out.indices < 0] = 0
            return out
    
    def _rmatmat(self, rho, mat, Cp=None, parallel=True):
        return self._matmat(rho, mat, Cp, parallel)

    def dot(self, rhs):
        if self.has_rho:
            if isinstance(rhs, np.ndarray):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError("Shape of the input vector does not match the number of nodes and dof.")
            elif issparse(rhs):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    if isinstance(rhs, csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError("Shape of the input matrix does not match the number of nodes and dof.")
            else:
                raise NotImplementedError("Only numpy arrays as vectors and scipy sparse matrices are supported.")
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")
    
class GeneralStiffnessKernel(StiffnessKernel):
    """
    Stiffness matrix assembly for fully heterogeneous meshes.
    
    Most general kernel supporting meshes with elements of varying sizes and topologies.
    Uses flat storage with pointer arrays for memory-efficient representation.
    
    Parameters
    ----------
    mesh : GeneralMesh
        GeneralMesh with potentially heterogeneous elements (mesh.is_uniform can be False)
        
    Attributes
    ----------
    K_flat : ndarray
        Flattened element stiffness matrices
    K_ptr : ndarray
        Pointer array for accessing K_flat: K_e = K_flat[K_ptr[e]:K_ptr[e+1]]
    elements_flat : ndarray
        Flattened element connectivity
    elements_ptr : ndarray
        Pointer array for accessing elements
    element_sizes : ndarray
        Number of nodes per element, shape (n_elements,)
        
    Methods
    -------
    Same interface as StructuredStiffnessKernel
        
    Notes
    -----
    - Supports mixed element types (triangles + quads, tets + hexes, etc.)
    - Flat storage with pointers enables variable element sizes
    - Slower than Uniform/Structured kernels due to indirection
    - Automatically used if mesh.is_uniform == False
    
    Examples
    --------
    >>> from pyFANTOM.CPU import GeneralMesh, GeneralStiffnessKernel
    >>> # Mixed triangular and quad mesh
    >>> nodes = np.array([[0,0], [1,0], [0,1], [1,1], [2,0]])
    >>> elements = [np.array([0,1,2]), np.array([1,4,3,2])]  # Triangle + Quad
    >>> mesh = GeneralMesh(nodes, elements)
    >>> kernel = GeneralStiffnessKernel(mesh=mesh)
    """
    def __init__(self, mesh: GeneralMesh):
        super().__init__()
        
        if mesh.is_uniform:
            raise ValueError("The mesh is uniform, you should use UniformStiffnessKernel instead.")
        
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        self.elements_flat = mesh.elements_flat
        self.element_sizes = mesh.element_sizes
        self.K_flat = mesh.K_flat
        
        self.dtype = mesh.dtype
    
        self.K_ptr = mesh.K_ptr
        self.elements_ptr = mesh.elements_ptr
        
        self.el_ids = np.arange(self.elements.shape[0], dtype=np.int32).repeat(self.element_sizes)
        self.sorter = np.argsort(self.elements_flat).astype(np.int32)
        self.node_ids = np.searchsorted(self.elements_flat, np.arange(self.nodes.shape[0]), sorter=self.sorter, side='left').astype(np.int32)
        self.n_nodes = self.nodes.shape[0]
        
        self.has_been_constructed = False
        self.ptr = None
        
        self.max_con_count = np.unique(self.elements_flat, return_counts=True)[1].max() * self.dof * self.element_sizes.max()
        
        self.constraints = np.zeros(self.n_nodes*self.dof, dtype=np.bool_)
        
        self.shape = (self.n_nodes*self.dof, self.n_nodes*self.dof)
        
        self.has_cons = False
        self.idx_map = np.arange(self.n_nodes*self.dof, dtype=np.int32)
    
    def diagonal(self, rho=None):
        """
        Extract diagonal of stiffness matrix.
        
        Parameters
        ----------
        rho : ndarray, optional
            Design variables. If None, uses self.rho (must be set via set_rho())
            
        Returns
        -------
        ndarray
            Diagonal entries, shape (n_dof,)
            
        Raises
        ------
        ValueError
            If rho is None and has_rho is False
        """
        if rho is None and not self.has_rho:
            raise ValueError("Rho has not been set. diagonal works only after setting rho or if rho is provided.")
        elif rho is None:
            return get_diagonal_node_basis_flat(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, self.rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.constraints)
        else:
            return get_diagonal_node_basis_flat(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.constraints)
    
    def set_constraints(self, constraints):
        """
        Set Dirichlet boundary conditions (replaces existing constraints).
        
        Parameters
        ----------
        constraints : ndarray
            DOF indices to constrain, shape (n_constrained_dofs,)
            
        Notes
        -----
        Clears all existing constraints and sets only the specified DOFs.
        Use add_constraints() to add constraints incrementally.
        """
        self.constraints[:] = False
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def add_constraints(self, constraints):
        """
        Add Dirichlet boundary conditions (accumulates with existing constraints).
        
        Parameters
        ----------
        constraints : ndarray
            DOF indices to constrain, shape (n_constrained_dofs,)
            
        Notes
        -----
        Adds to existing constraints. Use set_constraints() to replace all constraints.
        """
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def process_grad(self, U):
        """
        Compute element-wise sensitivities for adjoint method.
        
        Parameters
        ----------
        U : ndarray
            Displacement solution vector, shape (n_dof,)
            
        Returns
        -------
        ndarray
            Element-wise sensitivities dK/drho : U, shape (n_elements,)
            
        Notes
        -----
        Computes -U^T @ dK/drho @ U for each element, used in compliance
        sensitivity analysis. Result is per-element contribution to objective gradient.
        """
        return process_dk_flat(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, U, self.dof)
    
    def construct(self, rho):
        size = self.n_nodes*self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), parallel=True)
            self.ptr = np.copy(self.CSR.indptr)
        else:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), self.ptr, parallel=True)

        self.has_been_constructed = True
        
        return self.CSR
    
    def _matvec(self, rho, vec):
        if not self.has_cons:
            return mat_vec_node_basis_parallel_flat(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.dof)
        else:
            return mat_vec_node_basis_parallel_flat_wcon(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.constraints, self.dof)
    
    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)
    
    def _matmat(self, rho, mat, Cp=None, parallel=False):
        if not Cp is None:
            if not self.has_cons:
                return matmat_node_basis_flat_prallel_(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, mat, Cp)
            else:
                out = matmat_node_basis_flat_prallel_(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, mat, Cp, self.constraints)
                out.indices[out.indices < 0] = 0
                return out
        if not self.has_cons:
            return matmat_node_basis_flat_prallel(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, mat, parallel, self.max_con_count)
        else:
            out = matmat_node_basis_flat_prallel(self.K_flat, self.elements_flat, self.K_ptr, self.elements_ptr, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, mat, parallel, self.max_con_count, self.constraints)
            out.indices[out.indices < 0] = 0
            return out

    def _rmatmat(self, rho, mat, Cp=None, parallel=False):
        return self._matmat(rho, mat, Cp, parallel)
    
    def dot(self, rhs):
        if self.has_rho:
            if isinstance(rhs, np.ndarray):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError("Shape of the input vector does not match the number of nodes and dof.")
            elif issparse(rhs):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    if isinstance(rhs, csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError("Shape of the input matrix does not match the number of nodes and dof.")
            else:
                raise NotImplementedError("Only numpy arrays as vectors and scipy sparse matrices are supported.")
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")

class NLUniformStiffnessKernel(StiffnessKernel):
    """
    Stiffness matrix assembly for nonlinear elasticity with uniform meshes.
    
    Variant of UniformStiffnessKernel for geometrically nonlinear problems where element
    stiffness matrices vary with deformation. Stores full per-element stiffness matrices
    that are updated during Newton-Raphson iterations.
    
    Parameters
    ----------
    mesh : GeneralMesh
        GeneralMesh with uniform element topology (mesh.is_uniform must be True)
        
    Attributes
    ----------
    Ks : ndarray
        Element stiffness matrices (updated per iteration), shape (n_elements, dof_per_element, dof_per_element)
        
    Notes
    -----
    - Used with NLElasticity physics for large deformation problems
    - Element stiffness matrices are recomputed each nonlinear iteration
    - Requires full storage of Ks (memory-intensive for large meshes)
    - Use with NLFiniteElement for Newton-Raphson solution
    
    Examples
    --------
    >>> from pyFANTOM.CPU import GeneralMesh, NLUniformStiffnessKernel
    >>> from pyFANTOM import NLElasticity
    >>> physics = NLElasticity(E=1.0, nu=0.3)
    >>> mesh = GeneralMesh(nodes, elements, physics=physics)
    >>> kernel = NLUniformStiffnessKernel(mesh=mesh)
    """
    def __init__(self, mesh: GeneralMesh):
        super().__init__()
        
        if not mesh.is_uniform:
            raise ValueError("The mesh is not uniform, you should use GeneralStiffnessKernel instead.")
        
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        self.elements_flat = mesh.elements_flat
        self.el_ids = np.arange(self.elements.shape[0], dtype=np.int32).repeat(self.elements.shape[1])
        self.sorter = np.argsort(self.elements_flat).astype(np.int32)
        self.node_ids = np.searchsorted(self.elements_flat, np.arange(self.nodes.shape[0]), sorter=self.sorter, side='left').astype(np.int32)
        self.elements_size = self.elements.shape[1]
        self.n_nodes = self.nodes.shape[0]
        
        self.dtype = mesh.dtype
        
        self.Ks = mesh.Ks
        
        self.has_been_constructed = False
        self.ptr = None
        
        self.max_con_count = np.unique(self.elements_flat, return_counts=True)[1].max() * self.dof * self.elements_size
        
        self.constraints = np.zeros(self.n_nodes*self.dof, dtype=np.bool_)
        
        self.shape = (self.n_nodes*self.dof, self.n_nodes*self.dof)
        
        self.has_cons = False
        self.idx_map = np.arange(self.n_nodes*self.dof, dtype=np.int32)
        self.mat_vec = np.zeros(self.n_nodes*self.dof, dtype=self.dtype)
        

    
    def diagonal(self, rho=None):
        """
        Extract diagonal of tangent stiffness matrix.
        
        Parameters
        ----------
        rho : ndarray, optional
            Design variables. If None, uses self.rho (must be set via set_rho())
            
        Returns
        -------
        ndarray
            Diagonal entries, shape (n_dof,)
            
        Raises
        ------
        ValueError
            If rho is None and has_rho is False
            
        Notes
        -----
        Uses KTan (tangent stiffness) which is updated during Newton-Raphson iterations.
        Useful for Jacobi preconditioning in nonlinear solvers.
        """
        if rho is None and not self.has_rho:
            raise ValueError("Rho has not been set. diagonal works only after setting rho or if rho is provided.")
        elif rho is None:
            return get_diagonal_node_basis_full(self.Ks, self.elements_flat, self.el_ids, self.rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
        else:
            return get_diagonal_node_basis_full(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, self.constraints)
    
    def set_constraints(self, constraints):
        """
        Set Dirichlet boundary conditions (replaces existing constraints).
        
        Parameters
        ----------
        constraints : ndarray
            DOF indices to constrain, shape (n_constrained_dofs,)
            
        Notes
        -----
        Clears all existing constraints and sets only the specified DOFs.
        Use add_constraints() to add constraints incrementally.
        """
        self.constraints[:] = False
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def add_constraints(self, constraints):
        """
        Add Dirichlet boundary conditions (accumulates with existing constraints).
        
        Parameters
        ----------
        constraints : ndarray
            DOF indices to constrain, shape (n_constrained_dofs,)
            
        Notes
        -----
        Adds to existing constraints. Use set_constraints() to replace all constraints.
        """
        self.constraints[constraints] = True
        
        self.has_cons = True
        self.non_con_map = self.idx_map[~self.constraints]
    
    def process_grad(self, U):
        """
        Compute element-wise sensitivities (not implemented for nonlinear kernel).
        
        Parameters
        ----------
        U : ndarray
            Displacement solution vector, shape (n_dof,)
            
        Raises
        ------
        NotImplementedError
            Always raised. Sensitivity computation is handled in MinimumComplianceNL problem class.
            
        Notes
        -----
        For nonlinear problems, sensitivity computation requires special handling
        of the tangent stiffness matrix and is implemented in the problem class.
        """
        raise NotImplementedError("This is explicitly implemented in the MinimumComplianceNL problem class.")
        
        return process_dk_full(self.KTan, self.elements, U, self.dof, self.elements_size)
    
    def set_Ktan(self, KTan) -> None:
        """
        Set tangent stiffness matrix for nonlinear analysis.
        
        Parameters
        ----------
        KTan : ndarray
            Tangent stiffness matrices, shape (n_elements, dof_per_element, dof_per_element)
            
        Notes
        -----
        Called during Newton-Raphson iterations to update element stiffness matrices
        based on current deformation state. KTan is computed from NLElasticity physics.
        """
        self.KTan = KTan

    def construct(self, rho):
        """
        Build explicit CSR sparse matrix representation using tangent stiffness.
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
            
        Returns
        -------
        csr_matrix
            Sparse tangent stiffness matrix in CSR format
            
        Notes
        -----
        Uses KTan (set via set_Ktan()) instead of initial Ks. Required for direct
        solvers in nonlinear analysis. Reuses sparsity pattern on subsequent calls.
        """
        self.Ks = self.KTan
        size = self.n_nodes*self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), parallel=True)
            self.ptr = np.copy(self.CSR.indptr)
        else:
            self.CSR = self._matmat(rho, eye(size, format='csr',dtype=self.dtype), self.ptr, parallel=True)
        
        return self.CSR
    
    
    def _matvec(self, rho, vec):
        """
        Matrix-vector product KTan(rho) @ vec (internal).
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
        vec : ndarray
            Input vector, shape (n_dof,)
            
        Returns
        -------
        ndarray
            Output vector, shape (n_dof,)
            
        Notes
        -----
        Uses KTan (tangent stiffness) which is updated during Newton-Raphson iterations.
        """
        if not self.has_cons:
            mat_vec_node_basis_parallel_full(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.mat_vec, self.dof, self.elements_size)
        else:
            mat_vec_node_basis_parallel_full_wcon(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, vec, self.constraints, self.mat_vec, self.dof, self.elements_size)

        return self.mat_vec
    def _rmatvec(self, rho, vec):
        """
        Transpose matrix-vector product KTan(rho)^T @ vec (internal).
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
        vec : ndarray
            Input vector, shape (n_dof,)
            
        Returns
        -------
        ndarray
            Output vector, shape (n_dof,)
            
        Notes
        -----
        For symmetric KTan, same as _matvec. Uses tangent stiffness matrix.
        """
        return self._matvec(rho, vec)
        
    def _matmat(self, rho, mat, Cp=None, parallel=True):
        """
        Matrix-matrix product KTan(rho) @ mat (internal).
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
        mat : csr_matrix or ndarray
            Input matrix, shape (n_dof, n_cols)
        Cp : ndarray, optional
            Precomputed column pointer array for CSR matrix
        parallel : bool, optional
            Use parallel assembly (default: True)
            
        Returns
        -------
        csr_matrix or ndarray
            Output matrix, shape (n_dof, n_cols)
            
        Notes
        -----
        Uses KTan (tangent stiffness) for nonlinear analysis. Reuses sparsity pattern
        if Cp is provided for faster assembly.
        """
        if not Cp is None:
            if not self.has_cons:
                return matmat_node_basis_full_prallel_(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp)
            else:
                out = matmat_node_basis_full_prallel_(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, Cp, self.constraints)
                out.indices[out.indices < 0] = 0
                return out
            
        if not self.has_cons:
            return matmat_node_basis_full_prallel(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count)
        else:
            out = matmat_node_basis_full_prallel(self.Ks, self.elements_flat, self.el_ids, rho, self.sorter, self.node_ids, self.n_nodes, self.dof, self.elements_size, mat, parallel, self.max_con_count, self.constraints)
            out.indices[out.indices < 0] = 0
            return out
    
    def _rmatmat(self, rho, mat, Cp=None, parallel=True):
        """
        Transpose matrix-matrix product KTan(rho)^T @ mat (internal).
        
        Parameters
        ----------
        rho : ndarray
            Design variables, shape (n_elements,)
        mat : csr_matrix or ndarray
            Input matrix, shape (n_dof, n_cols)
        Cp : ndarray, optional
            Precomputed column pointer array
        parallel : bool, optional
            Use parallel assembly (default: True)
            
        Returns
        -------
        csr_matrix or ndarray
            Output matrix, shape (n_dof, n_cols)
            
        Notes
        -----
        For symmetric KTan, same as _matmat. Uses tangent stiffness matrix.
        """
        return self._matmat(rho, mat, Cp, parallel)

    def dot(self, rhs):
        """
        Matrix-vector or matrix-matrix product KTan(rho) @ rhs.
        
        Parameters
        ----------
        rhs : ndarray or csr_matrix
            Right-hand side vector or matrix
            - Vector: shape (n_dof,)
            - Matrix: shape (n_dof, n_cols)
            
        Returns
        -------
        ndarray or csr_matrix
            Result of KTan @ rhs, same shape as rhs
            
        Raises
        ------
        ValueError
            If rho has not been set (call set_rho() first)
        ValueError
            If rhs shape doesn't match matrix dimensions
        NotImplementedError
            If rhs type not supported
            
        Notes
        -----
        Requires rho to be set via set_rho(). Uses KTan (tangent stiffness) which
        must be updated via set_Ktan() before each Newton-Raphson iteration.
        """
        if self.has_rho:
            if isinstance(rhs, np.ndarray):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError("Shape of the input vector does not match the number of nodes and dof.")
            elif issparse(rhs):
                if rhs.shape[0] == self.n_nodes*self.dof:
                    if isinstance(rhs, csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError("Shape of the input matrix does not match the number of nodes and dof.")
            else:
                raise NotImplementedError("Only numpy arrays as vectors and scipy sparse matrices are supported.")
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")
    