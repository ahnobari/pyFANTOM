from ...core.CUDA import *
from ...geom.CUDA._mesh import CuStructuredMesh2D, CuStructuredMesh3D, CuGeneralMesh
import cupy as cp
from typing import Union

class StiffnessKernel:
    """
    Base class for CUDA-accelerated stiffness matrix assembly kernels.
    
    Abstract base class defining interface for GPU-accelerated finite element stiffness
    matrix operations. Subclasses implement specific assembly strategies (structured,
    uniform, general) optimized for GPU computation using CuPy.
    
    Attributes
    ----------
    has_rho : bool
        Whether density variables have been set
    rho : cupy.ndarray
        Element density variables on GPU, shape (n_elements,)
    shape : tuple
        Matrix shape (n_dof, n_dof)
    matvec : callable
        Alias for dot() method
    rmatvec : callable
        Alias for dot() method (symmetric matrices)
    matmat : callable
        Alias for dot() method
    rmatmat : callable
        Alias for dot() method (symmetric matrices)
    mat_vec : cupy.ndarray
        Cached matrix-vector product result
    diag : cupy.ndarray
        Cached diagonal entries
    dk : cupy.ndarray
        Cached derivative matrices
        
    Notes
    -----
    - All data stored as CuPy arrays on GPU
    - Matrix-free operations (no explicit matrix storage)
    - Supports matrix-vector products via dot()
    - Subclasses must implement construct(), _matvec(), dot(), diagonal()
    """
    def __init__(self):
        """
        Initialize base stiffness kernel.
        
        Sets up attributes for density storage and matrix operation aliases.
        Subclasses should call super().__init__() and set mesh-specific attributes.
        """
        self.has_rho = False
        self.rho = None
        self.shape = None
        self.matvec = self.dot
        self.rmatvec = self.dot
        self.matmat = self.dot
        self.rmatmat = self.dot
        
        self.mat_vec = None
        self.diag = None
        self.dk = None

    def construct(self, rho):
        """
        Construct stiffness matrix representation (abstract method).
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
            
        Notes
        -----
        Subclasses must implement this to assemble stiffness matrix based on
        density distribution. May precompute matrix entries or prepare for
        matrix-free operations.
        """
        pass

    def _matvec(self, rho, vec):
        """
        Matrix-vector product K(rho) @ vec (abstract method).
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
        vec : cupy.ndarray
            Input vector, shape (n_dof,)
            
        Returns
        -------
        cupy.ndarray
            Output vector K(rho) @ vec, shape (n_dof,)
            
        Notes
        -----
        Subclasses must implement this for matrix-free operations.
        Called internally by dot() method.
        """
        pass

    def _rmatvec(self, rho, vec):
        """
        Transpose matrix-vector product K(rho)^T @ vec (abstract method).
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
        vec : cupy.ndarray
            Input vector, shape (n_dof,)
            
        Returns
        -------
        cupy.ndarray
            Output vector K(rho)^T @ vec, shape (n_dof,)
            
        Notes
        -----
        For symmetric stiffness matrices, same as _matvec(). Subclasses
        should implement accordingly.
        """
        pass

    def _matmat(self, rho, mat):
        """
        Matrix-matrix product K(rho) @ mat (abstract method).
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
        mat : cupy.ndarray
            Input matrix, shape (n_dof, n_cols)
            
        Returns
        -------
        cupy.ndarray
            Output matrix K(rho) @ mat, shape (n_dof, n_cols)
            
        Notes
        -----
        Subclasses may implement via repeated _matvec() calls or optimized
        batch operations.
        """
        pass

    def _rmatmat(self, rho, mat):
        """
        Transpose matrix-matrix product K(rho)^T @ mat (abstract method).
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
        mat : cupy.ndarray
            Input matrix, shape (n_dof, n_cols)
            
        Returns
        -------
        cupy.ndarray
            Output matrix K(rho)^T @ mat, shape (n_dof, n_cols)
            
        Notes
        -----
        For symmetric matrices, same as _matmat(). Subclasses should
        implement accordingly.
        """
        pass

    def set_rho(self, rho):
        """
        Set element density variables.
        
        Parameters
        ----------
        rho : ndarray or cupy.ndarray
            Element density variables, shape (n_elements,)
            
        Notes
        -----
        Converts input to CuPy array if needed. Does not trigger matrix
        reconstruction; call construct() after setting rho.
        """
        self.rho = cp.array(rho, dtype=rho.dtype, copy=False)
        self.has_rho = True

    def dot(self, rhs):
        """
        Matrix-vector product K @ rhs (abstract method).
        
        Parameters
        ----------
        rhs : cupy.ndarray
            Right-hand side vector, shape (n_dof,) or (n_dof, n_cols)
            
        Returns
        -------
        cupy.ndarray
            Result K @ rhs, same shape as rhs
            
        Notes
        -----
        Subclasses must implement this. Should use cached rho if available,
        otherwise raise error. Handles both vectors and matrices.
        """
        pass

    def reset(self):
        """
        Reset kernel state to initial condition.
        
        Clears density variables, cached matrices, and constraint information.
        Useful when reinitializing optimization or changing problem configuration.
        """
        self.has_rho = False
        self.rho = None
        self.CSR = None
        self.ptr = None
        self.has_been_constructed = False
        self.constraints[:] = False
        self.has_con = False
        self.mat_vec = None
        self.diag = None
        self.dk = None

    def diagonal(self):
        """
        Get diagonal entries of stiffness matrix (abstract method).
        
        Returns
        -------
        cupy.ndarray
            Diagonal entries, shape (n_dof,)
            
        Notes
        -----
        Subclasses must implement this. May compute on-the-fly or return
        cached diagonal if available. Used for preconditioning and diagnostics.
        """
        pass

    def __matmul__(self, rhs):
        """
        Matrix multiplication operator (K @ rhs).
        
        Parameters
        ----------
        rhs : cupy.ndarray
            Right-hand side vector or matrix
            
        Returns
        -------
        cupy.ndarray
            Result of matrix multiplication
            
        Notes
        -----
        Enables Python @ operator syntax: result = kernel @ vector
        """
        return self.dot(rhs)


class StructuredStiffnessKernel(StiffnessKernel):
    """
    CUDA-accelerated stiffness kernel for structured meshes.
    
    GPU version of CPU StructuredStiffnessKernel using CuPy. Uses node-basis assembly with
    a single precomputed element stiffness matrix for maximum efficiency on uniform grid meshes.
    Identical API to CPU version but operates entirely on GPU memory for maximum performance.
    
    Parameters
    ----------
    mesh : CuStructuredMesh2D or CuStructuredMesh3D
        CUDA-accelerated structured mesh with uniform elements
        
    Attributes
    ----------
    K_single : cupy.ndarray
        Single element stiffness matrix used for all elements (on GPU)
    shape : tuple
        Global stiffness matrix dimensions (n_dof, n_dof)
    constraints : cupy.ndarray (bool)
        Boolean array marking constrained DOFs (on GPU)
    has_cons : bool
        True if boundary conditions have been applied
    non_con_map : cupy.ndarray
        Index map to non-constrained DOFs for reduced system (on GPU)
        
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
    - Fastest kernel for structured meshes (2-3x faster than UniformStiffnessKernel on GPU)
    - All arrays stored as CuPy arrays on GPU
    - 5-10x faster than CPU for large 3D problems
    - Automatically handles constrained DOFs by zeroing rows/columns
    - Matrix-free dot() operation uses optimized CUDA kernels
    - Use construct() only when explicit matrix is needed (e.g., for direct solvers)
    - Requires CUDA-capable GPU and CuPy
    - Use with CUDA solvers (CG, MultiGrid)
    - Memory limited by GPU VRAM
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import StructuredMesh2D, StructuredStiffnessKernel
    >>> mesh = StructuredMesh2D(nx=256, ny=128, lx=2.0, ly=1.0)
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> kernel.set_rho(rho)
    >>> u = kernel.dot(f)  # Matrix-free K @ f on GPU
    """
    def __init__(self, mesh: Union[CuStructuredMesh2D, CuStructuredMesh3D]):
        super().__init__()
        self.mesh = mesh
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        self.K_single = mesh.K_single
        
        self.dtype = mesh.dtype

        # move to gpu and cast to 32 bit
        self.nodes = cp.array(self.nodes, dtype=self.nodes.dtype, copy=False)
        self.elements = cp.array(self.elements, dtype=self.elements.dtype, copy=False)
        self.K_single = cp.array(self.K_single, dtype=self.K_single.dtype, copy=False)

        self.elements_flat = self.elements.flatten()
        self.el_ids = cp.arange(self.elements.shape[0], dtype=cp.int32).repeat(
            self.elements.shape[1]
        )
        self.sorter = cp.argsort(self.elements_flat).astype(cp.int32)
        self.node_ids = cp.searchsorted(
            self.elements_flat,
            cp.arange(self.nodes.shape[0], dtype=cp.int32),
            sorter=self.sorter,
            side="left",
        ).astype(cp.int32)
        self.elements_size = self.elements.shape[1]
        self.n_nodes = self.nodes.shape[0]

        self.has_been_constructed = False
        self.ptr = None

        self.max_con_count = (self.elements_size * int(self.dof)) * cp.unique(
            self.elements_flat, return_counts=True
        )[1].max()

        self.constraints = cp.zeros(self.n_nodes * self.dof, dtype=cp.bool_)

        self.shape = (self.n_nodes * self.dof, self.n_nodes * self.dof)

        self.has_con = False
        self.idx_map = cp.arange(self.n_nodes * self.dof, dtype=cp.int32)

    def diagonal(self, rho=None):
        """
        Extract diagonal of assembled stiffness matrix.
        
        Parameters
        ----------
        rho : cp.ndarray, optional
            Design densities. If None, uses stored rho
            
        Returns
        -------
        diag : cp.ndarray
            Diagonal entries of K(rho)
            
        Notes
        -----
        - Used for Jacobi preconditioning in iterative solvers
        - Cached on subsequent calls with same rho
        """
        if rho is None and not self.has_rho:
            raise ValueError(
                "Rho has not been set. diagonal works only after setting rho or if rho is provided."
            )
        elif rho is None:
            self.diag =  get_diagonal_node_basis(
                self.K_single,
                self.elements_flat,
                self.el_ids,
                self.rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.elements_size,
                self.constraints,
                self.diag
            )
            return self.diag
        else:
            self.diag = get_diagonal_node_basis(
                self.K_single,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.elements_size,
                self.constraints,
                self.diag
            )
            return self.diag

    def set_constraints(self, constraints):
        """
        Set Dirichlet boundary conditions (replaces existing constraints).
        
        Parameters
        ----------
        constraints : cp.ndarray
            DOF indices to constrain on GPU, shape (n_constrained_dofs,)
            
        Notes
        -----
        Clears all existing constraints and sets only the specified DOFs.
        Use add_constraints() to add constraints incrementally.
        """
        self.constraints[:] = False
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def add_constraints(self, constraints):
        """
        Add Dirichlet boundary conditions (accumulates with existing constraints).
        
        Parameters
        ----------
        constraints : cp.ndarray
            DOF indices to constrain on GPU, shape (n_constrained_dofs,)
            
        Notes
        -----
        Adds to existing constraints. Use set_constraints() to replace all constraints.
        """
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def process_grad(self, U):
        """
        Compute element-wise compliance sensitivities dC/drho.
        
        Parameters
        ----------
        U : cp.ndarray
            Displacement vector from FEA solve on GPU, shape (n_dof,)
            
        Returns
        -------
        dk : cp.ndarray
            Element sensitivities on GPU (compliance derivative per element), shape (n_elements,)
            
        Notes
        -----
        - Computes: dk[e] = -U[e]^T @ K_e @ U[e]
        - Used for compliance minimization gradient
        - Negative sign convention for minimization
        - All operations performed on GPU
        """
        self.dk = process_dk(
            self.K_single, self.elements_flat, U, self.dof, self.elements_size, self.dk
        )
        return self.dk

    def construct(self, rho):
        """
        Build explicit CSR sparse matrix representation on GPU.
        
        Parameters
        ----------
        rho : cp.ndarray
            Design variables on GPU, shape (n_elements,)
            
        Returns
        -------
        cupyx.scipy.sparse.csr_matrix
            Sparse stiffness matrix in CSR format on GPU
            
        Notes
        -----
        - Memory-intensive: O(nnz) storage on GPU
        - Required for direct solvers (SPSOLVE)
        - Reuses sparsity pattern on subsequent calls (faster)
        """
        size = self.n_nodes * self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype)
            )
            self.ptr = self.CSR.indptr
        else:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype), self.ptr
            )
        self.has_been_constructed = True

        return self.CSR

    def _matvec(self, rho, vec):
        if not self.has_con:
            self.mat_vec = mat_vec_node_basis_parallel(
                self.K_single,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                self.elements_size,
                out=self.mat_vec
            )
            return self.mat_vec

        self.mat_vec = mat_vec_node_basis_parallel(
            self.K_single,
            self.elements_flat,
            self.el_ids,
            rho,
            self.sorter,
            self.node_ids,
            self.n_nodes,
            vec,
            self.dof,
            self.elements_size,
            self.constraints,
            self.mat_vec
        )
        return self.mat_vec

    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)

    def _matmat(self, rho, mat, Cp=None):
        if Cp is None:
            if not self.has_con:
                return matmat_node_basis_parallel(
                    self.K_single,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    self.max_con_count,
                )
            else:
                out = matmat_node_basis_parallel(
                    self.K_single,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    self.max_con_count,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out
        else:
            if not self.has_con:
                return matmat_node_basis_parallel_(
                    self.K_single,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    Cp
                )
            else:
                out = matmat_node_basis_parallel_(
                    self.K_single,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    Cp,
                    self.constraints
                )
                out.indices[out.indices < 0] = 0
                return out

    def _rmatmat(self, rho, mat, Cp=None):
        return self._matmat(rho, mat, Cp)

    def dot(self, rhs):
        """
        Matrix-free matrix-vector or matrix-matrix product K(rho) @ rhs.
        
        Parameters
        ----------
        rhs : cp.ndarray or cp.sparse.csr_matrix
            Right-hand side vector or matrix
            
        Returns
        -------
        result : cp.ndarray or cp.sparse.csr_matrix
            Product K(rho) @ rhs
            
        Raises
        ------
        ValueError
            If rho not set or shape mismatch
        NotImplementedError
            If rhs type not supported
            
        Notes
        -----
        - Requires rho set via set_rho() first
        - Matrix-free: No explicit matrix storage
        - Faster than constructing full matrix for large problems
        """
        if self.has_rho:
            if isinstance(rhs, cp.ndarray):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError(
                        "Shape of the input vector does not match the number of nodes and dof."
                    )
            elif cp.sparse.issparse(rhs):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    if isinstance(rhs, cp.sparse.csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError(
                        "Shape of the input matrix does not match the number of nodes and dof."
                    )
            else:
                raise NotImplementedError(
                    "Only numpy arrays as vectors and scipy sparse matrices are supported."
                )
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")

class UniformStiffnessKernel(StiffnessKernel):
    """
    CUDA-accelerated stiffness kernel for uniform unstructured meshes.
    
    GPU version of CPU UniformStiffnessKernel. Handles unstructured meshes where all elements
    have the same number of nodes and identical geometry, using element-specific stiffness matrices.
    More general than StructuredStiffnessKernel but less efficient. Identical API to CPU version
    but operates entirely on GPU memory.
    
    Parameters
    ----------
    mesh : CuGeneralMesh
        CUDA general mesh with uniform element topology (mesh.is_uniform must be True)
        
    Attributes
    ----------
    Ks : cupy.ndarray
        Element stiffness matrices, shape (n_elements, dof_per_element, dof_per_element) (on GPU)
    shape : tuple
        Global stiffness matrix dimensions (n_dof, n_dof)
    constraints : cupy.ndarray (bool)
        Boolean array for constrained DOFs (on GPU)
    has_cons : bool
        True if boundary conditions have been applied
        
    Methods
    -------
    Same interface as StructuredStiffnessKernel (set_rho, dot, construct, diagonal, etc.)
        
    Notes
    -----
    - Requires mesh.is_uniform == True (all elements same size)
    - Stores per-element stiffness matrices (more memory than StructuredStiffnessKernel)
    - Use for unstructured uniform meshes (e.g., triangular or tetrahedral)
    - For structured grids, prefer StructuredStiffnessKernel for better performance
    - All arrays stored as CuPy arrays on GPU
    - All operations on GPU using CuPy
    - Use with CUDA solvers (CG, MultiGrid) for best performance
    - Memory limited by GPU VRAM
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import GeneralMesh, UniformStiffnessKernel
    >>> import numpy as np
    >>> # Triangular mesh
    >>> nodes = np.array([[0,0], [1,0], [0,1], [1,1]])
    >>> elements = np.array([[0,1,2], [1,3,2]])
    >>> mesh = GeneralMesh(nodes, elements)
    >>> kernel = UniformStiffnessKernel(mesh=mesh)
    """
    def __init__(self, mesh: CuGeneralMesh):
        super().__init__()

        if not mesh.is_uniform:
            raise ValueError(
                "Mesh is not uniform, you should use GeneralStiffnessKernel instead."
            )
        self.mesh = mesh
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        
        self.dtype = mesh.dtype

        # move to gpu and cast appropriately
        self.nodes = cp.array(self.nodes, dtype=self.dtype, copy=False)
        self.elements = cp.array(self.elements, dtype=cp.int32, copy=False)
        self.Ks = cp.array(mesh.Ks, dtype=self.dtype, copy=False)
        
        self.elements_flat = self.elements.flatten()
        self.el_ids = cp.arange(self.elements.shape[0], dtype=cp.int32).repeat(
            self.elements.shape[1]
        )
        self.sorter = cp.argsort(self.elements_flat).astype(cp.int32)
        self.node_ids = cp.searchsorted(
            self.elements_flat,
            cp.arange(self.nodes.shape[0], dtype=cp.int32),
            sorter=self.sorter,
            side="left",
        ).astype(cp.int32)
        self.elements_size = self.elements.shape[1]
        self.n_nodes = self.nodes.shape[0]

        self.has_been_constructed = False
        self.ptr = None
        self.indices = None
        self.data = None

        self.max_con_count = (self.elements_size * int(self.dof)) * cp.unique(
            self.elements_flat, return_counts=True
        )[1].max()

        self.constraints = cp.zeros(self.n_nodes * self.dof, dtype=cp.bool_)
        self.shape = (self.n_nodes * self.dof, self.n_nodes * self.dof)

        self.has_con = False
        self.idx_map = cp.arange(self.n_nodes * self.dof, dtype=cp.int32)

    def diagonal(self, rho=None):
        """
        Extract diagonal of stiffness matrix.
        
        Parameters
        ----------
        rho : cp.ndarray, optional
            Design variables on GPU. If None, uses self.rho (must be set via set_rho())
            
        Returns
        -------
        cp.ndarray
            Diagonal entries on GPU, shape (n_dof,)
            
        Raises
        ------
        ValueError
            If rho is None and has_rho is False
            
        Notes
        -----
        Useful for Jacobi preconditioning or diagonal scaling in iterative solvers.
        """
        if rho is None and not self.has_rho:
            raise ValueError(
                "Rho has not been set. diagonal works only after setting rho or if rho is provided."
            )
        elif rho is None:
            self.diag =  get_diagonal_node_basis(
                self.Ks,
                self.elements_flat,
                self.el_ids,
                self.rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.elements_size,
                self.constraints,
                self.diag
            )
            return self.diag
        else:
            self.diag = get_diagonal_node_basis(
                self.Ks,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.elements_size,
                self.constraints,
                self.diag
            )
            return self.diag

    def set_constraints(self, constraints):
        """
        Set Dirichlet boundary conditions (replaces existing constraints).
        
        Parameters
        ----------
        constraints : cp.ndarray
            DOF indices to constrain on GPU, shape (n_constrained_dofs,)
            
        Notes
        -----
        Clears all existing constraints and sets only the specified DOFs.
        Use add_constraints() to add constraints incrementally.
        """
        self.constraints[:] = False
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def add_constraints(self, constraints):
        """
        Add Dirichlet boundary conditions (accumulates with existing constraints).
        
        Parameters
        ----------
        constraints : cp.ndarray
            DOF indices to constrain on GPU, shape (n_constrained_dofs,)
            
        Notes
        -----
        Adds to existing constraints. Use set_constraints() to replace all constraints.
        """
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def process_grad(self, U):
        """
        Compute element-wise compliance sensitivities dC/drho.
        
        Parameters
        ----------
        U : cp.ndarray
            Displacement vector from FEA solve on GPU, shape (n_dof,)
            
        Returns
        -------
        dk : cp.ndarray
            Element sensitivities on GPU, shape (n_elements,)
            
        Notes
        -----
        Computes -U^T @ dK/drho @ U for each element, used in compliance
        sensitivity analysis. All operations performed on GPU.
        """
        self.dk = process_dk_full(
            self.Ks, self.elements_flat, U, self.dof, self.elements_size, self.dk
        )
        
        return self.dk

    def construct(self, rho):
        """
        Build explicit CSR sparse matrix representation on GPU.
        
        Parameters
        ----------
        rho : cp.ndarray
            Design variables on GPU, shape (n_elements,)
            
        Returns
        -------
        cupyx.scipy.sparse.csr_matrix
            Sparse stiffness matrix in CSR format on GPU
            
        Notes
        -----
        - Memory-intensive: O(nnz) storage on GPU
        - Required for direct solvers (SPSOLVE)
        - Reuses sparsity pattern on subsequent calls (faster)
        """
        size = self.n_nodes * self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype)
            )
            self.ptr = self.CSR.indptr
        else:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype), self.ptr
            )
        self.has_been_constructed = True

        return self.CSR

    def _matvec(self, rho, vec):
        if not self.has_con:
            self.mat_vec = mat_vec_node_basis_parallel_full(
                self.Ks,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                self.elements_size,
                out=self.mat_vec
            )
            return self.mat_vec
        else:
            self.mat_vec = mat_vec_node_basis_parallel_full(
                self.Ks,
                self.elements_flat,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                self.elements_size,
                self.constraints,
                self.mat_vec
            )
            return self.mat_vec

    def _rmatvec(self, rho, vec):
        return self._matvec(rho, vec)

    def _matmat(self, rho, mat, Cp=None):
        if Cp is None:
            if not self.has_con:
                return matmat_node_basis_full_parallel(
                    self.Ks,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    self.max_con_count,
                )
            else:
                out = matmat_node_basis_full_parallel(
                    self.Ks,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    self.max_con_count,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out
        else:
            if not self.has_con:
                return matmat_node_basis_full_parallel_(
                    self.Ks,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    Cp,
                )
            else:
                out = matmat_node_basis_full_parallel_(
                    self.Ks,
                    self.elements_flat,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    self.elements_size,
                    mat,
                    Cp,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out

    def _rmatmat(self, rho, mat, Cp=None):
        """
        Transpose matrix-matrix product K(rho)^T @ mat.
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
        mat : cupy.sparse.csr_matrix or cupy.ndarray
            Input matrix, shape (n_dof, n_cols)
        Cp : cupy.sparse.csr_matrix, optional
            Pre-allocated output sparsity pattern
            
        Returns
        -------
        cupy.sparse.csr_matrix
            Output matrix K(rho)^T @ mat, shape (n_dof, n_cols)
            
        Notes
        -----
        For symmetric stiffness matrices, same as _matmat(). Delegates to _matmat().
        """
        return self._matmat(rho, mat, Cp)

    def dot(self, rhs):
        """
        Matrix-vector or matrix-matrix product K @ rhs.
        
        Parameters
        ----------
        rhs : cupy.ndarray or cupy.sparse.csr_matrix
            Right-hand side vector or matrix
            - Vector: shape (n_dof,)
            - Matrix: shape (n_dof, n_cols) or sparse CSR matrix
            
        Returns
        -------
        cupy.ndarray or cupy.sparse.csr_matrix
            Result K @ rhs, same type and shape as rhs (except columns)
            
        Raises
        ------
        ValueError
            If rho not set or rhs shape mismatch
        NotImplementedError
            If rhs is not cupy array or sparse matrix
            
        Notes
        -----
        - Uses cached self.rho if available
        - Automatically detects vector vs matrix input
        - Converts sparse matrices to CSR format if needed
        - Main interface for matrix operations
        """
        if self.has_rho:
            if isinstance(rhs, cp.ndarray):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError(
                        "Shape of the input vector does not match the number of nodes and dof."
                    )
            elif cp.sparse.issparse(rhs):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    if isinstance(rhs, cp.sparse.csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError(
                        "Shape of the input matrix does not match the number of nodes and dof."
                    )
            else:
                raise NotImplementedError(
                    "Only cupy arrays and sparse matrices are supported."
                )
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")


class GeneralStiffnessKernel(StiffnessKernel):
    """
    CUDA-accelerated stiffness kernel for heterogeneous meshes.
    
    GPU version of CPU GeneralStiffnessKernel. Most general kernel supporting meshes with
    elements of varying sizes and topologies. Uses flat storage with pointer arrays for
    memory-efficient representation. Identical API to CPU version but operates entirely on GPU.
    
    Parameters
    ----------
    mesh : CuGeneralMesh
        CUDA general mesh with potentially heterogeneous elements (mesh.is_uniform can be False)
        
    Attributes
    ----------
    K_flat : cupy.ndarray
        Flattened element stiffness matrices (on GPU)
    K_ptr : cupy.ndarray
        Pointer array for accessing K_flat: K_e = K_flat[K_ptr[e]:K_ptr[e+1]] (on GPU)
    elements_flat : cupy.ndarray
        Flattened element connectivity (on GPU)
    elements_ptr : cupy.ndarray
        Pointer array for accessing elements (on GPU)
    element_sizes : cupy.ndarray
        Number of nodes per element, shape (n_elements,) (on GPU)
    shape : tuple
        Global stiffness matrix dimensions (n_dof, n_dof)
    constraints : cupy.ndarray (bool)
        Boolean array for constrained DOFs (on GPU)
    has_cons : bool
        True if boundary conditions have been applied
        
    Methods
    -------
    Same interface as StructuredStiffnessKernel (set_rho, dot, construct, diagonal, etc.)
        
    Notes
    -----
    - Supports mixed element types (triangles + quads, tets + hexes, etc.)
    - Flat storage with pointers enables variable element sizes
    - Slower than Uniform/Structured kernels due to indirection
    - Automatically used if mesh.is_uniform == False
    - All arrays stored as CuPy arrays on GPU
    - All operations on GPU using CuPy
    - Use with CUDA solvers (CG, MultiGrid) for best performance
    - Memory limited by GPU VRAM
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import GeneralMesh, GeneralStiffnessKernel
    >>> import numpy as np
    >>> # Mixed triangular and quad mesh
    >>> nodes = np.array([[0,0], [1,0], [0,1], [1,1], [2,0]])
    >>> elements = [np.array([0,1,2]), np.array([1,4,3,2])]  # Triangle + Quad
    >>> mesh = GeneralMesh(nodes, elements)
    >>> kernel = GeneralStiffnessKernel(mesh=mesh)
    """
    def __init__(self, mesh: CuGeneralMesh):
        super().__init__()

        if mesh.is_uniform:
            raise ValueError(
                "Mesh is uniform, you should use UniformStiffnessKernel instead."
            )
        self.mesh = mesh
        self.nodes, self.elements = mesh.nodes, mesh.elements
        self.dof = mesh.dof
        
        self.dtype = mesh.dtype

        # Initialize lists for flattening
        elements_flat = cp.array(mesh.elements_flat, dtype=cp.int32, copy=False)
        element_sizes = cp.array(mesh.element_sizes, dtype=cp.int32, copy=False)
        K_flat = mesh.K_flat

        # Move everything to GPU with appropriate types
        self.nodes = cp.array(self.nodes, dtype=self.dtype, copy=False)
        self.elements_flat = cp.array(elements_flat, dtype=cp.int32, copy=False)
        self.K_flat = cp.array(K_flat, dtype=self.dtype, copy=False)

        # Calculate pointers
        self.K_ptr = cp.array(mesh.K_ptr, dtype=cp.int32, copy=False)
        self.elements_ptr = cp.array(mesh.elements_ptr, dtype=cp.int32, copy=False)

        self.el_ids = cp.arange(self.elements.shape[0], dtype=cp.int32).repeat(
            element_sizes.get().tolist()
        )
        self.sorter = cp.argsort(self.elements_flat).astype(cp.int32)
        self.node_ids = cp.searchsorted(
            self.elements_flat,
            cp.arange(self.nodes.shape[0], dtype=cp.int32),
            sorter=self.sorter,
            side="left",
        ).astype(cp.int32)
        self.n_nodes = self.nodes.shape[0]

        self.has_been_constructed = False
        self.ptr = None
        self.indices = None
        self.data = None

        self.max_con_count = (
            cp.diff(self.elements_ptr).max() * int(self.dof)
        ) * cp.unique(self.elements_flat, return_counts=True)[1].max()

        self.constraints = cp.zeros(self.n_nodes * self.dof, dtype=cp.bool_)

        self.shape = (self.n_nodes * self.dof, self.n_nodes * self.dof)

        self.has_con = False
        self.idx_map = cp.arange(self.n_nodes * self.dof, dtype=cp.int32)

    def diagonal(self, rho=None):
        """
        Compute diagonal entries of stiffness matrix.
        
        Parameters
        ----------
        rho : cupy.ndarray, optional
            Element density variables, shape (n_elements,). If None, uses cached self.rho
            
        Returns
        -------
        cupy.ndarray
            Diagonal entries of K(rho), shape (n_dof,)
            
        Notes
        -----
        - Uses GPU-accelerated flat storage assembly
        - Respects Dirichlet boundary conditions (constrained DOFs zeroed)
        - Caches result in self.diag for reuse
        - Used for preconditioning and diagnostics
        """
        if rho is None and not self.has_rho:
            raise ValueError(
                "Rho has not been set. diagonal works only after setting rho or if rho is provided."
            )
        elif rho is None:
            self.diag =  get_diagonal_node_basis_flat(
                self.K_flat,
                self.elements_flat,
                self.K_ptr,
                self.elements_ptr,
                self.el_ids,
                self.rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.constraints,
                self.diag
            )
            return self.diag
        else:
            self.diag = get_diagonal_node_basis_flat(
                self.K_flat,
                self.elements_flat,
                self.K_ptr,
                self.elements_ptr,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                self.dof,
                self.constraints,
                self.diag
            )
            return self.diag

    def set_constraints(self, constraints):
        """
        Set Dirichlet boundary condition DOF (replaces existing constraints).
        
        Parameters
        ----------
        constraints : cp.ndarray or array-like
            Indices of constrained DOF
            
        Notes
        -----
        - Resets all previous constraints
        - Use add_constraints() to append to existing constraints
        - Constrained rows/columns are zeroed in matrix operations
        """
        self.constraints[:] = False
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def add_constraints(self, constraints):
        """
        Add Dirichlet boundary conditions (accumulates with existing constraints).
        
        Parameters
        ----------
        constraints : cp.ndarray
            DOF indices to constrain on GPU, shape (n_constrained_dofs,)
            
        Notes
        -----
        Adds to existing constraints. Use set_constraints() to replace all constraints.
        - Use set_constraints() to replace all constraints
        """
        self.constraints[constraints] = True
        self.has_con = True
        self.non_con_map = self.idx_map[~self.constraints]

    def process_grad(self, U):
        """
        Compute element-wise compliance sensitivities dC/drho.
        
        Parameters
        ----------
        U : cp.ndarray
            Displacement vector from FEA solve
            
        Returns
        -------
        dk : cp.ndarray
            Element sensitivities (compliance derivative per element)
            
        Notes
        -----
        - Computes: dk[e] = -U[e]^T @ K_e @ U[e]
        - Used for compliance minimization gradient
        - Negative sign convention for minimization
        """
        self.dk = process_dk(
            self.K_single, self.elements_flat, U, self.dof, self.elements_size, self.dk
        )
        return self.dk

    def construct(self, rho):
        """
        Construct CSR stiffness matrix representation.
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
            
        Returns
        -------
        cupy.sparse.csr_matrix
            Sparse CSR stiffness matrix K(rho), shape (n_dof, n_dof)
            
        Notes
        -----
        - Assembles full sparse matrix structure (first call) or updates values (subsequent)
        - Uses matrix-matrix product with identity to extract sparsity pattern
        - Caches sparsity pattern (ptr) for efficient updates
        - Sets has_been_constructed flag after first call
        - More memory-intensive than matrix-free operations but needed for some solvers
        """
        size = self.n_nodes * self.dof
        if not self.has_been_constructed:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype)
            )
            self.ptr = self.CSR.indptr
        else:
            self.CSR = self._matmat(
                rho, cp.sparse.eye(size, format="csr", dtype=self.dtype), self.ptr
            )
        self.has_been_constructed = True

        return self.CSR

    def _matvec(self, rho, vec):
        """
        Matrix-vector product K(rho) @ vec using GPU-accelerated flat storage.
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
        vec : cupy.ndarray
            Input vector, shape (n_dof,)
            
        Returns
        -------
        cupy.ndarray
            Output vector K(rho) @ vec, shape (n_dof,)
            
        Notes
        -----
        - Uses parallel GPU kernel for element-wise assembly
        - Handles Dirichlet constraints (zeroed rows/columns)
        - Caches result in self.mat_vec for reuse
        - Optimized for flat storage with pointer arrays
        """
        if not self.has_con:
            self.mat_vec = mat_vec_node_basis_parallel_flat(
                self.K_flat,
                self.elements_flat,
                self.K_ptr,
                self.elements_ptr,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                out=self.mat_vec
            )
            return self.mat_vec
        else:
            self.mat_vec = mat_vec_node_basis_parallel_flat(
                self.K_flat,
                self.elements_flat,
                self.K_ptr,
                self.elements_ptr,
                self.el_ids,
                rho,
                self.sorter,
                self.node_ids,
                self.n_nodes,
                vec,
                self.dof,
                self.constraints,
                self.mat_vec
            )
            return self.mat_vec

    def _rmatvec(self, rho, vec):
        """
        Transpose matrix-vector product K(rho)^T @ vec.
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
        vec : cupy.ndarray
            Input vector, shape (n_dof,)
            
        Returns
        -------
        cupy.ndarray
            Output vector K(rho)^T @ vec, shape (n_dof,)
            
        Notes
        -----
        For symmetric stiffness matrices, same as _matvec(). Delegates to _matvec().
        """
        return self._matvec(rho, vec)

    def _matmat(self, rho, mat, Cp=None):
        """
        Matrix-matrix product K(rho) @ mat using GPU-accelerated assembly.
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
        mat : cupy.sparse.csr_matrix or cupy.ndarray
            Input matrix, shape (n_dof, n_cols)
        Cp : cupy.sparse.csr_matrix, optional
            Pre-allocated output sparsity pattern. If None, computes new pattern
            
        Returns
        -------
        cupy.sparse.csr_matrix
            Output matrix K(rho) @ mat, shape (n_dof, n_cols)
            
        Notes
        -----
        - Uses parallel GPU kernel for batch element assembly
        - Handles Dirichlet constraints (zeroed rows/columns)
        - If Cp provided, reuses sparsity pattern for efficiency
        - Used by construct() to build full stiffness matrix
        - Negative indices set to 0 (constraint handling artifact)
        """
        if Cp is None:
            if not self.has_con:
                return matmat_node_basis_flat_parallel(
                    self.K_flat,
                    self.elements_flat,
                    self.K_ptr,
                    self.elements_ptr,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    mat,
                    self.max_con_count,
                )
            else:
                out = matmat_node_basis_flat_parallel(
                    self.K_flat,
                    self.elements_flat,
                    self.K_ptr,
                    self.elements_ptr,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    mat,
                    self.max_con_count,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out
        else:
            if not self.has_con:
                return matmat_node_basis_flat_parallel_(
                    self.K_flat,
                    self.elements_flat,
                    self.K_ptr,
                    self.elements_ptr,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    mat,
                    Cp,
                )
            else:
                out = matmat_node_basis_flat_parallel(
                    self.K_flat,
                    self.elements_flat,
                    self.K_ptr,
                    self.elements_ptr,
                    self.el_ids,
                    rho,
                    self.sorter,
                    self.node_ids,
                    self.n_nodes,
                    self.dof,
                    mat,
                    self.max_con_count,
                    self.constraints,
                )
                out.indices[out.indices < 0] = 0
                return out

    def _rmatmat(self, rho, mat, Cp=None):
        """
        Transpose matrix-matrix product K(rho)^T @ mat.
        
        Parameters
        ----------
        rho : cupy.ndarray
            Element density variables, shape (n_elements,)
        mat : cupy.sparse.csr_matrix or cupy.ndarray
            Input matrix, shape (n_dof, n_cols)
        Cp : cupy.sparse.csr_matrix, optional
            Pre-allocated output sparsity pattern
            
        Returns
        -------
        cupy.sparse.csr_matrix
            Output matrix K(rho)^T @ mat, shape (n_dof, n_cols)
            
        Notes
        -----
        For symmetric stiffness matrices, same as _matmat(). Delegates to _matmat().
        """
        return self._matmat(rho, mat, Cp)

    def dot(self, rhs):
        """
        Matrix-vector or matrix-matrix product K @ rhs.
        
        Parameters
        ----------
        rhs : cupy.ndarray or cupy.sparse.csr_matrix
            Right-hand side vector or matrix
            - Vector: shape (n_dof,)
            - Matrix: shape (n_dof, n_cols) or sparse CSR matrix
            
        Returns
        -------
        cupy.ndarray or cupy.sparse.csr_matrix
            Result K @ rhs, same type and shape as rhs (except columns)
            
        Raises
        ------
        ValueError
            If rho not set or rhs shape mismatch
        NotImplementedError
            If rhs is not cupy array or sparse matrix
            
        Notes
        -----
        - Uses cached self.rho if available
        - Automatically detects vector vs matrix input
        - Converts sparse matrices to CSR format if needed
        - Main interface for matrix operations
        """
        if self.has_rho:
            if isinstance(rhs, cp.ndarray):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    return self._matvec(self.rho, rhs)
                else:
                    raise ValueError(
                        "Shape of the input vector does not match the number of nodes and dof."
                    )
            elif cp.sparse.issparse(rhs):
                if rhs.shape[0] == self.n_nodes * self.dof:
                    if isinstance(rhs, cp.sparse.csr_matrix):
                        return self._matmat(self.rho, rhs)
                    else:
                        return self._matmat(self.rho, rhs.tocsr())
                else:
                    raise ValueError(
                        "Shape of the input matrix does not match the number of nodes and dof."
                    )
            else:
                raise NotImplementedError(
                    "Only cupy arrays and sparse matrices are supported."
                )
        else:
            raise ValueError("Rho has not been set. dot works only after setting rho.")
