import cupy as cp
from ..commons import Solver
from sksparse.cholmod import cholesky
from ...stiffness.CUDA._FEA import StiffnessKernel as CuStiffnessKernel
from ...stiffness.CUDA._FEA import GeneralStiffnessKernel as CuGeneralKernel
from ...stiffness.CUDA._FEA import UniformStiffnessKernel as CuUniformKernel
from cupyx.scipy.sparse.linalg import cg, gmres, spsolve, splu
import time
import logging
from ...core.CUDA._mgm import (
    apply_restriction_cuda,
    apply_prolongation_cuda,
    get_restricted_l0_cuda,
    get_restricted_l1p_cuda
)
from ...geom.CUDA._mesh import CuStructuredMesh2D as StructuredMesh2D
from ...geom.CUDA._mesh import CuStructuredMesh3D as StructuredMesh3D
from typing import Union
loger = logging.getLogger(__name__)


class CG(Solver):
    """
    CUDA-accelerated Conjugate Gradient iterative solver.
    
    GPU version of the CG solver using CuPy. Efficiently solves large sparse symmetric
    positive definite linear systems Kx=b on GPU with optional matrix-free operation.
    
    Parameters
    ----------
    kernel : CuStiffnessKernel
        CUDA stiffness kernel (matrix operator)
    maxiter : int, optional
        Maximum number of CG iterations (default: 1000)
    tol : float, optional
        Convergence tolerance for relative residual (default: 1e-5)
    matrix_free : bool, optional
        Whether to use matrix-free mode. If None, automatically determined:
        - True for StructuredStiffnessKernel (recommended)
        - False for General/UniformStiffnessKernel (faster)
        
    Notes
    -----
    - Matrix-free mode: Lower memory, slower per-iteration
    - Assembled matrix mode: Higher memory, faster per-iteration
    - For structured meshes, matrix-free is preferred
    - For general meshes, assembled matrix is faster
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import StructuredMesh2D, StructuredStiffnessKernel, CG
    >>> mesh = StructuredMesh2D(nx=256, ny=256, lx=1.0, ly=1.0)
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> solver = CG(kernel=kernel, maxiter=1000, tol=1e-6)
    """
    def __init__(self, kernel: CuStiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, CuGeneralKernel) or isinstance(kernel, CuUniformKernel):
            if matrix_free is None:
                matrix_free = False
                loger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                loger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
        elif matrix_free is None:
            matrix_free = True
        
        self.matrix_free = matrix_free
        
        if not self.matrix_free:
            self.K = None
    
    def solve(self, rhs, rho=None, use_last=True):
        """
        Solve linear system K(rho) @ U = rhs on GPU.
        
        Parameters
        ----------
        rhs : cp.ndarray
            Right-hand side force vector on GPU, shape (n_dof,)
        rho : cp.ndarray, optional
            Design variables (densities) on GPU, shape (n_elements,)
        use_last : bool, optional
            Use previous solution as initial guess (warm-starting), default: True
            
        Returns
        -------
        U : cp.ndarray
            Solution vector on GPU, shape (n_dof,)
        residual : float
            Relative residual: ||K@U - rhs|| / ||rhs||
            
        Raises
        ------
        ValueError
            If rho is not provided and kernel.has_rho is False
            
        Notes
        -----
        - All operations performed on GPU using CuPy
        - Warm-starting (use_last=True) accelerates convergence
        - Matrix-free mode uses kernel.dot() directly
        - Assembled mode constructs explicit CSR matrix first
        """
        if self.kernel.has_rho or (not rho is None):
            if rho is not None:
                self.kernel.set_rho(rho)
            if use_last:
                if self.matrix_free:
                    out = cg(self.kernel, rhs, x0=self.last_x0, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = cg(self.K, rhs, x0=self.last_x0, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.K@out)/cp.linalg.norm(rhs)
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = cg(self.kernel, rhs, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = cg(self.K, rhs, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.K@out)/cp.linalg.norm(rhs)
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        return out, residual

class GMRES(Solver):
    """
    CUDA-accelerated Generalized Minimal Residual iterative solver.
    
    GPU version of GMRES solver using CuPy. Solves non-symmetric linear systems
    Kx=b on GPU. More general than CG but typically slower and more memory-intensive.
    
    Parameters
    ----------
    kernel : CuStiffnessKernel
        CUDA stiffness kernel (matrix operator)
    maxiter : int, optional
        Maximum number of GMRES iterations (default: 1000)
    tol : float, optional
        Convergence tolerance for relative residual (default: 1e-5)
    matrix_free : bool, optional
        Whether to use matrix-free mode. If None, automatically determined:
        - True for StructuredStiffnessKernel
        - False for General/UniformStiffnessKernel (recommended)
        
    Notes
    -----
    - GMRES handles non-symmetric systems (rare in FEA)
    - CG is preferred for symmetric positive definite systems
    - Higher memory usage than CG due to Krylov subspace storage
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import StructuredMesh2D, StructuredStiffnessKernel, GMRES
    >>> solver = GMRES(kernel=kernel, maxiter=1000, tol=1e-6)
    """
    def __init__(self, kernel: CuStiffnessKernel, maxiter=1000, tol=1e-5, matrix_free=None):
        super().__init__()
        self.kernel = kernel
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        
        if isinstance(kernel, CuGeneralKernel) or isinstance(kernel, CuUniformKernel):
            if matrix_free is None:
                matrix_free = False
                loger.warning("Matrix free is set to False for general and uniform kernels. This is recommended for speed. Will use more memory.")
            elif matrix_free:
                loger.warning("Matrix free is set to True for general and uniform kernels. This is not recommended for speed. Will use less memory however.")
        elif matrix_free is None:
            matrix_free = True
        
        self.matrix_free = matrix_free
        
        if not self.matrix_free:
            self.K = None
    
    def solve(self, rhs, rho=None, use_last=True):
        """
        Solve linear system K(rho) @ U = rhs on GPU using GMRES.
        
        Parameters
        ----------
        rhs : cp.ndarray
            Right-hand side force vector on GPU, shape (n_dof,)
        rho : cp.ndarray, optional
            Design variables (densities) on GPU, shape (n_elements,)
        use_last : bool, optional
            Use previous solution as initial guess (warm-starting), default: True
            
        Returns
        -------
        U : cp.ndarray
            Solution vector on GPU, shape (n_dof,)
        residual : float
            Relative residual: ||K@U - rhs|| / ||rhs||
            
        Raises
        ------
        ValueError
            If rho is not provided and kernel.has_rho is False
            
        Notes
        -----
        - All operations performed on GPU using CuPy
        - Warm-starting (use_last=True) accelerates convergence
        - Matrix-free mode uses kernel.dot() directly
        - Assembled mode constructs explicit CSR matrix first
        - GMRES handles non-symmetric systems (rare in FEA)
        - CG is preferred for symmetric positive definite systems
        """
        if self.kernel.has_rho or (not rho is None):
            if rho is not None:
                self.kernel.set_rho(rho)
            if use_last:
                if self.matrix_free:
                    out = gmres(self.kernel, rhs, x0=self.last_x0, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = gmres(self.K, rhs, x0=self.last_x0, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.K@out)/cp.linalg.norm(rhs)
                self.last_x0 = out
            else:
                if self.matrix_free:
                    out = gmres(self.kernel, rhs, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
                else:
                    self.K = self.kernel.construct(self.kernel.rho)
                    out = gmres(self.K, rhs, tol = self.tol, maxiter=self.maxiter)[0]
                    residual = cp.linalg.norm(rhs - self.K@out)/cp.linalg.norm(rhs)
        else:
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        return out, residual
    
class SPSOLVE(Solver):
    """
    CUDA sparse direct solver using CuPy's spsolve.
    
    GPU-accelerated direct sparse LU factorization solver. Fast for small to medium problems
    but memory-intensive. Limited to < 3M DOF due to GPU memory constraints.
    
    Parameters
    ----------
    kernel : CuStiffnessKernel
        CUDA stiffness kernel (must construct explicit matrix)
        
    Raises
    ------
    ValueError
        If kernel.shape[0] > 3e6 (3 million DOF limit)
        
    Notes
    -----
    - Direct solver: No iterations, exact solution (within numerical precision)
    - High memory usage: Stores factorization on GPU
    - Use iterative solvers (CG, MultiGrid) for large problems
    - Automatically handles constrained DOF
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import StructuredMesh2D, StructuredStiffnessKernel, SPSOLVE
    >>> mesh = StructuredMesh2D(nx=64, ny=64, lx=1.0, ly=1.0)
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> solver = SPSOLVE(kernel=kernel)  # OK for small problems
    """
    def __init__(self, kernel: CuStiffnessKernel):
        super().__init__()
        
        if kernel.shape[0] > 3e6:
            raise ValueError("Currently we do not allow spsolve for problem size bigger than 3M degrees of freedom. You can override this by passing a dummy kernel and overriding the kernel attribute.")
        
        self.kernel = kernel
    
    def solve(self, rhs, rho=None, **kwargs):
        """
        Solve the linear system Kx = rhs using sparse direct solver.
        
        Parameters
        ----------
        rhs : cp.ndarray
            Right-hand side vector
        rho : cp.ndarray, optional
            Design variable densities. If None, uses kernel's stored rho
        **kwargs
            Additional keyword arguments (unused, for API compatibility)
            
        Returns
        -------
        U : cp.ndarray
            Solution vector
        residual : float
            Relative residual norm ||rhs - K@U|| / ||rhs||
            
        Notes
        -----
        - Direct solver: No warm start needed
        - Constructs and factorizes K every call (expensive)
        - Consider iterative solvers for repeated solves in optimization
        """
        if not rho is None:
            K = self.kernel.construct(rho)
        else:
            if not self.kernel.has_rho:
                raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
            K = self.kernel.construct(self.kernel.rho)
        
        out = cp.copy(rhs)
        
        if self.kernel.has_con:
            K = K[:,self.kernel.non_con_map][self.kernel.non_con_map,:]
            rhs_ = rhs[self.kernel.non_con_map]
        
        
        out[self.kernel.non_con_map] = spsolve(K, rhs_)
        
        residual = cp.linalg.norm(rhs - self.kernel@out)/cp.linalg.norm(rhs)
        
        return out, residual
    
class MultiGrid(Solver):
    """
    CUDA-accelerated Geometric Multigrid solver for structured meshes.
    
    GPU implementation of geometric multigrid with Jacobi smoothing and coarse grid correction.
    Most memory-efficient solver for large 3D problems on GPU. Uses restriction/prolongation
    operators on nested grids for O(n) complexity per iteration.
    
    Parameters
    ----------
    mesh : StructuredMesh2D or StructuredMesh3D
        CUDA structured mesh (required for geometric coarsening)
    kernel : CuStiffnessKernel
        CUDA stiffness kernel (must be StructuredStiffnessKernel)
    maxiter : int, optional
        Maximum outer iterations (preconditioned CG) (default: 1000)
    tol : float, optional
        Convergence tolerance for relative residual (default: 1e-5)
    n_smooth : int, optional
        Number of Jacobi smoothing iterations per level (default: 3)
    omega : float, optional
        Jacobi relaxation parameter (default: 0.5)
    n_level : int, optional
        Number of multigrid levels (default: 3)
    cycle : str, optional
        Multigrid cycle type: 'V' or 'W' (default: 'W')
    w_level : int or list, optional
        Levels to apply W-cycle at (default: 1)
    coarse_solver : str, optional
        Coarse grid solver: 'cholmod', 'cg', 'gmres', 'splu', 'spsolve' (default: 'cholmod')
    matrix_free : bool, optional
        Whether to use matrix-free operators (default: False, not yet implemented)
    low_level_tol : float, optional
        Tolerance for coarse grid iterative solver (default: 1e-8)
    low_level_maxiter : int, optional
        Max iterations for coarse grid iterative solver (default: 5000)
    min_omega : float, optional
        Minimum omega for adaptive relaxation (default: 0.4)
    omega_boost : float, optional
        Omega boost factor per level (default: 1.06)
        
    Notes
    -----
    - Only works with structured meshes (uniform grid coarsening)
    - Memory efficient: No explicit matrix storage
    - Preconditioned CG with multigrid preconditioner
    - Adaptive omega based on residual
    - CHOLMOD coarse solver requires CPU transfer (fast for small coarse grids)
    
    Raises
    ------
    ValueError
        If matrix_free=True with splu/spsolve coarse solver
        If coarse_solver not in ['cg', 'gmres', 'splu', 'spsolve', 'cholmod']
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import StructuredMesh3D, StructuredStiffnessKernel, MultiGrid
    >>> mesh = StructuredMesh3D(nx=128, ny=128, nz=128, lx=1.0, ly=1.0, lz=1.0)
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> solver = MultiGrid(mesh=mesh, kernel=kernel, n_level=4, cycle='W')
    """
    def __init__(self, mesh: Union[StructuredMesh2D,StructuredMesh3D],
                 kernel: CuStiffnessKernel, maxiter=1000, tol=1e-5, n_smooth=3,
                 omega=0.5 , n_level = 3, cycle='W', w_level=1, coarse_solver='cholmod',
                 matrix_free=False, low_level_tol = 1e-8, low_level_maxiter=5000, min_omega=0.4, omega_boost=1.06):
        super().__init__()
        self.kernel = kernel
        self.mesh = mesh
        self.last_x0 = None
        self.tol = tol
        self.maxiter = maxiter
        self.n_smooth = n_smooth
        self.omega = omega
        self.max_omega = omega
        self.min_omega = max(omega/2, min_omega)
        self.d_omega = (self.max_omega - self.min_omega)
        self.omega_boost = omega_boost
        self.n_level = n_level
        self.cycle = cycle
        self.w_level = w_level
        self.matrix_free = matrix_free
        self.dof = self.mesh.dof
        self.coarse_solver = coarse_solver
        self.low_level_tol = low_level_tol
        self.low_level_maxiter = low_level_maxiter
        self.factor = None
        
        if isinstance(self.w_level, int):
            self.w_level = [self.w_level]
        
        if self.coarse_solver in ['splu', 'spsolve'] and self.matrix_free:
            raise ValueError("Matrix free is not supported with splu solver use cg instead.")
        
        if self.coarse_solver not in ['cg', 'gmres', 'splu', 'spsolve', 'cholmod']:
            raise ValueError("Coarse solver not recognized.")
        
        if not self.matrix_free:
            self.ptr = None
            self.PRs = []
        else:
            raise NotImplementedError("Matrix free is not implemented yet.")
            
    def reset(self):
        """Reset cached multigrid hierarchy (forces rebuild on next solve)."""
        self.ptr = None
    
    def _jacobi_smoother(self, x, b, A, D_inv, n_step):
        for _ in range(n_step):
            x += self.omega * D_inv * (b - A @ x)
        return x
    
    def _setup(self):
        self.levels = []
        
        D = 1/self.kernel.diagonal()
        self.levels.append((self.kernel, D))
        
        for i in range(self.n_level):
            if i == 0 and not self.ptr is None:
                op = get_restricted_l0_cuda(self.mesh, self.kernel, Cp = self.ptr[0])
                D = 1/op.diagonal()
                self.levels.append((op, D))
                
            elif i == 0:
                self.ptr = []
                if self.mesh.nel.shape[0] == 2:
                    nnz = cp.ones((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*self.dof, dtype=cp.int32) * 18
                    Cp = cp.zeros((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*self.dof + 1, dtype=cp.int32)
                    Cp[1:] = cp.cumsum(nnz)
                else:
                    nnz = cp.ones((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*(self.mesh.nel[2]//2+1)*self.dof, dtype=cp.int32) * 81
                    Cp = cp.zeros((self.mesh.nel[0]//2+1)*(self.mesh.nel[1]//2+1)*(self.mesh.nel[2]//2+1)*self.dof + 1, dtype=cp.int32)
                    Cp[1:] = cp.cumsum(nnz)
                
                self.ptr.append(cp.copy(Cp))
                op = get_restricted_l0_cuda(self.mesh, self.kernel, Cp = Cp)
                D = 1/op.diagonal()
                self.levels.append((op, D))
                
            elif len(self.ptr) <= i:
                
                if self.mesh.nel.shape[0] == 2:
                    nnz = cp.ones((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*self.dof, dtype=cp.int32) * 18
                    Cp = cp.zeros((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*self.dof + 1, dtype=cp.int32)
                    Cp[1:] = cp.cumsum(nnz)
                else:
                    nnz = cp.ones((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*(self.mesh.nel[2]//(2**(i+1))+1)*self.dof, dtype=cp.int32) * 81
                    Cp = cp.zeros((self.mesh.nel[0]//(2**(i+1))+1)*(self.mesh.nel[1]//(2**(i+1))+1)*(self.mesh.nel[2]//(2**(i+1))+1)*self.dof + 1, dtype=cp.int32)
                    Cp[1:] = cp.cumsum(nnz)
                
                self.ptr.append(cp.copy(Cp))
                op = get_restricted_l1p_cuda(self.levels[-1][0], self.mesh.nel//(2**i), self.dof, Cp = Cp)
                D = 1/op.diagonal()
                
                # if i == self.n_level-1:
                #    op.sum_duplicates()
                
                self.levels.append((op, D))
            else:
                op = get_restricted_l1p_cuda(self.levels[-1][0], self.mesh.nel//(2**i), self.dof, Cp = self.ptr[i])
                D = 1/op.diagonal()
                # if i == self.n_level-1:
                #     op.sum_duplicates()
                self.levels.append((op, D))
                
            cp.get_default_memory_pool().free_all_blocks()

    def _coarse_solver(self, K):
        if self.coarse_solver == 'cg':
            return lambda rhs: cg(K, rhs, tol = self.low_level_tol, maxiter=self.low_level_maxiter)[0]
        elif self.coarse_solver == 'gmres':
            lambda rhs: gmres(K, rhs, tol = self.low_level_tol, maxiter=self.low_level_maxiter)[0]
        elif self.coarse_solver == 'splu':
            SOLVER = splu(K)
            return lambda rhs: SOLVER.solve(rhs)
        elif self.coarse_solver == 'spsolve':
            return lambda rhs: spsolve(K, rhs)
        elif self.coarse_solver == 'cholmod':
            K = K.tocsc().get()
            if self.factor is None:
                self.factor = cholesky(K, beta=1e-6)
            else:
                self.factor.cholesky_inplace(K, beta=1e-6)
            return lambda rhs: cp.array(self.factor(rhs.get()))
        else:
            raise ValueError("Coarse solver not recognized.")
    
    def _multi_grid(self, x, b, level):
        if level == self.n_level:
            return self.coarse_solve(b)
        
        # presmooth
        A, D = self.levels[level]
        self.omega = self.omega * (1.06)**(level)
        x = self._jacobi_smoother(x, b, A, D, self.n_smooth)
        self.omega = self.omega / (1.06)**(level)
        # residual
        r = b - A@x
        
        nel = (self.mesh.nel // 2**level).astype(cp.int32)
        
        # restrict
        coarse_residual = apply_restriction_cuda(r,nel,self.dof)
        # coarse_residual = self.PRs[level][1] @ r
        
        # go to next level
        coarse_u = coarse_residual*0.0
        e = self._multi_grid(coarse_u, coarse_residual, level+1)
        
        # prolongate
        e = apply_prolongation_cuda(e,nel,self.dof)
        # e = self.PRs[level][0] @ e
        
        self.omega = self.omega * (1.06)**(level)
        e = self._jacobi_smoother(e, r, A, D, self.n_smooth)
        self.omega = self.omega / (1.06)**(level)
        
        x += e
        
        if self.cycle == 'w' and level in self.w_level:
            r = b - A@x
        
            # restrict
            coarse_residual = apply_restriction_cuda(r,nel,self.dof)
            # coarse_residual = self.PRs[level][1] @ r
            
            # go to next level
            coarse_u = coarse_residual*0.0
            e = self._multi_grid(coarse_u, coarse_residual, level+1)
            
            # prolongate
            e = apply_prolongation_cuda(e,nel,self.dof)
            # e = self.PRs[level][0] @ e
            
            self.omega = self.omega * (1.06)**(level)
            e = self._jacobi_smoother(e, r, A, D, self.n_smooth)
            self.omega = self.omega / (1.06)**(level)
            
            x += e
        
        return x
    
    def solve(self, rhs, rho=None, use_last=True):
        """
        Solve linear system K(rho) @ U = rhs on GPU using multigrid-preconditioned CG.
        
        Parameters
        ----------
        rhs : cp.ndarray
            Right-hand side force vector on GPU, shape (n_dof,)
        rho : cp.ndarray, optional
            Design variables (densities) on GPU, shape (n_elements,)
        use_last : bool, optional
            Use previous solution as initial guess (warm-starting), default: True
            
        Returns
        -------
        U : cp.ndarray
            Solution vector on GPU, shape (n_dof,)
        residual : float
            Relative residual: ||K@U - rhs|| / ||rhs||
            
        Raises
        ------
        ValueError
            If rho is not provided and kernel.has_rho is False
            
        Notes
        -----
        - All operations performed on GPU using CuPy
        - First call builds multigrid hierarchy (expensive, ~1-5 seconds)
        - Subsequent calls reuse hierarchy (fast, ~0.1-1 second)
        - Uses preconditioned CG with multigrid V/W-cycle preconditioner
        - Adaptive omega adjusts relaxation based on residual reduction
        - Frees GPU memory after solve to prevent accumulation
        - Most memory-efficient solver for large 3D problems (>10M DOF)
        """
        if not (self.kernel.has_rho or (not rho is None)):
            raise ValueError("Solver requires a density vector to be passed or set on the kernel.")
        
        if not rho is None:
            rho = self.kernel.set_rho(rho)
        
        if use_last and self.last_x0 is not None:
            x = self.last_x0
        else:
            x = cp.zeros_like(rhs)
            
        if self.matrix_free:
            self._mat_free_setup()
        else:
            self._setup()
        
        v_cycle = self._mat_free_multi_grid if self.matrix_free else self._multi_grid
        
        self.coarse_solve = self._coarse_solver(self.levels[-1][0])

        self.omega = self.min_omega
        r = rhs - self.kernel.dot(x)
        z = v_cycle(cp.zeros_like(r), r, 0)
        p = z.copy()
        rho_old = cp.dot(r, z)
        norm_b = cp.linalg.norm(rhs)
        
        for i in range(self.maxiter):
            q = self.kernel.dot(p)
            alpha = rho_old / cp.dot(p, q)
            x += alpha * p
            r -= alpha * q
            
            norm_r = cp.linalg.norm(r)
            R = norm_r / norm_b
            if R < self.tol:
                break
            self.omega = self.min_omega + self.d_omega/2*cp.exp((-cp.clip(R,self.tol,1e-1)+self.tol)*500)
            z = v_cycle(cp.zeros_like(r), r,0)
            rho_new = cp.dot(r, z)
            beta = rho_new / rho_old
            p = z + beta * p
            rho_old = rho_new
        
        residual = norm_r / norm_b
        
        self.last_x0 = x
        
        del self.levels, self.coarse_solve
        
        cp._default_memory_pool.free_all_blocks()
        
        return x, residual