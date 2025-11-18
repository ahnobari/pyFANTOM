from ...core.CPU._filter import (apply_filter_2D_parallel,
                                apply_filter_3D_parallel,
                                filter_kernel_2D_general,
                                filter_kernel_3D_general,
                                get_filter_weights_2D,
                                get_filter_weights_3D,
                                apply_filter_2D_parallel_transpose,
                                apply_filter_3D_parallel_transpose,
                                )
import numpy as np
from ..commons._filters import FilterKernel
from ._mesh import StructuredMesh2D, StructuredMesh3D, GeneralMesh

class StructuredFilter3D(FilterKernel):
    """
    Convolution-based density filter for 3D structured meshes.
    
    Implements smoothing filter for topology optimization to ensure manufacturability and
    mesh-independent solutions. Uses efficient convolution with precomputed weights and offsets
    for structured grids.
    
    Parameters
    ----------
    mesh : StructuredMesh3D
        3D structured mesh
    r_min : float
        Filter radius in element units (e.g., r_min=1.5 includes elements within 1.5 element widths)
        
    Attributes
    ----------
    shape : tuple
        Filter matrix dimensions (n_elements, n_elements)
    weights : ndarray
        Precomputed filter weights for neighbor elements
    offsets : ndarray
        Element index offsets for neighbors within filter radius
    normalizer : ndarray
        Per-element normalization factors for adjoint operation
        
    Methods
    -------
    dot(rho)
        Apply forward filter: filtered_rho = H @ rho
    _rmatvec(sens)
        Apply adjoint filter for sensitivity backpropagation: H^T @ sens
        
    Notes
    -----
    - Filter uses cone-shaped kernel: w(d) = (r_min - d) / r_min for d < r_min
    - Automatically handles anisotropic elements (dx != dy != dz)
    - Adjoint includes proper normalization for optimization gradients
    - Memory-efficient: stores only unique weights and offsets, not full matrix
    
    Examples
    --------
    >>> from pyFANTOM.CPU import StructuredMesh3D, StructuredFilter3D
    >>> mesh = StructuredMesh3D(nx=32, ny=32, nz=16, lx=1.0, ly=1.0, lz=0.5)
    >>> filter = StructuredFilter3D(mesh=mesh, r_min=1.5)
    >>> rho_filtered = filter.dot(rho_raw)
    """
    def __init__(self, mesh: StructuredMesh3D, r_min):
        super().__init__()
        self.dtype = mesh.dtype

        self.nelx = mesh.nelx
        self.nely = mesh.nely
        self.nelz = mesh.nelz
        self.r_min = r_min
        self.shape = (self.nelx * self.nely * self.nelz, self.nelx * self.nely * self.nelz)
        dx, dy, dz = mesh.dx, mesh.dy, mesh.dz
        self.scales = np.array([dx, dy, dz], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = np.arange(-n_neighbours, n_neighbours + 1, dtype=np.int32)
        
        a, b, c = np.meshgrid(offset_range, offset_range, offset_range, indexing='ij')
        offsets = np.vstack([a.ravel(), b.ravel(), c.ravel()]).T.astype(np.int32)
        offsets_adjusted = offsets * self.scales[None]
        
        distances = np.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        weights = weights.astype(self.dtype)
        
        self.weights = weights
        self.offsets = offsets
        
        self.normalizer = get_filter_weights_3D(self.nelx, self.nely, self.nelz, self.offsets, self.weights)
        
    def _matvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_3D_parallel(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights)
        return v_out
    
    def _rmatvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_3D_parallel_transpose(rho, v_out, self.nelx, self.nely, self.nelz, self.offsets, self.weights, self.normalizer)
        return v_out


class StructuredFilter2D(FilterKernel):
    """
    Convolution-based density filter for 2D structured meshes.
    
    Implements smoothing filter for 2D topology optimization using efficient convolution.
    Ensures minimum feature sizes and mesh-independent solutions.
    
    Parameters
    ----------
    mesh : StructuredMesh2D
        2D structured mesh
    r_min : float
        Filter radius in element units
        
    Attributes
    ----------
    shape : tuple
        Filter matrix dimensions (n_elements, n_elements)
    weights : ndarray
        Precomputed filter weights
    offsets : ndarray
        Element index offsets for neighbors
    normalizer : ndarray
        Per-element normalization for adjoint
        
    Methods
    -------
    dot(rho)
        Forward filter application
    _rmatvec(sens)
        Adjoint filter for sensitivity backpropagation
        
    Notes
    -----
    - Uses cone filter: w(d) = (r_min - d) / r_min
    - Handles non-square elements (dx != dy) automatically
    - Typical r_min values: 1.5-3.0 element units
    - Larger r_min produces smoother, coarser designs
    
    Examples
    --------
    >>> from pyFANTOM.CPU import StructuredMesh2D, StructuredFilter2D
    >>> mesh = StructuredMesh2D(nx=128, ny=64, lx=2.0, ly=1.0)
    >>> filter = StructuredFilter2D(mesh=mesh, r_min=2.0)
    >>> rho_smooth = filter.dot(rho)
    """
    def __init__(self, mesh: StructuredMesh2D, r_min):
        super().__init__()
        self.nelx = mesh.nel[0]
        self.nely = mesh.nel[1]
        self.r_min = r_min
        self.shape = (self.nelx * self.nely, self.nelx * self.nely)
        self.dtype = mesh.dtype
        
        dx, dy = mesh.dx, mesh.dy
        self.scales = np.array([dx, dy], dtype=self.dtype)
        self.scales = self.scales / self.scales.min()
        
        filter_rad = r_min
        n_neighbours = int(np.ceil(filter_rad))
        offset_range = np.arange(-n_neighbours, n_neighbours + 1, dtype=np.int32)
        
        a, b = np.meshgrid(offset_range, offset_range, indexing='ij')
        offsets = np.vstack([a.ravel(), b.ravel()]).T.astype(np.int32)
        offsets_adjusted = offsets * self.scales[None]
        
        distances = np.linalg.norm(offsets_adjusted, axis=1)
        weights = (r_min - distances) / r_min
        valid_mask = weights > 0
        offsets = offsets[valid_mask]
        weights = weights[valid_mask]
        weights /= weights.sum()
        weights = weights.astype(self.dtype)
        
        self.weights = weights
        self.offsets = offsets
        
        self.normalizer = get_filter_weights_2D(self.nelx, self.nely, self.offsets, self.weights)
        
        
    def _matvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_2D_parallel(rho, v_out, self.nelx, self.nely, self.offsets, self.weights)
        return v_out
    
    def _rmatvec(self, rho):
        v_out = np.zeros_like(rho)
        apply_filter_2D_parallel_transpose(rho, v_out, self.nelx, self.nely, self.offsets, self.weights, self.normalizer)
        return v_out

class GeneralFilter(FilterKernel):
    """
    KDTree-based density filter for unstructured meshes.
    
    Implements density filtering for general unstructured meshes using spatial search
    to identify neighbors. Stores explicit filter matrix (sparse) unlike structured filters.
    
    Parameters
    ----------
    mesh : GeneralMesh
        General unstructured mesh
    r_min : float
        Filter radius in physical units (not element units)
        
    Attributes
    ----------
    kernel : ndarray
        Explicit filter matrix, shape (n_elements, n_elements)
    shape : tuple
        Filter matrix dimensions
        
    Methods
    -------
    dot(rho)
        Forward filter: H @ rho
    _rmatvec(sens)
        Adjoint filter: H^T @ sens
        
    Notes
    -----
    - Uses spatial search (KDTree internally) to build filter
    - Stores explicit sparse matrix (memory-intensive for large meshes)
    - r_min is in physical units, not element-relative
    - Automatically detects 2D vs 3D from mesh
    - Slower initialization than StructuredFilter but handles arbitrary topologies
    
    Examples
    --------
    >>> from pyFANTOM.CPU import GeneralMesh, GeneralFilter
    >>> mesh = GeneralMesh(nodes, elements)  # Triangular mesh
    >>> filter = GeneralFilter(mesh=mesh, r_min=0.05)  # Physical radius
    >>> rho_filtered = filter.dot(rho)
    """
    def __init__(self, mesh: GeneralMesh, r_min):
        super().__init__()
        self.dtype = mesh.dtype
        self.nd = mesh.nodes.shape[1]
        if self.nd == 2:
            self.kernel = filter_kernel_2D_general(mesh.elements, mesh.centeroids, r_min)
        else:
            self.kernel = filter_kernel_3D_general(mesh.elements, mesh.centeroids, r_min)
        
        self.kernel = self.kernel.astype(self.dtype)
        self.shape = self.kernel.shape 
        
        self.weights = np.empty(self.shape[0], dtype=self.dtype)
        
    def _matvec(self, rho):
        return (self.kernel @ rho).reshape(rho.shape)
    
    def _rmatvec(self, rho):
        return (self.kernel.T @ rho).reshape(rho.shape)