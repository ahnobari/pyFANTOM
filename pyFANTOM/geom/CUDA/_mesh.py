from ...core.CUDA._geom import generate_structured_mesh_cuda
import numpy as np
import cupy as cp
from ...physics._physx import Physx
from ...physics.LinearElasticity import LinearElasticity
from ..commons._mesh import StructuredMesh
from ..CPU._mesh import GeneralMesh
import logging
logger = logging.getLogger(__name__)

class CuStructuredMesh2D(StructuredMesh):
    """
    CUDA-accelerated 2D structured mesh with uniform rectangular elements.
    
    GPU version of StructuredMesh2D using CuPy arrays for all data storage.
    Identical API to CPU version but operates on GPU memory for efficient assembly and computation.
    
    Parameters
    ----------
    nx : int
        Number of elements in x-direction
    ny : int
        Number of elements in y-direction
    lx : float
        Physical length of domain in x-direction
    ly : float
        Physical length of domain in y-direction
    dtype : np.dtype, optional
        Data type for arrays (default: np.float64)
    physics : Physx, optional
        Physics model defining material behavior (default: LinearElasticity(E=1.0, nu=1/3))
        
    Notes
    -----
    - All arrays are stored as CuPy arrays on GPU
    - Element stiffness matrix is computed on CPU then transferred to GPU
    - Use with CuStructuredStiffnessKernel for GPU-accelerated assembly
    - Requires CUDA-capable GPU and CuPy installation
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import StructuredMesh2D
    >>> from pyFANTOM import LinearElasticity
    >>> mesh = StructuredMesh2D(nx=128, ny=64, lx=2.0, ly=1.0)
    >>> print(f"GPU memory usage: {mesh.nodes.nbytes + mesh.elements.nbytes} bytes")
    """
    def __init__(self, nx, ny, lx, ly, dtype=np.float64, physics: Physx = LinearElasticity(E=1.0, nu=1/3)):
        super().__init__()
        self.nelx = nx
        self.nely = ny
        self.lx = lx
        self.ly = ly
        self.nel = np.array([nx, ny], dtype=np.int32)
        self.dim = np.array([lx, ly], dtype=dtype)
        self.elements, self.nodes = generate_structured_mesh_cuda(self.dim,self.nel, dtype=dtype)
        self.elements_size = self.elements.shape[1]
        
        self.dx = lx / nx
        self.dy = ly / ny
        
        single_element = self.nodes[self.elements[0]].get()
        K = physics.K(single_element)
        self.K_single = cp.array(K, dtype=dtype)
        
        self.locals = physics.locals(single_element)
        for i in range(len(self.locals)):
            self.locals[i] = cp.array(self.locals[i], dtype=dtype)
            
        A = physics.volume(single_element)
        self.A_single = cp.array([A], dtype=dtype)        
        self.As = self.A_single
        
        self.volume = self.A_single[0] * self.nelx * self.nely
        
        self.dof = int(K.shape[0]/self.elements_size)
        
        self.dtype = dtype
        
        self.physics = physics
        
        self.centroids = cp.meshgrid(
            cp.linspace(self.dx/2, self.lx - self.dx/2, self.nelx, dtype=dtype),
            cp.linspace(self.dy/2, self.ly - self.dy/2, self.nely, dtype=dtype),
            indexing='ij'
        )
        self.centroids = cp.stack(self.centroids, axis=-1).reshape(-1, 2).astype(dtype)


class CuStructuredMesh3D(StructuredMesh):
    """
    CUDA-accelerated 3D structured mesh with uniform hexahedral elements.
    
    GPU version of StructuredMesh3D using CuPy arrays for all data storage.
    Identical API to CPU version but operates on GPU memory for efficient 3D topology optimization.
    
    Parameters
    ----------
    nx : int
        Number of elements in x-direction
    ny : int
        Number of elements in y-direction
    nz : int
        Number of elements in z-direction
    lx : float
        Physical length of domain in x-direction
    ly : float
        Physical length of domain in y-direction
    lz : float
        Physical length of domain in z-direction
    dtype : np.dtype, optional
        Data type for arrays (default: np.float64)
    physics : Physx, optional
        Physics model defining material behavior (default: LinearElasticity(E=1.0, nu=1/3))
        
    Notes
    -----
    - All arrays are stored as CuPy arrays on GPU
    - Use MultiGrid solver for large 3D problems to manage GPU memory
    - Element stiffness matrix computed on CPU then transferred
    - Requires sufficient GPU memory for 3D problems (typically >8GB for 100^3 elements)
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import StructuredMesh3D
    >>> mesh = StructuredMesh3D(nx=64, ny=64, nz=32, lx=1.0, ly=1.0, lz=0.5)
    >>> print(f"Total DOF: {len(mesh.nodes) * mesh.dof}")
    """
    def __init__(self, nx, ny, nz, lx, ly, lz, dtype=np.float64, physics: Physx = LinearElasticity(E=1.0, nu=1/3)):
        super().__init__()
        self.nelx = nx
        self.nely = ny
        self.nelz = nz
        self.lx = lx
        self.ly = ly
        self.lz = lz
        self.nel = np.array([nx, ny, nz], dtype=np.int32)
        self.dim = np.array([lx, ly, lz], dtype=dtype)
        self.elements, self.nodes = generate_structured_mesh_cuda(self.dim,self.nel, dtype=dtype)
        self.elements_size = self.elements.shape[1]
        
        self.dx = lx / nx
        self.dy = ly / ny
        self.dz = lz / nz
        
        single_element = self.nodes[self.elements[0]].get()
        K = physics.K(single_element)
        self.K_single = cp.array(K, dtype=dtype)
        self.locals = physics.locals(single_element)
        for i in range(len(self.locals)):
            self.locals[i] = cp.array(self.locals[i], dtype=dtype)
        A = physics.volume(single_element)
        self.A_single = cp.array([A], dtype=dtype)
        
        self.As = self.A_single
        
        self.volume = self.A_single[0] * self.nelx * self.nely * self.nelz
        
        self.dof = int(K.shape[0]/self.elements_size)
        
        self.dtype = dtype
        
        self.physics = physics
        
        self.centroids = cp.meshgrid(
            cp.linspace(self.dx/2, self.lx - self.dx/2, self.nelx, dtype=dtype),
            cp.linspace(self.dy/2, self.ly - self.dy/2, self.nely, dtype=dtype),
            cp.linspace(self.dz/2, self.lz - self.dz/2, self.nelz, dtype=dtype),
            indexing='ij'
        )
        self.centroids = cp.stack(self.centroids, axis=-1).reshape(-1, 3).astype(dtype)
        
class CuGeneralMesh(GeneralMesh):
    """
    CUDA-accelerated general unstructured mesh.
    
    GPU version of GeneralMesh. Inherits CPU mesh initialization logic then transfers
    all arrays to GPU memory. Supports both uniform and heterogeneous element topologies.
    
    Parameters
    ----------
    nodes : ndarray
        Node coordinates (CPU array), shape (n_nodes, spatial_dim)
    elements : list or ndarray
        Element connectivity (CPU array or list)
    dtype : np.dtype, optional
        Data type for arrays (default: np.float64)
    physics : Physx, optional
        Physics model (default: LinearElasticity(E=1.0, nu=1/3))
        
    Notes
    -----
    - Mesh processing (node cleanup, stiffness computation) happens on CPU
    - Resulting arrays are transferred to GPU after initialization
    - Use CuGeneralStiffnessKernel for GPU-accelerated assembly
    - For large unstructured meshes, GPU offers significant speedup over CPU
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import GeneralMesh
    >>> import numpy as np
    >>> nodes = np.array([[0,0], [1,0], [0,1], [1,1]])
    >>> elements = np.array([[0,1,2], [1,3,2]])
    >>> mesh = GeneralMesh(nodes, elements)
    """
    def __init__(self, nodes, elements, dtype=np.float64, physics: Physx = LinearElasticity(E=1.0, nu=1/3)):
        super().__init__(nodes, elements, dtype, physics)

        if self.is_uniform:
            self.Ks = cp.array(self.Ks, dtype=dtype)
            self.locals = [cp.array(local, dtype=dtype) for local in self.locals]
            self.As = cp.array(self.As, dtype=dtype)
        else:
            self.K_flat = cp.array(self.K_flat, dtype=dtype)
            self.locals_flat = [cp.array(local, dtype=dtype) for local in self.locals_flat]
            self.As = cp.array(self.As, dtype=dtype)
            self.K_ptr = cp.array(self.K_ptr, dtype=cp.int32)
            self.locals_ptr = [cp.array(local_ptr, dtype=cp.int32) for local_ptr in self.locals_ptr]
            self.elements_ptr = cp.array(self.elements_ptr, dtype=cp.int32)
            self.element_sizes = cp.array(self.element_sizes, dtype=cp.int32)
        
        self.elements_flat = cp.array(self.elements_flat, dtype=cp.int32)
        self.centroids = cp.array(self.centeroids, dtype=dtype)
        self.nodes = cp.array(self.nodes, dtype=dtype)
        self.elements = cp.array(self.elements, dtype=cp.int32)