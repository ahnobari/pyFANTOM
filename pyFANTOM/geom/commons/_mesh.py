class Mesh:
    """
    Base class for finite element meshes.
    
    Abstract interface for mesh representations. Meshes store node coordinates,
    element connectivity, and provide element-level quantities (stiffness matrices,
    volumes, etc.).
    
    Notes
    -----
    Subclasses must provide:
    - nodes: Node coordinates, shape (n_nodes, spatial_dim)
    - elements: Element connectivity
    - dof: Degrees of freedom per node
    - volume: Total domain volume
    """
    pass

class StructuredMesh(Mesh):
    """
    Base class for structured (uniform grid) meshes.
    
    Structured meshes have uniform element sizes and enable optimized assembly
    using a single precomputed element stiffness matrix (K_single).
    
    Attributes
    ----------
    K_single : ndarray
        Single element stiffness matrix (same for all elements)
    elements : ndarray
        Element connectivity array
    nodes : ndarray
        Node coordinates
    dof : int
        Degrees of freedom per node
    volume : float
        Total domain volume
        
    Notes
    -----
    Structured meshes are more memory-efficient and faster than general meshes
    for uniform grids. Use StructuredMesh2D or StructuredMesh3D for concrete implementations.
    """
    def __init__(self):
        pass