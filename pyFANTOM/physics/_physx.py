class Physx:
    """
    Base class for physics models in pyFANTOM.
    
    Abstract interface defining methods all physics implementations must provide.
    Physics models compute element-level quantities (stiffness matrices, areas/volumes, etc.).
    
    Methods
    -------
    K(x0s)
        Compute element stiffness/conductivity matrix from nodal coordinates
    locals(x0s)
        Compute local matrices [D, B, ...] for post-processing (stress, strain, etc.)
    volume(x0s)
        Compute element area (2D) or volume (3D)
    neumann(x0s)
        Compute Neumann boundary contribution (not yet implemented)
        
    Notes
    -----
    All methods accept:
    - Single element: x0s shape (n_nodes_per_element, spatial_dim)
    - Batch: x0s shape (n_elements, n_nodes_per_element, spatial_dim)
    
    Subclasses must implement all abstract methods.
    
    Examples
    --------
    >>> from pyFANTOM import LinearElasticity
    >>> physics = LinearElasticity(E=200e9, nu=0.3)
    >>> K = physics.K(element_nodes)  # Shape: (dof, dof)
    """
    def __init__(self):
        pass
    
    def K(self, x0s):
        raise NotImplementedError("K method must be implemented in subclasses.")
    
    def locals(self, x0s):
        raise NotImplementedError("locals method must be implemented in subclasses.")
    
    def volume(self, x0s):
        raise NotImplementedError("volume method must be implemented in subclasses.")
    
    def neumann(self, x0s):
        raise NotImplementedError("neumann method must be implemented in subclasses.")