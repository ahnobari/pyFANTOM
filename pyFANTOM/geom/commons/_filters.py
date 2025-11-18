class _TransposeView:
    def __init__(self, original):
        self._original = original
    
    def __matmul__(self, rhs):
        return self._original._rmatvec(rhs)
    
    def dot(self, rhs):
        return self._original._rmatvec(rhs)
    
    @property
    def T(self):
        return self._original

    def getattr(self, name):
        # Delegate any other attribute access to original
        return getattr(self._original, name)
    
class FilterKernel:
    """
    Base class for density filters in topology optimization.
    
    Filters smooth design variables to ensure minimum feature sizes and mesh-independent
    solutions. Implements matrix-vector product interface for forward and adjoint operations.
    
    Attributes
    ----------
    shape : tuple
        Filter matrix dimensions (n_elements, n_elements)
    weights : ndarray
        Filter weights (implementation-specific)
    offsets : ndarray
        Neighbor offsets (implementation-specific)
        
    Methods
    -------
    dot(rho)
        Apply forward filter: filtered_rho = H @ rho
    _rmatvec(sens)
        Apply adjoint filter: H^T @ sens (for sensitivity backpropagation)
    __matmul__(rhs)
        Convenience: filter @ rho calls dot(rho)
    T
        Property returning transpose view for adjoint operations
        
    Notes
    -----
    - Forward filter: rho_filtered = H @ rho_raw
    - Adjoint filter: sens_raw = H^T @ sens_filtered (used in optimization gradients)
    - Subclasses implement _matvec() and _rmatvec() for specific filter types
    - Filter ensures minimum feature size ~2*r_min element widths
    
    Examples
    --------
    >>> filter = StructuredFilter2D(mesh=mesh, r_min=1.5)
    >>> rho_smooth = filter.dot(rho_raw)
    >>> sens_raw = filter._rmatvec(sens_smooth)  # For adjoint method
    """
    def __init__(self):
        self.weights = None
        self.offsets = None
        self.shape = None
        self.matvec = self.dot
        
    def _matvec(self, rho):
        """
        Forward filter application (internal).
        
        Parameters
        ----------
        rho : ndarray
            Raw design variables, shape (n_elements,)
            
        Returns
        -------
        ndarray
            Filtered design variables, shape (n_elements,)
            
        Notes
        -----
        Must be implemented in subclasses.
        """
        raise NotImplementedError("_matvec method must be implemented in subclasses.")
    
    def _rmatvec(self, rho):
        """
        Adjoint filter application (internal).
        
        Parameters
        ----------
        rho : ndarray
            Filtered sensitivities, shape (n_elements,)
            
        Returns
        -------
        ndarray
            Raw sensitivities, shape (n_elements,)
            
        Notes
        -----
        Must be implemented in subclasses. Used for backpropagating gradients
        through the filter in optimization.
        """
        raise NotImplementedError("_rmatvec method must be implemented in subclasses.")
    
    def dot(self, rho):
        """
        Apply forward density filter.
        
        Parameters
        ----------
        rho : ndarray
            Raw design variables, shape (n_elements,)
            
        Returns
        -------
        ndarray
            Filtered design variables, shape (n_elements,)
            
        Raises
        ------
        ValueError
            If input shape doesn't match filter dimensions
        NotImplementedError
            If input is not a 1D vector
        """
        if isinstance(rho, type(self.weights)):
            if rho.ndim == 1:
                if rho.shape[0] == self.shape[1]:
                    return self._matvec(rho)
                else:
                    raise ValueError("Input vector size does not match the filter kernel size.")
            else:
                raise NotImplementedError("Only vector inputs are supported.")
        else:
            raise ValueError(f"Input must be a {type(self.weights)} array vector.")
    
    def __matmul__(self, rhs):
        """Convenience: filter @ rho calls dot(rho)."""
        return self.dot(rhs)
    
    @property
    def T(self):
        """
        Transpose view for adjoint operations.
        
        Returns
        -------
        _TransposeView
            Object that applies _rmatvec when used with @ operator
            
        Examples
        --------
        >>> sens_raw = filter.T @ sens_filtered  # Equivalent to filter._rmatvec(sens_filtered)
        """
        return _TransposeView(self)