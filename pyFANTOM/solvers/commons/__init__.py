class Solver:
    """
    Base class for linear system solvers.
    
    Abstract interface for solving linear systems K(rho) @ U = F arising from
    finite element analysis. Subclasses implement specific algorithms (direct,
    iterative, multigrid).
    
    Methods
    -------
    solve(rhs, rho=None, **kwargs)
        Solve linear system, returns (U, residual)
    reset()
        Reset solver state (clear factorizations, etc.)
    __call__(rhs, rho=None, **kwargs)
        Convenience: solver(rhs, rho) calls solve()
        
    Notes
    -----
    All solvers accept:
    - rhs: Right-hand side force vector, shape (n_dof,)
    - rho: Design variables (densities), shape (n_elements,)
    - Returns: (U, residual) where U is solution and residual is ||K@U - F||/||F||
    
    Subclasses must implement solve() method.
    """
    def __init__(self):
        pass
    
    def __call__(self, *args, **kwargs):
        """Convenience method: solver(rhs, rho) calls solve(rhs, rho)."""
        return self.solve(*args, **kwargs)
    
    def solve(self, *args, **kwargs):
        """
        Solve linear system K(rho) @ U = rhs.
        
        Parameters
        ----------
        rhs : ndarray
            Right-hand side force vector, shape (n_dof,)
        rho : ndarray, optional
            Design variables (densities), shape (n_elements,)
        **kwargs
            Additional solver-specific parameters
            
        Returns
        -------
        U : ndarray
            Solution vector, shape (n_dof,)
        residual : float
            Relative residual: ||K@U - rhs|| / ||rhs||
            
        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses
        """
        raise NotImplementedError("solve method must be implemented in subclasses.")
    
    def reset(self):
        """
        Reset solver state.
        
        Clears internal state such as factorizations, preconditioners, or iteration
        history. Useful when mesh or boundary conditions change significantly.
        """
        pass