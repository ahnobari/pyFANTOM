class Problem:
    """Abstract optimization problem.

    Subclasses represent specific optimization formulations (e.g., minimum
    compliance). The Problem interface exposes methods for initializing and
    querying design variables, computing objective and constraint values and
    their gradients.
    """
    def __init__(self, *args, **kwargs):
        """Initialize problem state.

        Subclasses may accept finite-element handlers, filters, and other
        configuration parameters.
        """
        pass
    
    def init_desvars(self, *args, **kwargs):
        """Return initial design variables as an array-like object."""
        raise NotImplementedError("init_desvars method must be implemented in subclasses.")
    
    def set_desvars(self, *args, **kwargs):
        """Set the current design variables on the problem."""
        raise NotImplementedError("set_desvars method must be implemented in subclasses.")
    
    def get_desvars(self, *args, **kwargs):
        """Return the current design variables."""
        raise NotImplementedError("get_desvars method must be implemented in subclasses.")
    
    def f(self, *args, **kwargs):
        """Compute and return the objective function value."""
        raise NotImplementedError("Objective method must be implemented in subclasses.")

    def nabla_f(self, *args, **kwargs):
        """Return the gradient of the objective with respect to design variables."""
        raise NotImplementedError("Gradient method must be implemented in subclasses.")
    
    def g(self, *args, **kwargs):
        """Compute constraint values (may return a vector)."""
        raise NotImplementedError("Constraints method must be implemented in subclasses.")
    
    def nabla_g(self, *args, **kwargs):
        """Return gradients of the constraints with respect to design variables."""
        raise NotImplementedError("Gradient of constraints method must be implemented in subclasses.")
    
    def constraint_map(self, *args, **kwargs):
        """Return mapping / metadata describing constraints (optional)."""
        raise NotImplementedError("Constraint map method must be implemented in subclasses.")
    
    def ill_conditioned(self, *args, **kwargs):
        """Return True if the current problem state is ill-conditioned.

        Default implementation returns False. Subclasses may override.
        """
        return False
    
    def is_terminal(self):
        """Return True if the problem is terminal/complete (no iterative solves)."""
        return True
    
    def N(self, *args, **kwargs):
        """Return problem size (number of design variables or elements)."""
        raise NotImplementedError("N method must be implemented in subclasses.")
    
    def m(self, *args, **kwargs):
        """Return number of constraints or related metric."""
        raise NotImplementedError("m method must be implemented in subclasses.")
    
    def bounds(self, *args, **kwargs):
        """Return bounds for design variables as (lower, upper) arrays."""
        raise NotImplementedError("Bounds method must be implemented in subclasses.")
    
    def is_independent(self, *args, **kwargs):
        """Return True if design variables are independent (no coupling)."""
        raise NotImplementedError("is_independent method must be implemented in subclasses.")
    