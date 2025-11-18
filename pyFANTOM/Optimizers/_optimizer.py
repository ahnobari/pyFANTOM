from ..Problem._problem import Problem


class Optimizer:
    """Base optimizer interface.

    Optimizers operate on a :class:`pyFANTOM.Problem._problem.Problem` instance
    and update design variables until convergence.
    """
    def __init__(self, problem: Problem, *args, **kwargs):
        """Create an optimizer bound to a problem.

        Parameters
        - problem: instance of :class:`Problem` providing objective and gradients.
        """
        self.problem = problem
        self.desvars = problem.init_desvars()
        
    def iter(self, *args, **kwargs):
        """Perform a single optimization iteration.

        Subclasses should update `self.desvars` and track any internal state.
        """
        raise NotImplementedError("iter method must be implemented in subclasses.")

    def converged(self, *args, **kwargs):
        """Return True when the optimizer has converged to a solution."""
        raise NotImplementedError("converged method must be implemented in subclasses.")
    
    def logs(self, *args, **kwargs):
        """Return diagnostic logs (history, objective values, or custom stats)."""
        raise NotImplementedError("logs method must be implemented in subclasses.")