import numpy as np
from .._optimizer import Optimizer
from ...Problem._problem import Problem
import time

class OC(Optimizer):
    """
    Optimality Criteria (OC) method for topology optimization.
    
    Classic heuristic optimizer specifically designed for topology optimization with independent
    volume constraints. Very fast but limited to specific problem formulations.
    
    Parameters
    ----------
    problem : Problem
        Optimization problem (must have independent constraints)
    move : float, optional
        Move limit for design variables (default: 0.2)
    change_tol : float, optional
        Convergence tolerance for design change (default: 1e-4)
    fun_tol : float, optional
        Convergence tolerance for objective change (default: 1e-6)
    timer : bool, optional
        Return iteration time (default: False)
        
    Attributes
    ----------
    ocP : ndarray
        OC update parameter computed from sensitivities
        
    Methods
    -------
    iter()
        Perform one OC iteration
    converged()
        Check convergence
        
    Notes
    -----
    - **Best for**: Classic minimum compliance with volume constraint
    - **Fastest**: 2-3x faster than MMA per iteration
    - **Limited scope**: Only works with independent constraints
    - **Heuristic**: Not guaranteed to find KKT point
    - **Typical use**: Quick prototyping, benchmarking
    - **Update formula**: x_new = x * sqrt(-df/dg) projected onto move limits
    - **Bisection**: Finds Lagrange multiplier via bisection on volume constraint
    
    Examples
    --------
    >>> from pyFANTOM.CPU import OC
    >>> optimizer = OC(problem=problem, move=0.2)
    >>> for i in range(200):
    >>>     optimizer.iter()
    >>>     print(f"Iter {i}: C={optimizer.problem.f():.2e}")
    >>>     if optimizer.converged():
    >>>         break
    """
    def __init__(self, problem: Problem, move: float = 0.2, change_tol=1e-4, fun_tol=1e-6, timer=False):
        super().__init__(problem)

        if not problem.is_independent():
            raise ValueError("OC optimizer requires a problem with independent constraints.")

        self.last_desvars = np.copy(problem.get_desvars())
        self.last_f = problem.f()
                
        self.m = problem.m()
        self.ocP = None
        self.lambda_map = problem.constraint_map()
        self.move = move
        self.bounds = problem.bounds()
        self.change = np.inf
        self.change_f = np.inf
        self.change_tol = change_tol
        self.fun_tol = fun_tol
        self.timer = timer
        
        self.iteration = 0
        
        self.iteration = 0
            
    def _OCP(self):
        desvars = self.problem.get_desvars()
        dg = self.problem.nabla_g()
        df = self.problem.nabla_f()
        
        if self.m > 1:
            ocP = desvars * np.nan_to_num(np.sqrt(np.maximum(-df / dg.sum(axis=0), 0)), nan=0)
        else:
            ocP = desvars * np.nan_to_num(np.sqrt(np.maximum(-df / dg, 0)), nan=0)
            
        if np.abs(ocP).sum() == 0:
            ocP = np.ones_like(ocP) * 1e-3
        self.ocP = ocP
        
    
    def iter(self):
        """
        Perform one OC optimization iteration.
        
        Updates design variables using optimality criteria update formula with
        bisection to find Lagrange multiplier satisfying volume constraint.
        
        Returns
        -------
        float, optional
            If timer=True, returns iteration time in seconds
            
        Notes
        -----
        - Computes OC parameter: ocP = x * sqrt(-df/dg)
        - Uses bisection to find Lagrange multiplier lambda
        - Updates: x_new = clip(xL, xU, ocP/lambda)
        - Enforces move limits: xL = x - move, xU = x + move
        """
        if self.timer:
            start_time = time.time()
            
        self._OCP()
        
        desvars = self.problem.get_desvars()
        xU = np.clip(desvars + self.move, self.bounds[0], self.bounds[1])
        xL = np.clip(desvars - self.move, self.bounds[0], self.bounds[1])
        
        l1 = 1e-9 * np.ones(self.m, dtype=desvars.dtype)
        l2 = 1e9 * np.ones(self.m, dtype=desvars.dtype)
        
        while np.any((l2 - l1) / (l2 + l1) > 1e-8):
            l_mid = (l1 + l2) / 2
            
            if self.m > 1:
                l_mid_adjusted = (l_mid.reshape(1, -1) @ self.lambda_map).reshape(-1)
            else:
                l_mid_adjusted = l_mid
            
            desvars_new = np.maximum(
                self.bounds[0], np.maximum(xL, np.minimum(1.0, np.minimum(xU, self.ocP / l_mid_adjusted)))
            )

            valids = self.problem.g(desvars_new) <= 0.

            l2[valids] = l_mid[valids]
            l1[~valids] = l_mid[~valids]
        
        self.iteration += 1
        
        if self.timer:
            end_time = time.time()
        
        self.problem.set_desvars(desvars_new)
        self.change = np.linalg.norm(self.last_desvars - desvars_new)
        self.change_f = np.abs((self.problem.f()-self.last_f)/self.problem.f())
        
        self.last_f = self.problem.f()
        self.last_desvars = desvars_new.copy()
        
        if self.timer:
            return end_time - start_time
        
    def converged(self, *args, **kwargs):
        """
        Check if optimizer has converged.
        
        Returns
        -------
        bool
            True if convergence criteria are met:
            - Problem penalty continuation is complete (is_terminal() == True)
            - Design variable change <= change_tol
            - Objective function change <= fun_tol
            
        Notes
        -----
        Convergence requires both design change and objective change to be below
        tolerances. Also checks that penalty continuation (if used) has finished.
        """
        if not self.problem.is_terminal():
            return False
        elif self.change <= self.change_tol and self.change_f <= self.fun_tol:
            return True
        else:
            return False

    def logs(self):
        """
        Return diagnostic information for current iteration.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'objective': Current objective function value
            - 'variable change': L2 norm of design variable change
            - 'function change': Relative objective function change
            - Additional keys from problem.logs() (e.g., 'iteration', 'residual')
            
        Notes
        -----
        Used for monitoring optimization progress and convergence.
        """
        problem_logs = self.problem.logs()
        return{
            'objective': float(self.last_f),
            'variable change': float(self.change),
            'function change': float(self.change_f),
            **problem_logs
        }