import cupy as np
from .._optimizer import Optimizer
from ...Problem._problem import Problem
import time

class PGD(Optimizer):
    """
    CUDA-accelerated Projected Gradient Descent optimizer.
    
    GPU version of PGD using CuPy arrays. Fast first-order optimizer using BFGS-based
    step size adaptation and projection onto feasible set. Identical API to CPU version.
    
    Parameters
    ----------
    problem : Problem
        Optimization problem with CUDA backend
    change_tol : float, optional
        Design variable change tolerance (default: 1e-4)
    fun_tol : float, optional
        Objective function change tolerance (default: 1e-6)
    maxiter_N : int, optional
        Max iterations for Newton's method in projection (default: 50)
    tol_B : float, optional
        Bisection tolerance (default: 1e-8)
    tol_N : float, optional
        Newton solver tolerance (default: 1e-6)
    C : float, optional
        Large constant for penalty (default: 1e12)
    fall_back_move : float, optional
        Fallback step size when BFGS fails (default: 0.2)
    alpha_max : float, optional
        Maximum step size (default: 1e2)
    relaxation : float, optional
        Step size relaxation parameter (default: 1.0)
    warmup_iter : int, optional
        Iterations before enforcing strict feasibility (default: 50)
    timer : bool, optional
        Return iteration time (default: False)
    conjugate_directions : bool, optional
        Use conjugate gradient directions (default: True)
        
    Notes
    -----
    - All arrays stored as CuPy arrays on GPU
    - Faster than CPU: ~30-50% faster per iteration
    - Best for: Simple bounds, single constraint problems
    - Less robust: May struggle with multiple nonlinear constraints
    - Requires CUDA-capable GPU and CuPy
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import PGD
    >>> optimizer = PGD(problem=problem, conjugate_directions=True)
    >>> for i in range(100):
    >>>     optimizer.iter()
    >>>     if optimizer.converged():
    >>>         break
    """
    def __init__(self, 
                 problem: Problem,
                 change_tol=1e-4,
                 fun_tol=1e-6,
                 maxiter_N=50,
                 tol_B=1e-8,
                 tol_N=1e-6,
                 C=1e12,
                 fall_back_move=0.2,
                 alpha_max=1e2,
                 relaxation=1.0,
                 warmup_iter=50,
                 timer = False,
                 conjugate_directions=True):
        super().__init__(problem)

        self.last_desvars = np.copy(problem.get_desvars())
        self.last_f = problem.f()
        self.last_nabla_f = self.problem.nabla_f().copy()
                
        self.m = problem.m()
        self.lambda_map = problem.constraint_map()
        self.bounds = problem.bounds()
        self.change = np.inf
        self.change_f = np.inf
        self.change_tol = change_tol
        self.fun_tol = fun_tol
        self.tol_B = tol_B
        self.tol_N = tol_N
        self.C = C
        self.fall_back_move = fall_back_move
        self.alpha_max = alpha_max
        self.maxiter_N = maxiter_N
        self.timer = timer
        self.relaxation = relaxation
        self.warmup_iter = warmup_iter
        self.conjugate = conjugate_directions

        self.iteration = 0
        
        self.is_independent = problem.is_independent()
        
        desvars = self.problem.get_desvars()
        desvars_new = desvars.copy()
        dg = self.problem.nabla_g()
        desvars_new = self.project_to_feasible(desvars_new, desvars, dg)
        self.problem.set_desvars(desvars_new)
    
    def alpha(self, desvars, df):
        
        ndf = max(np.linalg.norm(df)/np.sqrt(self.problem.N()), 1e-6)
        if self.conjugate:
            if self.iteration == 0 :
                d = -df
            else:
                Delta_x_n_1 = -self.last_nabla_f
                Delta_x_n = -df
                beta = (Delta_x_n.T @ (Delta_x_n - Delta_x_n_1))/(Delta_x_n_1.T @ Delta_x_n_1)
                beta = max(beta,0)
                d = -df + beta * self.d_last
        else:
            d = -df
        if self.iteration == 0:
            alpha = self.fall_back_move/np.linalg.norm(df,ord=np.inf)
        elif self.problem.g().max() > self.tol_N and self.iteration>self.warmup_iter:
            alpha = self.fall_back_move/np.linalg.norm(df,ord=np.inf)
        else:
            s = self.last_desvars - desvars
            y = self.last_nabla_f - df
            sy = np.dot(s, y)

            if sy > 1e-6:
                alpha = np.dot(s, s) / sy
                L = np.nan_to_num(np.linalg.norm(y) / np.linalg.norm(s), nan=0)
                alpha = np.clip(alpha, 1e-6, min(2 * self.relaxation/ (L + 1e-6), self.alpha_max))
            else:
                if np.allclose(y, 0):
                    # perfect linear behaviour
                    L = np.linalg.norm(df) * (np.linalg.norm(s) / np.sqrt(self.problem.N())) # Adjust step size based on projected change in variables
                    alpha = self.relaxation / (L + 1e-6)
                else:
                    L = np.nan_to_num(np.linalg.norm(y) / np.linalg.norm(s), nan=0)

                    alpha = min(self.relaxation / (L + 1e-6), self.alpha_max)
            
        return alpha, d
    
    def project_to_feasible(self, desvars_new, desvars, dg):
        
        if self.is_independent or self.m == 1:
            l1 = 0 * np.ones(self.m, dtype=desvars.dtype)
            l2 = -1e12 * np.ones(self.m, dtype=desvars.dtype)
            
            if self.m > 1:
                d_new = np.clip(desvars_new + (l1.reshape(1, -1) @ dg).reshape(-1), self.bounds[0], self.bounds[1])
            else:
                d_new = np.clip(desvars_new + l1 * dg, self.bounds[0], self.bounds[1])
            
            valids = self.problem.g(d_new) <= 0.
            if np.all(valids):
                return d_new
            
            while np.any((l2 - l1) / (l2 + l1) > self.tol_B):
                l_mid = (l1 + l2) / 2
                
                if self.m > 1:
                    d_new = np.clip(desvars_new + (l_mid.reshape(1, -1) @ dg).reshape(-1), self.bounds[0], self.bounds[1])
                else:
                    d_new = np.clip(desvars_new + l_mid * dg, self.bounds[0], self.bounds[1])
                
                valids = self.problem.g(d_new) <= 0.
                l2[valids] = l_mid[valids]
                l1[~valids] = l_mid[~valids]

            desvars_new = d_new
        
        else:
            # First check if single active constraint solution exists
            l1 = 0 * np.ones(self.m, dtype=desvars.dtype)
            l2 = -1e12 * np.ones(self.m, dtype=desvars.dtype)
            G = self.problem.g(desvars)
            single_con_found = False
            
            while np.any((l2 - l1) / (l2 + l1) > self.tol_B):
                l_mid = (l1 + l2) / 2
                d_news = np.clip(desvars_new.reshape(1, -1) + l_mid.reshape(-1,1) * dg, self.bounds[0], self.bounds[1])
                diff = ((desvars.reshape(1, -1) - d_news) * dg).sum(axis=1)
                
                valids = G - diff <= 0.
                l2[valids] = l_mid[valids]
                l1[~valids] = l_mid[~valids]
                
            for i in range(self.m):
                if np.all(self.problem.g(d_news[i]) <= self.tol_N):
                    desvars_new = d_news[i]
                    single_con_found = True
                    break

            if not single_con_found:
                l = l_mid
                for _ in range(self.maxiter_N):
                    Phi, J_Phi, d_new = self._Phi(l, dg, desvars_new, J=True)
                    
                    if np.all(Phi <= self.tol_N):
                        break
                        
                    Delta = np.linalg.solve(J_Phi, Phi)
                    
                    alpha, l, Phi = self.wolfe_line_search(l, Delta, dg, desvars_new)
                    
                desvars_new = np.clip(desvars_new + (l.reshape(1, -1) @ dg).reshape(-1), self.bounds[0], self.bounds[1])
                
        return desvars_new
    
    def iter(self):
        """
        Perform one PGD optimization iteration on GPU.
        
        Updates design variables using projected gradient descent with adaptive
        step size (BFGS-based) and projection onto feasible set. All operations
        performed on GPU using CuPy.
        
        Returns
        -------
        float, optional
            If timer=True, returns iteration time in seconds
            
        Notes
        -----
        - Computes step size using BFGS approximation (alpha)
        - Uses conjugate gradient directions if enabled
        - Projects update onto feasible set (satisfies constraints)
        - For independent constraints: uses bisection
        - For coupled constraints: uses Newton's method with Wolfe line search
        """
        if self.timer:
            start_time = time.time()
        desvars = self.problem.get_desvars()
        dg = self.problem.nabla_g()
        df = self.problem.nabla_f()
        f = self.problem.f()
        
        alpha, d = self.alpha(desvars, df)
            
        self.last_nabla_f = df.copy()
        self.d_last = d.copy()
        self.last_desvars = desvars.copy()
        self.last_f = f
        
        desvars_new = desvars + alpha * d
        self.iteration += 1
        
        desvars_new = self.project_to_feasible(desvars_new, desvars, dg)
        
        if self.timer:
            end_time = time.time()
        self.problem.set_desvars(desvars_new)
        self.change = np.linalg.norm(self.last_desvars - desvars_new)
        self.change_f = np.abs((self.problem.f()-self.last_f)/self.problem.f())
        
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
        
    
    def _Phi(self, l, dg, desvars_new, J=False):
        d_new = desvars_new + (l.reshape(1, -1) @ dg).reshape(-1)
        
        if J:
            non_saturated = np.logical_and(
                d_new > self.bounds[0],
                d_new < self.bounds[1]
            ).astype(desvars_new.dtype)
        
        d_new = np.clip(d_new, self.bounds[0], self.bounds[1])
            
        h = self.problem.g(d_new) + 2 * l / self.C
        
        if J:
            J_h = (dg * non_saturated.reshape(1,-1)) @ dg.T + 2 * np.eye(self.m, dtype=desvars_new.dtype)/self.C
            
        Phi = (h>0) * h + (h<=0) * (-l)
        
        if J:
            D_Phi = np.diag(h > 0).astype(desvars_new.dtype)
            J_Phi = D_Phi @ J_h - np.eye(self.m, dtype=desvars_new.dtype) + D_Phi
            
            return Phi, J_Phi, d_new
        
        return Phi, d_new
    
    def wolfe_line_search(self, l, delta, dg, desvars_new, c1=1e-4, c2=0.9, max_iter=10):
        """
        Wolfe conditions line search for Newton step
        """
        # Initial merit function and gradient
        Phi0, J_Phi, _ = self._Phi(l, dg, desvars_new, J=True)
        merit0 = 0.5 * np.dot(Phi0, Phi0)
        
        # Gradient of merit function w.r.t. l: grad_merit = J_Phi^T @ Phi
        grad_merit0 = J_Phi.T @ Phi0
        directional_deriv = -np.dot(grad_merit0, delta)
        
        alpha = 1.0
        
        for i in range(max_iter):
            l_new = l - alpha * delta
            
            # Compute new merit function
            Phi_new, J_Phi_new, _ = self._Phi(l_new, dg, desvars_new, J=True)
            merit_new = 0.5 * np.dot(Phi_new, Phi_new)
            
            # Armijo condition (sufficient decrease)
            if merit_new <= merit0 + c1 * alpha * directional_deriv:
                
                # Compute gradient for curvature condition
                grad_merit_new = J_Phi_new.T @ Phi_new
                new_directional_deriv = -np.dot(grad_merit_new, delta)
                
                # Wolfe curvature condition
                if new_directional_deriv >= c2 * directional_deriv:
                    return alpha, l_new, Phi_new
            
            alpha *= 0.5
        
        # Fallback: return last step
        return alpha, l_new, Phi_new