from .._problem import Problem
from ...FiniteElement.CPU.FiniteElement import FiniteElement
from ...geom.CPU._filters import StructuredFilter2D, StructuredFilter3D, GeneralFilter
from typing import Union, Callable
import numpy as np

class ComplianceConstrainedMinimumVolume(Problem):
    """
    Minimum volume topology optimization with compliance constraint.
    
    Inverse formulation of MinimumCompliance: minimize material usage while maintaining
    structure stiffness above threshold. Useful for lightweight design with performance requirements.
    
    Parameters
    ----------
    FE : FiniteElement
        Finite element analysis engine
    filter : StructuredFilter2D, StructuredFilter3D, or GeneralFilter
        Density filter
    E : float, optional
        Young's modulus (default: 1.0)
    void : float, optional
        Minimum density to avoid singularity (default: 1e-6)
    penalty : float, optional
        SIMP penalization exponent (default: 3.0)
    compliance_limit : float, optional
        Maximum allowable compliance (normalized by initial compliance) (default: 0.25)
    penalty_schedule : callable, optional
        Penalty continuation function(p, iteration)
    heavyside : bool, optional
        Apply Heaviside projection (default: True)
    beta : float, optional
        Heaviside sharpness (default: 2)
    eta : float, optional
        Heaviside threshold (default: 0.5)
        
    Notes
    -----
    - **Objective**: Minimize volume
    - **Constraint**: Compliance ≤ compliance_limit * initial_compliance
    - **Use case**: Lightweight design, material cost minimization
    - **Comparison**: More challenging than MinimumCompliance (compliance constraint is nonlinear)
    - **Typical compliance_limit**: 0.2-0.5 of initial design
    
    Examples
    --------
    >>> problem = ComplianceConstrainedMinimumVolume(
    >>>     FE=FE, filter=filter,
    >>>     compliance_limit=0.3,  # 30% of initial compliance
    >>>     penalty=3.0
    >>> )
    """
    def __init__(self,
                 FE: FiniteElement,
                 filter: Union[StructuredFilter2D, StructuredFilter3D, GeneralFilter],
                 E: float = 1.0,
                 void: float = 1e-6,
                 penalty: float = 3.0,
                 compliance_limit: float = 0.25,
                 penalty_schedule: Callable[[float, int], float] = None,
                 heavyside: bool = True,
                 beta: float = 2,
                 eta: float = 0.5):

        super().__init__()

        self.E = E
        self.comp_limit = compliance_limit
        self.void = void
        self.penalty = penalty
        self.penalty_schedule = penalty_schedule
        self.heavyside = heavyside
        self.beta = beta
        self.eta = eta
        self.filter = filter
        self.FE = FE
        self.dtype = FE.dtype
        self.independant = True
        
        self.iteration = 0
        self.desvars = None
        
        self._f = None
        self._g = None
        self._nabla_f = self.FE.mesh.As
        self._nabla_g = None
        self._residual = None
        
        self.num_vars = len(self.FE.mesh.elements)
        self.nel = len(self.FE.mesh.elements)
                
        self.is_3D = self.FE.mesh.nodes.shape[1] == 3
        
        self.is_single_material = True
        
        if self._nabla_f.shape[0] == 1 and self.is_single_material:
            self._nabla_f = np.tile(self._nabla_f, self.num_vars)
        
        self._nabla_f = self._nabla_f / self.FE.mesh.volume
        
    def N(self):
        """
        Return the number of design variables.
        
        Returns
        -------
        int
            Total number of design variables (n_elements)
        """
        return self.num_vars

    def m(self):
        """
        Return the number of constraints.
        
        Returns
        -------
        int
            Number of constraints (1: compliance constraint)
        """
        return 1
    
    def is_independent(self):
        """
        Check if constraints are independent (required for OC optimizer).
        
        Returns
        -------
        bool
            True if constraints are independent (always True for ComplianceConstrainedMinimumVolume)
        """
        return True
    
    def constraint_map(self):
        """
        Return mapping of constraints to design variables.
        
        Returns
        -------
        int
            Scalar value 1 indicating all design variables affect the compliance constraint
            
        Notes
        -----
        Used by optimizers to identify which design variables affect the constraint.
        For this problem, all variables affect compliance.
        """
        return 1
    
    def bounds(self):
        """
        Return bounds for design variables.
        
        Returns
        -------
        tuple
            (lower_bound, upper_bound) where both are 0.0 and 1.0 respectively
            
        Notes
        -----
        Design variables represent normalized densities in [0, 1].
        """
        return (0, 1.0)
            
    def visualize_problem(self, **kwargs):
        """
        Visualize problem setup (mesh, BCs, loads).
        
        Parameters
        ----------
        **kwargs
            Arguments passed to FE.visualize_problem()
            
        Returns
        -------
        matplotlib.axes.Axes or k3d.Plot
            Plot object
        """
        self.FE.visualize_problem(**kwargs)
    
    def visualize_solution(self, **kwargs):
        """
        Visualize optimized design (density distribution).
        
        Parameters
        ----------
        **kwargs
            Arguments passed to FE.visualize_density()
            
        Returns
        -------
        matplotlib.axes.Axes or k3d.Plot
            Plot object
            
        Notes
        -----
        Shows current design variables as density field.
        """
        rho = self.get_desvars()
        # if self.n_material > 1:
        #     rho = rho.reshape(self.n_material, -1).T
        self.FE.visualize_density(rho, **kwargs)
    
    def init_desvars(self):
        """
        Initialize design variables to full density (volume = 1.0).
        
        Sets all design variables to 1.0 (full material) and performs initial
        FEA solve to compute objective and constraints. This ensures initial
        design satisfies compliance constraint.
        
        Notes
        -----
        - All variables set to 1.0 (full material)
        - Resets iteration counter to 0
        - Triggers _compute() to evaluate objective and constraints
        """
        self.desvars = np.ones(self.num_vars, dtype=self.dtype)
        self.iteration = 0
        self._compute()
    
    def set_desvars(self, desvars: np.ndarray):
        """
        Set design variables and recompute objective/constraints.
        
        Parameters
        ----------
        desvars : ndarray
            Design variables, shape (n_vars,). Values should be in [0, 1]
            
        Raises
        ------
        ValueError
            If desvars shape doesn't match num_vars
            
        Notes
        -----
        - Triggers FEA solve and sensitivity computation
        - Increments iteration counter
        - Updates cached values: _f, _g, _nabla_f, _nabla_g
        """
        if desvars.shape[0] != self.num_vars:
            raise ValueError(f"Expected {self.num_vars} design variables, got {desvars.shape[0]}.")
        
        self.desvars = desvars
        self._compute()
        self.iteration += 1
    
    def get_desvars(self):
        """
        Get current design variables.
        
        Returns
        -------
        ndarray
            Current design variables, shape (n_vars,)
            
        Notes
        -----
        Returns raw (unfiltered) design variables. For filtered densities,
        access via problem.filter.dot(problem.get_desvars()).
        """
        return self.desvars
    
    def penalize(self, rho: np.ndarray):
        pen = self.penalty

        if self.penalty_schedule is not None:
            pen = self.penalty_schedule(self.penalty, self.iteration)

        if self.is_single_material:
            if self.heavyside:
                _rho = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            else:
                _rho = rho
            _rho = _rho**pen
            _rho = np.clip(_rho, self.void, 1.0)
            _rho = _rho*self.E
            _rho = np.clip(_rho, self.void, None)
            
            return _rho
            
        else:
            if self.heavyside:
                _rho = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            else:
                _rho = rho
            rho_ = _rho**pen
            rho__ = 1 - rho_
            
            rho_ *= (
                rho__[
                    :,
                    np.where(~np.eye(self.n_material, dtype=bool))[1].reshape(
                        self.n_material, -1
                    ),
                ]
                .transpose(1, 0, 2)
                .prod(axis=-1)
                .T
            )

            E = (rho_ * self.E_mul[np.newaxis, :]).sum(axis=1)
            E = np.clip(E, self.void, None)
            
            return E

    def penalize_grad(self, rho: np.ndarray):
        pen = self.penalty

        if self.penalty_schedule is not None:
            pen = self.penalty_schedule(self.penalty, self.iteration)
            
        if self.is_single_material:
            if self.heavyside:
                rho_heavy = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
                df = pen * rho_heavy ** (pen - 1) * self.beta * (1 - np.tanh(self.beta * (rho-self.eta))**2) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            else:
                df = pen * rho ** (pen - 1)

            return df*self.E
        
        else:
            if self.heavyside:
                rho_heavy = (np.tanh(self.beta + self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
                
                rho_ = pen * rho_heavy ** (pen - 1)
                rho__ = 1 - rho_heavy**pen
                rho___ = rho_heavy**pen

                d = rho__[np.newaxis, :, :].repeat(self.n_material, 0)
                d[np.arange(self.n_material), :, np.arange(self.n_material)] = rho___.T
                d = d[np.newaxis, :, :, :].repeat(self.n_material, 0)
                d[np.arange(self.n_material), :, :, np.arange(self.n_material)] = 1
                d = d.prod(axis=-1).transpose(0, 2, 1)

                mul = -rho_.T[:, :, np.newaxis].repeat(self.n_material, -1)
                mul[np.arange(self.n_material), :, np.arange(self.n_material)] *= -1

                d *= mul
                d = d @ self.E_mul[:, np.newaxis]
                
                df = d.squeeze().T * self.beta * (1 - np.tanh(self.beta * (rho-self.eta))**2) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
                
                return df
            else:
                rho_ = pen * rho ** (pen - 1)
                rho__ = 1 - rho**pen
                rho___ = rho**pen

                d = rho__[np.newaxis, :, :].repeat(self.n_material, 0)
                d[np.arange(self.n_material), :, np.arange(self.n_material)] = rho___.T
                d = d[np.newaxis, :, :, :].repeat(self.n_material, 0)
                d[np.arange(self.n_material), :, :, np.arange(self.n_material)] = 1
                d = d.prod(axis=-1).transpose(0, 2, 1)

                mul = -rho_.T[:, :, np.newaxis].repeat(self.n_material, -1)
                mul[np.arange(self.n_material), :, np.arange(self.n_material)] *= -1

                d *= mul
                d = d @ self.E_mul[:, np.newaxis]
                
                df = d.squeeze().T
                
                return df
    
    def _compute(self):
        
        rho = self.filter.dot(self.desvars)
        rho_ = self.penalize(rho)
        
        U, residual = self.FE.solve(rho_)
        
        compliance = self.FE.rhs.dot(U)

        df = self.FE.kernel.process_grad(U)

        if rho.ndim > 1:
            df = df.reshape(-1,1)
        
        dr = self.penalize_grad(rho) * df

        dr = dr.reshape(dr.shape[0], -1)

        for i in range(dr.shape[1]):
            dr[:, i] = self.filter._rmatvec(dr[:, i])
        
        self._g = compliance
        self._nabla_g = dr.reshape(-1)
        self._residual = residual
        
        vf = (self._nabla_f @ self.desvars.reshape(-1, 1)).squeeze()
        
        self._f = vf

    def f(self, rho: np.ndarray = None):
        """
        Compute objective function value (volume).
        
        Parameters
        ----------
        rho : ndarray, optional
            Design variables for linearization. If None, returns cached value
            
        Returns
        -------
        float
            Volume fraction (normalized by total domain volume). Lower is better.
            
        Notes
        -----
        - If rho provided: returns linearized approximation f(x) + df/dx @ (rho - x)
        - If rho is None: returns cached value from last set_desvars() call
        - Objective is to minimize material usage while satisfying compliance constraint
        """
        if rho is None:
            return self._f
        else:
            return (self._nabla_f @ rho.reshape(-1, 1)).squeeze()

    def nabla_f(self, rho: np.ndarray = None):
        """
        Compute objective function gradient (volume sensitivities).
        
        Parameters
        ----------
        rho : ndarray, optional
            Unused (for interface compatibility)
            
        Returns
        -------
        ndarray
            Gradient of volume w.r.t. design variables, shape (n_vars,)
            
        Notes
        -----
        - Constant gradient: 1/volume_total per element
        - All elements contribute equally to volume
        - Used by optimizers to minimize material usage
        """
        return self._nabla_f
    
    def g(self, rho=None):
        """
        Compute constraint values (compliance constraint violations).
        
        Parameters
        ----------
        rho : ndarray, optional
            Design variables. If None, uses current desvars
            
        Returns
        -------
        float
            Normalized compliance constraint violation. Negative = satisfied.
            g = (compliance - compliance_limit) / compliance_limit
            
        Notes
        -----
        - Constraint satisfied when g <= 0
        - Normalized by compliance_limit for better scaling
        - If rho provided, uses linearized approximation
        """
        if rho is None:
            return (self._g - self.comp_limit)/self.comp_limit
        else:
            # local linear approximation
            return (self._g + (rho - self.desvars).dot(self._nabla_g) - self.comp_limit)/self.comp_limit

    def nabla_g(self):
        """
        Compute constraint gradients (compliance sensitivities).
        
        Returns
        -------
        ndarray
            Gradient of compliance constraint w.r.t. design variables, shape (n_vars,)
            
        Notes
        -----
        - Normalized by compliance_limit for better scaling
        - Uses adjoint method: dC/drho = -U^T @ dK/drho @ U
        - Includes filter adjoint: sens_raw = H^T @ sens_filtered
        - Used by optimizers to enforce compliance constraint
        """
        return self._nabla_g/self.comp_limit

    def ill_conditioned(self):
        """
        Check if FEA system is ill-conditioned.
        
        Returns
        -------
        bool
            True if residual >= 1e-2 (indicates poor solver convergence)
            
        Notes
        -----
        - Residual > 1e-2 suggests numerical issues (check void parameter, penalty)
        - May indicate near-singular stiffness matrix (too many void elements)
        - Consider increasing void parameter or reducing penalty
        """
        if self._residual >= 1e-2:
            return True
        else:
            return False
        
    def is_terminal(self):
        """
        Check if penalty continuation has reached final value.
        
        Returns
        -------
        bool
            True if penalty schedule is complete or not used
            
        Notes
        -----
        - Used by optimizers to determine if continuation is finished
        - If penalty_schedule is None, always returns True
        - If penalty_schedule exists, checks if current iteration has reached final penalty
        """
        if self.penalty_schedule is not None:
            if self.penalty_schedule(self.penalty, self.iteration) == self.penalty:
                return True
            else:
                return False
        else:
            return True
        
    def logs(self):
        """
        Return diagnostic information for current iteration.
        
        Returns
        -------
        dict
            Dictionary with keys:
            - 'iteration': Current iteration number
            - 'residual': FEA solver residual (||K@U - F|| / ||F||)
            
        Notes
        -----
        Used by optimizers to track convergence and diagnose issues.
        """
        return {
            'iteration': int(self.iteration),
            'residual': float(self._residual)
        }