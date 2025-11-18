from .._problem import Problem
from ...FiniteElement.CUDA.FiniteElement import FiniteElement
from ...geom.CUDA._filters import (
    CuStructuredFilter2D as StructuredFilter2D, 
    CuStructuredFilter3D as StructuredFilter3D, 
    CuGeneralFilter as GeneralFilter)
from typing import Union, Callable
import cupy as np

class MinimumCompliance(Problem):
    """
    CUDA-accelerated minimum compliance topology optimization problem.
    
    GPU version of MinimumCompliance using CuPy arrays. Identical API to CPU version
    but operates on GPU memory for maximum performance in large-scale optimization.
    
    Parameters
    ----------
    FE : FiniteElement
        CUDA finite element analysis engine with mesh, kernel, and solver
    filter : StructuredFilter2D, StructuredFilter3D, or GeneralFilter
        CUDA density filter for ensuring minimum feature sizes
    E_mul : list of float, optional
        Young's modulus multipliers for each material (default: [1.0] for single material)
    void : float, optional
        Minimum density to avoid singularity (default: 1e-6)
    penalty : float, optional
        SIMP penalization exponent (default: 3.0). Higher = more binary designs
    volume_fraction : list of float, optional
        Volume fraction constraint for each material (default: [0.25])
    penalty_schedule : callable, optional
        Function(p, iteration) for penalty continuation. If None, uses constant penalty
    heavyside : bool, optional
        Apply Heaviside projection for sharper 0-1 designs (default: True)
    beta : float, optional
        Heaviside projection sharpness parameter (default: 2)
    eta : float, optional
        Heaviside projection threshold (default: 0.5)
        
    Notes
    -----
    - All arrays stored as CuPy arrays on GPU
    - 5-10x faster than CPU for large problems (>1M DOF)
    - Requires CUDA-capable GPU and CuPy
    - Identical SIMP formulation and Heaviside projection as CPU version
    
    Examples
    --------
    >>> from pyFANTOM.CUDA import *
    >>> mesh = StructuredMesh2D(nx=256, ny=128, lx=2.0, ly=1.0)
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> solver = CG(kernel=kernel)
    >>> FE = FiniteElement(mesh=mesh, kernel=kernel, solver=solver)
    >>> filter = StructuredFilter2D(mesh=mesh, r_min=1.5)
    >>> problem = MinimumCompliance(FE=FE, filter=filter, penalty=3.0, volume_fraction=[0.3])
    """
    def __init__(self,
                 FE: FiniteElement,
                 filter: Union[StructuredFilter2D, StructuredFilter3D, GeneralFilter],
                 E_mul: list[float] = [1.0],
                 void: float = 1e-6,
                 penalty: float = 3.0,
                 volume_fraction: list[float] = [0.25],
                 penalty_schedule:  Callable[[float, int], float] = None,
                 heavyside: bool = True,
                 beta: float = 2,
                 eta: float = 0.5):

        super().__init__()
        
        if len(E_mul) != len(volume_fraction):
            raise ValueError("E and volume_fraction must have the same length.")
        
        if len(E_mul) == 1:
            self.is_single_material = True
            self.E_mul = E_mul[0]
            self.volume_fraction = volume_fraction[0]
            self.n_material = 1
        else:
            self.is_single_material = False
            self.E_mul = np.array(E_mul, dtype=FE.dtype)
            self.volume_fraction = np.array(volume_fraction, dtype=FE.dtype)
            self.n_material = len(E_mul)

        self.void = void
        self.penalty = penalty
        self.penalty_schedule = penalty_schedule
        self.heavyside = heavyside
        self.beta = beta
        self.eta = eta
        self.filter = filter
        self.FE = FE
        self.dtype = FE.dtype
        
        self.iteration = 0
        self.desvars = None
        
        self._f = None
        self._g = None
        self._nabla_f = None
        self._nabla_g = self.FE.mesh.As/self.FE.mesh.volume
        self._residual = None
        
        self.num_vars = len(self.FE.mesh.elements) * len(E_mul)
        self.nel = len(self.FE.mesh.elements)
        
        if self._nabla_g.shape[0] == 1 and self.is_single_material:
            self._nabla_g = np.tile(self._nabla_g, self.num_vars)
        elif self._nabla_g.shape[0] == 1 and not self.is_single_material:
            self._nabla_g = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            for i in range(self.n_material):
                self._nabla_g[i, i*self.nel:(i+1)*self.nel] = self.FE.mesh.As[0]/self.FE.mesh.volume
        elif self._nabla_g.shape[0] != self.num_vars and not self.is_single_material:
            self._nabla_g = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            for i in range(self.n_material):
                self._nabla_g[i, i*self.nel:(i+1)*self.nel] = self.FE.mesh.As/self.FE.mesh.volume

        self.is_3D = self.FE.mesh.nodes.shape[1] == 3
            
    def N(self):
        """
        Return the number of design variables.
        
        Returns
        -------
        int
            Total number of design variables (n_elements * n_materials)
        """
        return self.num_vars

    def m(self):
        """
        Return the number of constraints.
        
        Returns
        -------
        int
            Number of volume constraints (1 for single material, n_materials for multi-material)
        """
        return 1 if self.is_single_material else len(self.E_mul)
    
    def is_independent(self):
        """
        Check if constraints are independent (required for OC optimizer).
        
        Returns
        -------
        bool
            True if constraints are independent (always True for MinimumCompliance)
        """
        return True
    
    def constraint_map(self):
        """
        Return mapping of constraints to design variables.
        
        Returns
        -------
        int or cp.ndarray
            - Single material: 1 (scalar)
            - Multi-material: array of shape (n_materials, n_vars) with 1s indicating
              which design variables belong to each material constraint
              
        Notes
        -----
        Used by optimizers to identify which design variables affect each constraint.
        For multi-material, constraint i affects variables [i*n_elements:(i+1)*n_elements].
        """
        if self.is_single_material:
            return 1
        else:
            mapping = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            
            for i in range(self.n_material):
                mapping[i, i*self.nel:(i+1)*self.nel] = 1
                
            return mapping
    
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
            
        Notes
        -----
        GPU arrays are automatically transferred to CPU for visualization.
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
        Shows current design variables as density field. For multi-material problems,
        displays combined material distribution. GPU arrays transferred to CPU for visualization.
        """
        rho = self.get_desvars()
        if self.n_material > 1:
            rho = rho.reshape(self.n_material, -1).T
        self.FE.visualize_density(rho, **kwargs)
    
    def init_desvars(self):
        """
        Initialize design variables to uniform density at volume fraction.
        
        Sets all design variables to the volume fraction constraint value and
        performs initial FEA solve to compute objective and constraints.
        
        Notes
        -----
        - Single material: all variables set to volume_fraction
        - Multi-material: all variables set to min(volume_fraction) for each material
        - Resets iteration counter to 0
        - Triggers _compute() to evaluate objective and constraints
        - All operations performed on GPU
        """
        if self.is_single_material:
            self.desvars = np.ones(self.num_vars, dtype=self.dtype) * self.volume_fraction
        else:
            self.desvars = np.ones(self.num_vars, dtype=self.dtype) * min(self.volume_fraction)
        self.iteration = 0
        self._compute()
    
    def set_desvars(self, desvars: np.ndarray):
        """
        Set design variables and recompute objective/constraints.
        
        Parameters
        ----------
        desvars : cp.ndarray
            Design variables on GPU, shape (n_vars,). Values should be in [0, 1]
            
        Raises
        ------
        ValueError
            If desvars shape doesn't match num_vars
            
        Notes
        -----
        - Triggers FEA solve and sensitivity computation on GPU
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
        cp.ndarray
            Current design variables on GPU, shape (n_vars,)
            
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
            _rho = _rho*self.E_mul
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

            return df*self.E_mul
        
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
        
        if self.is_single_material:
            rho = self.filter.dot(self.desvars)
        else:
            rho = np.copy(self.desvars).reshape(self.n_material, -1).T
            for i in range(self.n_material):
                rho[:, i] = self.filter.dot(rho[:, i])
        
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
        
        self._f = compliance
        if self.is_single_material:
            self._nabla_f = dr.reshape(-1)
        else:
            self._nabla_f = dr.T.reshape(-1)
        self._residual = residual

    def f(self, rho: np.ndarray = None):
        """
        Compute objective function value (compliance).
        
        Parameters
        ----------
        rho : cp.ndarray, optional
            Design variables on GPU for linearization. If None, returns cached value
            
        Returns
        -------
        float
            Compliance value (F^T @ U). Lower is better (stiffer structure).
            
        Notes
        -----
        - If rho provided: returns linearized approximation f(x) + df/dx @ (rho - x)
        - If rho is None: returns cached value from last set_desvars() call
        - Compliance = F^T @ U = U^T @ K @ U (strain energy)
        - All operations performed on GPU
        """
        if rho is None:
            return self._f
        else:
            return self._f + rho.T @ self._nabla_f

    def nabla_f(self, rho: np.ndarray = None):
        """
        Compute objective function gradient (compliance sensitivities).
        
        Parameters
        ----------
        rho : cp.ndarray, optional
            Unused (for interface compatibility)
            
        Returns
        -------
        cp.ndarray
            Gradient of compliance w.r.t. design variables on GPU, shape (n_vars,)
            
        Notes
        -----
        - Uses adjoint method: dC/drho = -U^T @ dK/drho @ U
        - Includes filter adjoint: sens_raw = H^T @ sens_filtered
        - Negative gradient means increasing density reduces compliance (good)
        - All operations performed on GPU
        """
        return self._nabla_f
    
    def g(self, rho=None):
        """
        Compute constraint values (volume fraction violations).
        
        Parameters
        ----------
        rho : cp.ndarray, optional
            Design variables on GPU. If None, uses current desvars
            
        Returns
        -------
        cp.ndarray
            Constraint violations, shape (n_constraints,). Negative = satisfied.
            g[i] = (volume_fraction[i] - actual_volume_fraction[i])
            
        Notes
        -----
        - Constraint satisfied when g <= 0
        - For single material: returns scalar
        - For multi-material: returns array with one constraint per material
        - All operations performed on GPU
        """
        if rho is None:
            vf = (self._nabla_g @ self.desvars.reshape(-1, 1))
            
            return vf.reshape(-1) - self.volume_fraction
        else:
            vf = (self._nabla_g @ rho.reshape(-1, 1))
            
            return vf.reshape(-1) - self.volume_fraction
        
    def nabla_g(self):
        """
        Compute constraint gradients (volume fraction sensitivities).
        
        Returns
        -------
        cp.ndarray
            Gradient of constraints w.r.t. design variables on GPU.
            - Single material: shape (n_vars,)
            - Multi-material: shape (n_materials, n_vars)
            
        Notes
        -----
        - Each row is d(volume_fraction[i])/drho
        - For uniform elements, gradient is constant: 1/volume_total per element
        - Used by optimizers to enforce volume constraints
        """
        return self._nabla_g

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
        
    def FEA(self, thresshold: bool = True):
        if self.desvars is None:
            raise ValueError("Design variables are not initialized. Call init_desvars() or set_desvars() first.")

        if thresshold:
            rho = (self.get_desvars()>0.5).astype(self.dtype) + self.void
            
        if not self.is_single_material:
            rho = rho.reshape(self.n_material, -1).T
            rho = (rho * self.E_mul[np.newaxis, :]).sum(axis=1)
        else:
            rho = rho * self.E_mul
        
        if hasattr(self.FE.solver, 'maxiter'):
            maxiter = self.FE.solver.maxiter + 0
            self.FE.solver.maxiter = maxiter * 4
            
       
        U,residual = self.FE.solve(rho)
        compliance = self.FE.rhs.dot(U)

        
        out = {
            'compliance': compliance,
            'Displacements': U
        }
        
        return out