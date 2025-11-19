from .._problem import Problem
from ...FiniteElement.CPU.FiniteElement import FiniteElement
from ...geom.CPU._mesh import StructuredMesh
from ...geom.CPU._filters import StructuredFilter2D, StructuredFilter3D, GeneralFilter
from ...core.CPU._ops import FEA_locals_node_basis_parallel, FEA_locals_node_basis_parallel_flat, FEA_locals_node_basis_parallel_full
from typing import Union, List
from scipy.spatial import KDTree
import numpy as np

class WeightDistributionMinimumCompliance(Problem):
    """
    Minimum compliance with weight distribution constraint.
    
    Extends MinimumCompliance with an additional constraint on the centroid location
    of the material distribution. Useful for balancing structures or controlling
    center of mass for dynamic applications.
    
    Parameters
    ----------
    FE : FiniteElement
        Finite element analysis engine with mesh, kernel, and solver
    filter : StructuredFilter2D, StructuredFilter3D, or GeneralFilter
        Density filter for ensuring minimum feature sizes
    target_centroid : List[float]
        Target centroid location in physical coordinates, shape (2,) for 2D or (3,) for 3D
    maximum_distance : float
        Maximum allowed distance from target centroid (constraint: ||centroid - target|| <= maximum_distance)
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
        
    Attributes
    ----------
    is_single_material : bool
        True for single material, False for multi-material optimization
    n_material : int
        Number of materials
    iteration : int
        Current iteration count
    target_centroid : ndarray
        Target centroid location
    maximum_distance : float
        Maximum allowed distance from target
        
    Methods
    -------
    f()
        Compute compliance objective
    nabla_f()
        Compute compliance sensitivities
    g()
        Compute constraint values (volume + centroid distance)
    nabla_g()
        Compute constraint gradients
    get_desvars()
        Get current design variables
    set_desvars(rho)
        Set design variables and trigger FEA solve
    visualize_solution(**kwargs)
        Plot optimized design
        
    Notes
    -----
    **Constraints:**
    - Volume constraint: Same as MinimumCompliance
    - Centroid constraint: ||centroid(rho) - target_centroid|| <= maximum_distance
    - Constraints are coupled (not independent), so OC optimizer cannot be used
    
    **Centroid Computation:**
    - Centroid = sum(rho[i] * volume[i] * centroid[i]) / sum(rho[i] * volume[i])
    - Weighted by element density and volume
    
    **Use Cases:**
    - Balancing structures for dynamic loads
    - Controlling center of mass
    - Symmetric designs with off-center loading
    - Multi-material designs with weight distribution requirements
    
    Examples
    --------
    >>> from pyFANTOM.CPU import *
    >>> # Setup FEA
    >>> mesh = StructuredMesh2D(nx=128, ny=64, lx=2.0, ly=1.0)
    >>> kernel = StructuredStiffnessKernel(mesh=mesh)
    >>> solver = CHOLMOD(kernel=kernel)
    >>> FE = FiniteElement(mesh=mesh, kernel=kernel, solver=solver)
    >>> 
    >>> # Apply BCs and loads
    >>> FE.add_dirichlet_boundary_condition(node_ids=fixed_nodes, rhs=0)
    >>> FE.add_point_forces(node_ids=load_nodes, forces=np.array([[0, -1.0]]))
    >>> 
    >>> # Define optimization problem with centroid constraint
    >>> filter = StructuredFilter2D(mesh=mesh, r_min=1.5)
    >>> problem = WeightDistributionMinimumCompliance(
    >>>     FE=FE, filter=filter,
    >>>     target_centroid=[1.0, 0.5],  # Center of 2D domain
    >>>     maximum_distance=0.1,  # 10% of domain size
    >>>     penalty=3.0, volume_fraction=[0.3]
    >>> )
    """
    def __init__(self,
                 FE: FiniteElement,
                 filter: Union[StructuredFilter2D, StructuredFilter3D, GeneralFilter],
                 target_centroid: List[float],
                 maximum_distance: float,
                 E_mul: list[float] = [1.0],
                 void: float = 1e-6,
                 penalty: float = 3.0,
                 volume_fraction: list[float] = [0.25],
                 penalty_schedule: list[float] = None,
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
            self._nabla_g = np.tile(self._nabla_g, self.num_vars).reshape(1, -1)
        elif self._nabla_g.shape[0] == 1 and not self.is_single_material:
            self._nabla_g = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            for i in range(self.n_material):
                self._nabla_g[i, i*self.nel:(i+1)*self.nel] = self.FE.mesh.As[0]/self.FE.mesh.volume
        elif self._nabla_g.shape[0] != self.num_vars and not self.is_single_material:
            self._nabla_g = np.zeros((self.n_material, self.num_vars), dtype=self.dtype)
            for i in range(self.n_material):
                self._nabla_g[i, i*self.nel:(i+1)*self.nel] = self.FE.mesh.As/self.FE.mesh.volume
        self._nabla_g = np.concatenate([self._nabla_g, np.zeros((1, self.num_vars), dtype=self.dtype)], axis=0)
        
        self.is_3D = self.FE.mesh.nodes.shape[1] == 3
        
        if self.is_3D and len(target_centroid) != 3:
            raise ValueError("Target centroid must be a 3D point.")
        if not self.is_3D and len(target_centroid) != 2:
            raise ValueError("Target centroid must be a 2D point.")
        
        self.target_centroid = np.array(target_centroid, dtype=self.dtype)
        self.maximum_distance = maximum_distance
            
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
            Number of constraints:
            - Single material: 2 (volume + centroid distance)
            - Multi-material: n_materials + 1 (volume per material + centroid distance)
        """
        return 2 if self.is_single_material else len(self.E_mul) + 1
    
    def is_independent(self):
        """
        Check if constraints are independent (required for OC optimizer).
        
        Returns
        -------
        bool
            Always False. Centroid constraint couples all design variables, making
            constraints non-independent. Use MMA or PGD optimizer instead of OC.
        """
        return False
    
    def constraint_map(self):
        """
        Return mapping of constraints to design variables.
        
        Returns
        -------
        int or ndarray
            - Single material: 1 (scalar, all variables affect both constraints)
            - Multi-material: array of shape (n_materials + 1, n_vars) where:
              - Rows 0 to n_materials-1: volume constraints per material
              - Last row: centroid constraint (all variables = 1)
              
        Notes
        -----
        Used by optimizers to identify which design variables affect each constraint.
        The centroid constraint depends on all design variables, making constraints coupled.
        """
        if self.is_single_material:
            return 1
        else:
            mapping = np.zeros((self.n_material + 1, self.num_vars), dtype=self.dtype)
            
            for i in range(self.n_material):
                mapping[i, i*self.nel:(i+1)*self.nel] = 1
                
            mapping[-1, :] = 1
                
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
        displays combined material distribution.
        """
        rho = self.get_desvars()
        if self.n_material > 1:
            rho = rho.reshape(self.n_material, -1).T
        self.FE.visualize_density(rho, **kwargs)
    
    def init_desvars(self):
        """
        Initialize design variables with random smoothed distribution.
        
        Uses random initialization followed by triple filtering to create a smooth
        initial guess. This helps avoid getting stuck in poor local minima that
        may violate the centroid constraint.
        
        Notes
        -----
        - Random initialization: uniform distribution in [min(0, 2*vf-1), 1.0]
        - Triple filtering: applies filter 3 times for smoothness
        - Normalized to match volume fraction constraint
        - Resets iteration counter to 0
        - Triggers _compute() to evaluate objective and constraints
        """
        # random initialization prevents getting stuck in bad local minima
        np.random.seed(0)
        self.desvars = np.random.uniform(low=min(0,2*self.volume_fraction-1), high=1.0, size=self.num_vars)
        # filter 3x to smooth initial guess
        self.desvars = self.filter.dot(self.filter.dot(self.filter.dot(self.desvars))).reshape(-1)
        self.desvars *= self.volume_fraction / self.desvars.mean()
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
        - Computes centroid and centroid constraint gradient
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
                _rho = (np.tanh(self.beta * self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            else:
                _rho = rho
            _rho = _rho**pen
            _rho = np.clip(_rho, self.void, 1.0)
            _rho = _rho*self.E_mul
            _rho = np.clip(_rho, self.void, None)
            
            return _rho
            
        else:
            if self.heavyside:
                _rho = (np.tanh(self.beta * self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
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
                rho_heavy = (np.tanh(self.beta * self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
                df = pen * rho_heavy ** (pen - 1) * self.beta * (1 - np.tanh(self.beta * (rho-self.eta))**2) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
            else:
                df = pen * rho ** (pen - 1)

            return df*self.E_mul
        
        else:
            if self.heavyside:
                rho_heavy = (np.tanh(self.beta * self.eta) + np.tanh(self.beta * (rho-self.eta))) / (np.tanh(self.beta*self.eta) + np.tanh(self.beta * (1-self.eta)))
                
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
        
        vf = self._nabla_g[:-1].squeeze() @ self.desvars.reshape(-1, 1)
        V = vf * self.FE.mesh.volume
        
        if self.is_single_material:
            mul = self.FE.mesh.centroids * self._nabla_g[0][:, None] * self.FE.mesh.volume
            cmm = (mul * self.desvars.reshape(-1, 1)).sum(0)
        else:
            mul = (self.FE.mesh.centroids * self._nabla_g[0, :self.nel, None])[None] * self.FE.mesh.volume
            cmm = (mul * self.desvars.reshape(self.n_material, -1, 1)).sum(0).sum(0)

        cm = cmm / V.sum()
        distance = np.square(cm - self.target_centroid).sum()
        
        self._g = np.concatenate([
            vf.reshape(-1) - self.volume_fraction,
            np.array([distance - self.maximum_distance**2], dtype=self.dtype)
        ]).reshape(-1)

        nabla_centroid = (2 * (cm - self.target_centroid)[None] * (V.sum() * mul.squeeze() - cmm[None] * self._nabla_g[0][:self.nel, None] * self.FE.mesh.volume) / (V.sum()**2)).sum(1)
        if self.is_single_material:
            self._nabla_g[-1] = nabla_centroid
        else:
            for i in range(self.n_material):
                self._nabla_g[-1, i*self.nel:(i+1)*self.nel] = nabla_centroid
                
        self._cm = cm

    def f(self, rho: np.ndarray = None):
        """
        Compute objective function value (compliance).
        
        Parameters
        ----------
        rho : ndarray, optional
            Design variables for linearization. If None, returns cached value
            
        Returns
        -------
        float
            Compliance value (F^T @ U). Lower is better (stiffer structure).
            
        Notes
        -----
        - If rho provided: returns linearized approximation f(x) + df/dx @ (rho - x)
        - If rho is None: returns cached value from last set_desvars() call
        - Compliance = F^T @ U = U^T @ K @ U (strain energy)
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
        rho : ndarray, optional
            Unused (for interface compatibility)
            
        Returns
        -------
        ndarray
            Gradient of compliance w.r.t. design variables, shape (n_vars,)
            
        Notes
        -----
        - Uses adjoint method: dC/drho = -U^T @ dK/drho @ U
        - Includes filter adjoint: sens_raw = H^T @ sens_filtered
        - Negative gradient means increasing density reduces compliance (good)
        """
        return self._nabla_f
    
    def g(self, rho=None):
        """
        Compute constraint values (volume + centroid distance violations).
        
        Parameters
        ----------
        rho : ndarray, optional
            Design variables. If None, uses current desvars
            
        Returns
        -------
        ndarray
            Constraint violations, shape (n_constraints,). Negative = satisfied.
            - First n_materials constraints: volume fraction violations
            - Last constraint: centroid distance violation (distance^2 - maximum_distance^2)
            
        Notes
        -----
        - Constraint satisfied when g <= 0
        - For single material: returns [volume_violation, centroid_violation]
        - For multi-material: returns [vol_viol_1, ..., vol_viol_n, centroid_violation]
        - If rho provided, uses linearized approximation
        """
        if rho is None:
            return self._g
        else:
            # local linear approximation
            return self._g + (self._nabla_g @ (rho - self.desvars).reshape(-1, 1)).squeeze()
        
    def nabla_g(self):
        """
        Compute constraint gradients (volume + centroid sensitivities).
        
        Returns
        -------
        ndarray
            Gradient of constraints w.r.t. design variables.
            - Single material: shape (2, n_vars) - [volume_grad, centroid_grad]
            - Multi-material: shape (n_materials + 1, n_vars) - [vol_grad_1, ..., vol_grad_n, centroid_grad]
            
        Notes
        -----
        - First rows: d(volume_fraction[i])/drho (constant for uniform elements)
        - Last row: d(centroid_distance^2)/drho (depends on element positions)
        - Centroid gradient accounts for weighted average of element centroids
        - Used by optimizers to enforce constraints
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

        if residual > 1e-5:
            print(f"Solver residual is above 1e-5 ({residual:.4e}). Consider higher iterations (rerun this function and more iteration from prior solve will be applied).")
            
        if isinstance(self.FE.mesh, StructuredMesh):
            strain, stress, strain_energy = FEA_locals_node_basis_parallel(self.FE.mesh.K_single,
                                                                            self.FE.mesh.locals[0],
                                                                            self.FE.mesh.locals[1],
                                                                            self.FE.kernel.elements_flat,
                                                                            rho.shape[0],
                                                                            rho,
                                                                            U,
                                                                            self.FE.mesh.dof,
                                                                            self.FE.mesh.elements_size,
                                                                            self.FE.mesh.locals[1].shape[0])
        elif self.FE.mesh.is_uniform:
            strain, stress, strain_energy = FEA_locals_node_basis_parallel_full(self.FE.mesh.Ks,
                                                                                    self.FE.mesh.locals[0],
                                                                                    self.FE.mesh.locals[1],
                                                                                    self.FE.kernel.elements_flat,
                                                                                    rho.shape[0],
                                                                                    rho,
                                                                                    U,
                                                                                    self.FE.mesh.dof,
                                                                                    self.FE.mesh.elements.shape[1],
                                                                                    self.FE.mesh.locals[1].shape[1])
        else:
            B_size = (self.FE.mesh.locals_ptr[1][1]-self.FE.mesh.locals_ptr[1][0])//((self.FE.mesh.elements_ptr[1]-self.FE.mesh.elements_ptr[0])*self.FE.mesh.dof)
            strain, stress, strain_energy = FEA_locals_node_basis_parallel_flat(self.FE.mesh.K_flat,
                                                                                    self.FE.mesh.locals_flat[0],
                                                                                    self.FE.mesh.locals_flat[1],
                                                                                    self.FE.kernel.elements_flat,
                                                                                    self.FE.mesh.elements_ptr,
                                                                                    self.FE.mesh.K_ptr,
                                                                                    self.FE.mesh.locals_ptr[1],
                                                                                    self.FE.mesh.locals_ptr[0],
                                                                                    rho.shape[0],
                                                                                    rho,
                                                                                    U,
                                                                                    self.FE.mesh.dof,
                                                                                    B_size)
            
        if self.FE.mesh.nodes.shape[1] == 2:
            von_mises = np.sqrt(stress[:, 0] ** 2 + stress[:, 1] ** 2 - stress[:, 0] * stress[:, 1] + 3 * stress[:, 2] ** 2)
        else:
            von_mises = np.sqrt(0.5 * ((stress[:, 0] - stress[:, 1]) ** 2 + (stress[:, 1] - stress[:, 2]) ** 2 + (stress[:, 2] - stress[:, 0]) ** 2 + 6 * (stress[:, 3] ** 2 + stress[:, 4] ** 2 + stress[:, 5] ** 2)))

        
        out = {
            'strain': strain,
            'stress': stress,
            'strain_energy': strain_energy,
            'von_mises': von_mises,
            'compliance': compliance,
            'Displacements': U
        }
        
        return out
    
    def visualize_field(self, field, ax=None, thresshold=True, **kwargs):
        if thresshold:
            rho = (self.desvars > 0.5).astype(self.dtype)
        else:
            rho = None
            
        if not self.is_single_material and rho is not None:
            rho = rho.reshape(self.n_material, -1).T
            rho = (rho).sum(axis=1)>0

        self.FE.visualize_field(field, ax=ax, rho=rho)