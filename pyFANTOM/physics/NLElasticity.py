from ._physx import Physx
import numpy as np

class NLElasticity(Physx):
    """
    Geometrically nonlinear elasticity for large deformations.
    
    Implements finite strain elasticity with St. Venant-Kirchhoff material model.
    For large displacement problems where linear elasticity is insufficient.
    
    Parameters
    ----------
    E : float, optional
        Young's modulus (default: 1.0)
    nu : float, optional
        Poisson's ratio (default: 1/3)
    thickness : float, optional
        Thickness for 2D elements (default: 1.0)
    type : str, optional
        '2D formulation: 'PlaneStress' or 'PlaneStrain' (default: 'PlaneStress')
        
    Methods
    -------
    K(x0s)
        Initial stiffness matrix (linear)
    KTan(x0s, xs, rho)
        Tangent stiffness matrix at current configuration
        
    Notes
    -----
    - **Use case**: Large deformations, geometric nonlinearity
    - **Solver**: Requires NLFiniteElement with Newton-Raphson
    - **Elements**: Currently supports 4-node quads only
    - **Material**: St. Venant-Kirchhoff (simple hyperelastic)
    - **Limitations**: Not suitable for very large strains (>20%)
    
    Examples
    --------
    >>> from pyFANTOM import NLElasticity
    >>> physics = NLElasticity(E=1.0, nu=0.3)
    >>> # Use with NLFiniteElement and NLUniformStiffnessKernel
    """
    def __init__(self, E=1.0, nu=1./3., thickness=1.0, type='PlaneStress'):
        super().__init__()
        self.E = E
        self.nu = nu
        self.thickness = thickness
        if type == 'PlaneStress':
            self.type = 0
        else:
            self.type = 1

    def K(self, x0s):
        if (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 2):
            return _quadrilateral_element_stiffness(x0s, E=self.E, NU=self.nu, e_type=self.type, thickness=1.0)[0]
        else:
            raise ValueError("Invalid input shape")
        
    def KTan(self, x0s, xs, rho):
        return _quadrilateral_element_tangent_stiffness(x0s, xs, rho, E=self.E, nu=self.nu, thickness=self.thickness)
    
        
    def locals(self, x0s):
        if (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 2):
            return list(_quadrilateral_element_stiffness(x0s, E=self.E, NU=self.nu, e_type=self.type, thickness=1.0)[1:])
        raise UserWarning("locals function inside physics class is being called")
    
    def volume(self, x0s):
        if (x0s.ndim == 2 and x0s.shape[0] == 4 and x0s.shape[1] == 2) or (x0s.ndim == 3 and x0s.shape[1] == 4 and x0s.shape[2] == 2):
            return _quadrilateral_element_area(x0s)
        else:
            raise ValueError("Invalid input shape")
    
    def neumann(self, x0s):
        pass

    def stressCurrent(self):
        return self.stressCurrent

    def stressLastSolved(self):
        return self.stressLastSolved

    # These are bad, but whatever for now
    def set_stressCurrent(self, stress) -> None:
        self.stressCurrent = stress

    def set_stressLastSolved(self, stress) -> None:
        self.stressLastSolved = stress

    
def _quadrilateral_element_area(x0s_in):
    """
    Vectorized function to compute the area of quadrilateral elements.

    Parameters:
        x0s_in (np.array): Array with the nodal positions. 
                          Shape (4,2) for single quad or (n_elements, 4, 2) for batch

    Returns:
        A (float or np.array): Area(s) of the quadrilateral element(s).
    """
    # Handle single element case
    if x0s_in.ndim == 2:
        x0s_in = x0s_in[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    # First triangle: nodes 0, 1, 2
    x0s_tri1 = x0s_in[:, 0:3, :]
    A1 = (
        x0s_tri1[:, 0, 0] * (x0s_tri1[:, 1, 1] - x0s_tri1[:, 2, 1])
        + x0s_tri1[:, 1, 0] * (x0s_tri1[:, 2, 1] - x0s_tri1[:, 0, 1])
        + x0s_tri1[:, 2, 0] * (x0s_tri1[:, 0, 1] - x0s_tri1[:, 1, 1])
    ) / 2
    
    # Second triangle: nodes 0, 2, 3
    x0s_tri2 = x0s_in[:, [0, 2, 3], :]
    A2 = (
        x0s_tri2[:, 0, 0] * (x0s_tri2[:, 1, 1] - x0s_tri2[:, 2, 1])
        + x0s_tri2[:, 1, 0] * (x0s_tri2[:, 2, 1] - x0s_tri2[:, 0, 1])
        + x0s_tri2[:, 2, 0] * (x0s_tri2[:, 0, 1] - x0s_tri2[:, 1, 1])
    ) / 2
    
    A = A1 + A2
    
    if single_element:
        return A[0]
    else:
        return A
    
def _quadrilateral_element_stiffness(x0s, E=1.0, NU=0.33, e_type=0, thickness=1.0):
    # This was left over from linear elasticity. Tangent stiffness matrix is the same at 0 displacement.
    """
    Vectorized function to compute the stiffness matrix for quadrilateral elements.

    Parameters:
        x0s (np.array): Array with the nodal positions. 
                       Shape (4,2) for single quad or (n_elements, 4, 2) for batch
        E (float or np.array): Young's modulus
        NU (float or np.array): Poisson's ratio  
        e_type (int): 0 for Plane Stress, 1 for Plane Strain
        t (float or np.array): Thickness of the element

    Returns:
        k (np.array): Stiffness matrices. Shape (8,8) or (n_elements, 8, 8)
        D (np.array): Constitutive matrices. Shape (3,3) or (n_elements, 3, 3)
        B_global (np.array): B-Operators. Shape (3,8) or (n_elements, 3, 8)
    """
    # Handle single element case
    if x0s.ndim == 2:
        x0s = x0s[np.newaxis, ...]
        single_element = True
    else:
        single_element = False
    
    n_elements = x0s.shape[0]
    
    # Ensure material properties are arrays
    E = np.atleast_1d(E)
    NU = np.atleast_1d(NU)
    t = np.atleast_1d(thickness)
    
    # Broadcast to match number of elements
    if E.size == 1:
        E = np.full(n_elements, E[0])
    if NU.size == 1:
        NU = np.full(n_elements, NU[0])
    if t.size == 1:
        t = np.full(n_elements, t[0])
    
    # Check areas
    A = _quadrilateral_element_area(x0s)
    if single_element:
        A = np.array([A])
    
    if np.any(A < 0):
        negative_indices = np.where(A < 0)[0]
        raise Exception(f"Node Order Is Not Correct for elements: {negative_indices}")

    # Gaussian quadrature points and weights
    gauss_points = np.array([-0.577350269189626, 0.577350269189626])
    weights = np.ones(2)

    # Create constitutive matrices for all elements
    D = np.zeros((n_elements, 3, 3))
    
    if e_type == 0:  # Plane Stress
        coeff = E / (1 - NU * NU)
        D[:, 0, 0] = coeff
        D[:, 1, 1] = coeff
        D[:, 0, 1] = coeff * NU
        D[:, 1, 0] = coeff * NU
        D[:, 2, 2] = coeff * (1 - NU) / 2
    else:  # Plane Strain
        coeff = E / ((1 + NU) * (1 - 2 * NU))
        D[:, 0, 0] = coeff * (1 - NU)
        D[:, 1, 1] = coeff * (1 - NU)
        D[:, 0, 1] = coeff * NU
        D[:, 1, 0] = coeff * NU
        D[:, 2, 2] = coeff * (1 - 2 * NU) / 2

    # Initialize stiffness matrices and global B matrices
    k = np.zeros((n_elements, 8, 8))
    B_global = np.zeros((n_elements, 3, 8))

    # Gaussian quadrature integration
    for i in range(2):
        r = gauss_points[i]
        wi = weights[i]
        
        for j in range(2):
            s = gauss_points[j]
            wj = weights[j]

            # Shape function derivatives in natural coordinates
            dNdr = _quadrilateral_element_intPoly(coord=np.array([r, s]), order='linear', derivative='r')
            dNds = _quadrilateral_element_intPoly(coord=np.array([r, s]), order='linear', derivative='s')

            # Compute Jacobian matrices for all elements
            J = np.zeros((n_elements, 2, 2))
            
            for m in range(4):
                J[:, 0, 0] += dNdr[m] * x0s[:, m, 0]
                J[:, 0, 1] += dNdr[m] * x0s[:, m, 1]
                J[:, 1, 0] += dNds[m] * x0s[:, m, 0]
                J[:, 1, 1] += dNds[m] * x0s[:, m, 1]

            # Determinant and inverse of Jacobian
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # Shape function derivatives in physical coordinates
            dNdx = np.zeros((n_elements, 4))
            dNdy = np.zeros((n_elements, 4))
            
            for m in range(4):
                dNdx[:, m] = invJ[:, 0, 0] * dNdr[m] + invJ[:, 0, 1] * dNds[m]
                dNdy[:, m] = invJ[:, 1, 0] * dNdr[m] + invJ[:, 1, 1] * dNds[m]

            # Construct B matrices for all elements
            B = np.zeros((n_elements, 3, 8))
            for m in range(4):
                B[:, 0, m*2] = dNdx[:, m]      # dN/dx for u
                B[:, 1, m*2+1] = dNdy[:, m]    # dN/dy for v
                B[:, 2, m*2] = dNdy[:, m]      # dN/dy for u (shear)
                B[:, 2, m*2+1] = dNdx[:, m]    # dN/dx for v (shear)

            # Integration coefficient
            coeff = t * wi * wj * detJ

            # Add contribution to stiffness matrix: k += B^T @ D @ B * coeff
            # Using einsum for efficient batch computation
            k += np.einsum('i,ijk,ikl,ilm->ijm', coeff,
                          np.transpose(B, (0, 2, 1)),  # B^T
                          D,
                          B)
            
            # Accumulate B for global B matrix (averaged)
            B_global += B / 4

    # Return single element results if input was single element
    if single_element:
        return k[0], D[0], B_global[0]
    else:
        return k, D, B_global
    
def _quadrilateral_element_tangent_stiffness(x0s, xs, rho, E=1.0, nu=0.33, thickness=1.0):
    # Handle single element case
    if x0s.ndim == 2:
        raise UserWarning("single element case does not work with nonlinear analysis")
    else:
        single_element = False
    
    n_elements = x0s.shape[0]

    # Ensure material properties are arrays
    E = np.atleast_1d(E)
    nu = np.atleast_1d(nu)
    thickness = np.atleast_1d(thickness)

    # Nodal Displacement
    t0u = xs-x0s
   
    # Broadcast to match number of elements
    if E.size == 1:
        E = np.full(n_elements, E[0])
    if nu.size == 1:
        nu = np.full(n_elements, nu[0])
    if thickness.size == 1:
        thickness = np.full(n_elements, thickness[0])
    
    # Check areas
    A = _quadrilateral_element_area(x0s)
    if single_element:
        A = np.array([A])
    
    if np.any(A < 0):
        negative_indices = np.where(A < 0)[0]
        raise Exception(f"Node Order Is Not Correct for elements: {negative_indices}")

    # Gaussian quadrature points and weights
    gauss_points = np.array([-0.577350269189626, 0.577350269189626])
    weights = np.ones(2)

    # Create constitutive matrices for all elements
    D = np.zeros((n_elements, 3, 3))

    # Initialize stiffness matrices and global B matrices
    KTan = np.zeros((n_elements, 8, 8))
    t0F = np.zeros((n_elements, 8)) # global internal force vector
    dR_Drho = np.zeros((n_elements, 8))

    # Gaussian quadrature integration
    for i in range(2):
        r = gauss_points[i]
        wi = weights[i]
        
        for j in range(2):
            s = gauss_points[j]
            wj = weights[j]

            # Shape function derivatives in natural coordinates
            dNdr = _quadrilateral_element_intPoly(coord=np.array([r, s]), order='linear', derivative='r')
            dNds = _quadrilateral_element_intPoly(coord=np.array([r, s]), order='linear', derivative='s')

            # This is hardcoded for a 4-noded quad element
            dN_scalar = np.array([dNdr, dNds]).transpose()
            dN_vector = np.zeros((4, 8))
            dN_vector[(0,1), ::2] = dN_scalar.transpose()
            dN_vector[(2,3), 1::2] = dN_scalar.transpose()

            # Compute Jacobian matrices for all elements
            J = np.zeros((n_elements, 2, 2))
            
            for m in range(4):
                J[:, 0, 0] += dNdr[m] * x0s[:, m, 0]
                J[:, 0, 1] += dNdr[m] * x0s[:, m, 1]
                J[:, 1, 0] += dNds[m] * x0s[:, m, 0]
                J[:, 1, 1] += dNds[m] * x0s[:, m, 1]

            # Determinant and inverse of Jacobian
            detJ = np.linalg.det(J)
            invJ = np.linalg.inv(J)

            # Shape function derivatives in physical coordinates
            dNdx = np.zeros((n_elements, 4))
            dNdy = np.zeros((n_elements, 4))
            
            for m in range(4):
                dNdx[:, m] = invJ[:, 0, 0] * dNdr[m] + invJ[:, 0, 1] * dNds[m]
                dNdy[:, m] = invJ[:, 1, 0] * dNdr[m] + invJ[:, 1, 1] * dNds[m]

            # Gradient operator
            G = np.einsum('ijk,kl->ijl', 
                          np.block([[invJ, np.zeros((n_elements, 2, 2))],
                                    [np.zeros((n_elements, 2, 2)), invJ]]),
                          dN_vector)
            
            dispGrad = np.einsum('ijk,ikl->ij', G, np.transpose(t0u.reshape(n_elements, 1, -1), (0, 2, 1)))

            # Infinitestimal Strain
            dispGradTensor = np.zeros((n_elements, 3, 3))
            dispGradTensor[:, 0:2, 0:2] = dispGrad.reshape(n_elements, 2, 2)

            F = np.transpose(np.array(([dispGrad[:,0]+1], [dispGrad[:,1]], np.zeros((1,n_elements)),
                                       [dispGrad[:,2]], [1 + dispGrad[:,3]], np.zeros((1, n_elements)),
                                       np.zeros((1, n_elements)), np.zeros((1, n_elements)), np.ones((1, n_elements)))),
                                       (2, 1, 0)).reshape(n_elements,3,3)
            

            B0F = np.array([
                    [F[:, 0, 0], np.zeros(n_elements), F[:, 1, 0], np.zeros(n_elements)],
                    [np.zeros(n_elements), F[:, 0, 1], np.zeros(n_elements), F[:, 1, 1]],
                    [F[:, 0, 1], F[:, 0, 0], F[:, 1, 1], F[:, 1, 0]]
                ]).transpose(2, 0, 1)

            B = np.einsum('ijk,ikl->ijl',
                          B0F, G)

            C = np.einsum('ijk,ikl->ijl',
                          np.transpose(F, (0, 2, 1)),
                          F)
            
            GLstrain = 0.5 * (C - np.tile(np.eye(3), (n_elements,1,1)))

            S_vec_wo_rho = np.zeros((n_elements, 3))
            S_mat = np.zeros((n_elements, 4, 4))
            D = np.zeros((n_elements, 3, 3))
            lamb = nu * E / (1 - nu**2)
            mu = E / 2 / (1 + nu)

            # Compute stresses for all elements at once
            stressGL, D_tensor = StVK(lamb, mu, GLstrain, rho=rho)

            S_vec = np.stack([stressGL[:, 0, 0], stressGL[:, 1, 1], stressGL[:, 0, 1]], axis=1)

            S_mat = np.zeros((n_elements, 4, 4))
            S_mat[:, 0, 0] = stressGL[:, 0, 0]
            S_mat[:, 0, 1] = stressGL[:, 0, 1]
            S_mat[:, 1, 0] = stressGL[:, 1, 0]
            S_mat[:, 1, 1] = stressGL[:, 1, 1]
            S_mat[:, 2, 2] = stressGL[:, 0, 0]
            S_mat[:, 2, 3] = stressGL[:, 0, 1]
            S_mat[:, 3, 2] = stressGL[:, 1, 0]
            S_mat[:, 3, 3] = stressGL[:, 1, 1]

            D_voigt = np.zeros((n_elements, 3, 3))
            D_voigt[:, 0, 0] = D_tensor[:, 0, 0, 0, 0]
            D_voigt[:, 0, 1] = D_tensor[:, 0, 0, 1, 1]
            D_voigt[:, 0, 2] = D_tensor[:, 0, 0, 0, 1]
            D_voigt[:, 1, 0] = D_tensor[:, 1, 1, 0, 0]
            D_voigt[:, 1, 1] = D_tensor[:, 1, 1, 1, 1]
            D_voigt[:, 1, 2] = D_tensor[:, 1, 1, 0, 1]
            D_voigt[:, 2, 0] = D_tensor[:, 0, 1, 0, 0]
            D_voigt[:, 2, 1] = D_tensor[:, 0, 1, 1, 1]
            D_voigt[:, 2, 2] = D_tensor[:, 0, 1, 0, 1]

            # This is for gradient calculation
            stressGL_wo_rho, _ = StVK(lamb, mu, GLstrain)
            S_vec_wo_rho = np.stack([stressGL_wo_rho[:, 0, 0], stressGL_wo_rho[:, 1, 1], stressGL_wo_rho[:, 0, 1]], axis=1)
            
            # Integration coefficient
            coeff = thickness * wi * wj * detJ

            # Compute internal forces
            t0F += np.einsum('i,ijk,ik->ij',
                             coeff,
                             np.transpose(B, (0, 2, 1)),
                             S_vec)

            # Compute tangent stiffness matrix
            KTan += (np.einsum('i,ijk,ikl,ilm->ijm',
                              coeff,
                              np.transpose(B, (0, 2, 1)),
                              D_voigt, 
                              B)
                     +
                     np.einsum('i,ijk,ikl,ilm->ijm',
                              coeff,
                              np.transpose(G, (0, 2, 1)),
                              S_mat,
                              G)) / rho[:, None, None] # This gets reapplied during the solving process

            dR_Drho += -(np.einsum('i,ij->ij',
                            coeff,
                            (np.einsum('ijk,ik->ij', 
                                    np.transpose(B, (0,2,1)), 
                                    S_vec_wo_rho) 
                            )
                            )
                        )

    # Return single element results if input was single element
    if single_element:
        raise UserWarning("single element case does not work with nonlinear analysis")
    else:
        return KTan, t0F, dR_Drho
    

def _quadrilateral_element_intPoly(coord, order, derivative):
    if order == 'linear':
        if derivative == 'r':
            N = np.array([
                -0.25 * (1 - coord[1]),
                0.25 * (1 - coord[1]),
                0.25 * (1 + coord[1]),
                -0.25 * (1 + coord[1])
            ])
            return N

        elif derivative == 's':
            N = np.array([
                -0.25 * (1 - coord[0]),
                -0.25 * (1 + coord[0]),
                0.25 * (1 + coord[0]),
                0.25 * (1 - coord[0])
            ])
            return N

        elif derivative == None:
            N = np.array([
                0.25 * (1 - coord[0]) * (1 - coord[1]),
                0.25 * (1 + coord[0]) * (1 - coord[1]),
                0.25 * (1 + coord[0]) * (1 + coord[1]),
                0.25 * (1 - coord[0]) * (1 + coord[1])
            ])
            return N
        else:
            raise ValueError("Invalid derivative direction. Use 'r', 's', None.")
    else:
        raise ValueError("Invalid order. Only 'linear' is supported.")

def StVK(lamb, mu, strain, rho=None):
    # Ensure inputs are arrays
    lamb = np.asarray(lamb)
    mu = np.asarray(mu)
    strain = np.asarray(strain)
    
    # Apply density scaling if provided
    if rho is not None:
        rho = np.asarray(rho)
        lamb = lamb * rho
        mu = mu * rho
    
    # Identity tensor
    I = np.eye(3, 3)
    
    # II tensor
    II = np.einsum('ij,kl->ijkl', I, I)
    
    # II_sym tensor 
    II_sym = 0.5 * (np.einsum('ik,jl->ijkl', I, I) + np.einsum('il,jk->ijkl', I, I))
    
    # Compute trace for each element: shape (n_elements,)
    trace_strain = np.trace(strain, axis1=1, axis2=2)
    
    # Stress tensor:
    stress = lamb[:, None, None] * trace_strain[:, None, None] * I[None, :, :] + \
             2 * mu[:, None, None] * strain
    
    # Constitutive tensor: 
    D_tensor = lamb[:, None, None, None, None] * II[None, :, :, :, :] + \
        2 * mu[:, None, None, None, None] * II_sym[None, :, :, :, :]
    
    return stress, D_tensor