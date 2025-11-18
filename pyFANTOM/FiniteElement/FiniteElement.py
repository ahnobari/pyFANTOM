from ..geom.commons._mesh import Mesh, StructuredMesh


class FiniteElement:
    """Abstract base for finite-element wrappers.

    Subclasses implement backend-specific behaviour (CPU/CUDA). The base
    class defines the common interface used by problems and optimizers.
    """
    def __init__(self):
        """Initialize finite-element wrapper state.

        Subclasses may extend the initializer to accept mesh, kernel or solver
        objects.
        """
        pass
    
    def add_dirichlet_boundary_condition(self, condition):
        """Add Dirichlet (essential) boundary condition.

        Parameters
        - condition: backend-specific description of constrained DOFs.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def add_neumann_boundary_condition(self, condition):
        """Add Neumann (natural) boundary condition.

        Parameters
        - condition: backend-specific description of traction/flux loads.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def add_point_forces(self, forces):
        """Apply point forces to nodes.

        Parameters
        - forces: array-like of force vectors or a mapping node->force.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset_forces(self):
        """Clear all applied loads for the current problem instance."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def reset_dirichlet_boundary_conditions(self):
        """Remove all Dirichlet boundary conditions previously added."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def visualize_problem(self, **kwargs):
        """Visualize nodes, elements, boundary conditions and loads.

        Keyword arguments are backend-specific and forwarded to visualization
        helpers.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def visualize_density(self, **kwargs):
        """Visualize a density field (design variable) on the mesh."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def visualize_field(self, **kwargs):
        """Visualize an arbitrary scalar or vector field defined on nodes or elements."""
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def solve(self, **kwargs):
        """Run the linear (or non-linear) finite element solve.

        Returns solver-specific results (e.g. displacement vector and residual).
        """
        raise NotImplementedError("This method should be implemented by subclasses.")