"""Physics models exported by pyFANTOM.

This module exposes the built-in physics model classes. Each physics model
implements the :class:`pyFANTOM.physics._physx.Physx` interface and provides
element-level stiffness/conductivity computations used by assembly kernels.

Available models
- LinearElasticity: small-strain linear elasticity
- NLElasticity: nonlinear elasticity (St. Venant–Kirchhoff style)
- SteadyHeatTransfer: steady-state thermal conduction
- LinearSymbolic: user-defined symbolic bilinear forms
"""

from .physics._physx import Physx
from .physics.LinearElasticity import LinearElasticity
from .physics.NLElasticity import NLElasticity
from .physics.SteadyHeatTransfer import SteadyHeatTransfer
from .physics.LinearSymbolic import LinearSymbolic