"""CPU backend public API.

Importing from this module gives access to CPU implementations of core
pyFANTOM components (meshes, filters, kernels, solvers, optimizers and
finite-element helpers). Users typically switch to the CPU backend with:

>>> from pyFANTOM.CPU import StructuredMesh2D, FiniteElement, MinimumCompliance
"""

from .Kernels import *
from .Filters import *
from .Mesh import *
from .Optimizers import *
from .Problems import *
from .Solvers import *
from .FiniteElement import *