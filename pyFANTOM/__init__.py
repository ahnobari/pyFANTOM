"""pyFANTOM public package.

This package exposes the public API for the pyFANTOM topology-optimization
framework. Typical usage imports subpackages for a specific backend (CPU or
CUDA) and physics models from :mod:`pyFANTOM.Physics`.

Examples
--------
>>> from pyFANTOM.CPU import StructuredMesh2D, FiniteElement, MinimumCompliance
>>> from pyFANTOM import Physics
"""
