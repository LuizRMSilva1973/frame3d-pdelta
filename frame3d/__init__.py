"""frame3d: linear and P-Delta analysis of 3D frames, with gamma_z (NBR 6118) and OpenSees cross-checks."""

from .analysis import Result, load_vector, solve_linear, solve_pdelta
from .model import Model, Section, rectangular_section
from .stability import GammaZ, gamma_z

__all__ = [
    "Model", "Section", "rectangular_section",
    "Result", "load_vector", "solve_linear", "solve_pdelta",
    "GammaZ", "gamma_z",
]
