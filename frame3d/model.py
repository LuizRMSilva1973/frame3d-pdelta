"""Model definition: nodes, elements, supports and loads (SI units: kN, m)."""

from dataclasses import dataclass, field

import numpy as np

from .element import rotation


@dataclass(frozen=True)
class Section:
    A: float
    Iy: float
    Iz: float
    J: float


def rectangular_section(b, h):
    """Rectangle with width b along local y and depth h along local z.

    J uses the Saint-Venant approximation for a solid rectangle
    (Roark: J = a·c³·[1/3 − 0.21·(c/a)·(1 − c⁴/(12·a⁴))], a ≥ c).
    """
    a, c = max(b, h), min(b, h)
    J = a * c**3 * (1 / 3 - 0.21 * (c / a) * (1 - c**4 / (12 * a**4)))
    return Section(A=b * h, Iy=b * h**3 / 12, Iz=h * b**3 / 12, J=J)


@dataclass
class Element:
    i: int
    j: int
    E: float
    G: float
    section: Section
    vecxz: tuple
    w_local: np.ndarray = field(default_factory=lambda: np.zeros(3))


class Model:
    def __init__(self):
        self.nodes = []
        self.elements = []
        self.fixed = {}
        self.nodal_loads = {}

    @property
    def n_nodes(self):
        return len(self.nodes)

    def add_node(self, x, y, z):
        self.nodes.append(np.array([x, y, z], dtype=float))
        return len(self.nodes) - 1

    def add_element(self, i, j, E, G, section, vecxz):
        self.elements.append(Element(i, j, E, G, section, tuple(vecxz)))
        return len(self.elements) - 1

    def fix(self, node, dofs=(1, 1, 1, 1, 1, 1)):
        self.fixed[node] = tuple(bool(d) for d in dofs)

    def add_nodal_load(self, node, fx=0.0, fy=0.0, fz=0.0, mx=0.0, my=0.0, mz=0.0):
        load = self.nodal_loads.setdefault(node, np.zeros(6))
        load += [fx, fy, fz, mx, my, mz]

    def element_geometry(self, e):
        el = self.elements[e]
        return rotation(self.nodes[el.i], self.nodes[el.j], el.vecxz)

    def add_uniform_load_local(self, e, wx=0.0, wy=0.0, wz=0.0):
        self.elements[e].w_local = self.elements[e].w_local + np.array([wx, wy, wz])

    def add_uniform_load_global(self, e, wx=0.0, wy=0.0, wz=0.0):
        """Uniform load per unit length of element, given in global axes."""
        _, r3 = self.element_geometry(e)
        self.add_uniform_load_local(e, *(r3 @ np.array([wx, wy, wz])))
