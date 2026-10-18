"""Global stability indicators.

gamma_z follows ABNT NBR 6118, item 15.5.3:

    gamma_z = 1 / (1 - dM_tot,d / M1_tot,d)

M1_tot,d  first-order overturning moment: sum of the horizontal design forces
          times their height above the base;
dM_tot,d  sum of the vertical design forces times the horizontal displacements
          of their points of application, from the first-order analysis.

The code accepts gamma_z <= 1.1 as a non-sway (fixed-node) structure and
allows it as a moment amplifier up to 1.3.
"""

from dataclasses import dataclass

import numpy as np

from .analysis import load_vector


@dataclass
class GammaZ:
    gamma_z: float
    M1: float
    dM: float


def gamma_z(model, first_order, direction=(1.0, 0.0, 0.0), base_z=None):
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    F = load_vector(model).reshape(-1, 6)
    xyz = np.array(model.nodes)
    if base_z is None:
        base_z = xyz[:, 2].min()
    horizontal = F[:, :3] @ d
    vertical = -F[:, 2]                         # gravity is -Z: downward load positive
    drift = first_order.displacements[:, :3] @ d
    M1 = float(np.sum(horizontal * (xyz[:, 2] - base_z)))
    dM = float(np.sum(vertical * drift))
    if M1 <= 0:
        raise ValueError("no overturning moment from horizontal loads in this direction")
    return GammaZ(gamma_z=1.0 / (1.0 - dM / M1), M1=M1, dM=dM)
