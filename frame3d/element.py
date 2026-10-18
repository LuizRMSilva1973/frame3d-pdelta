"""3D Euler-Bernoulli beam-column element.

Degrees of freedom per node, in this order: ux, uy, uz, rx, ry, rz.
Local axes follow the OpenSees convention for ``geomTransf``:
x runs from node i to node j, ``vecxz`` lies in the local x-z plane,
y = vecxz × x and z = x × y.
"""

import numpy as np


def rotation(xi, xj, vecxz):
    """Return the element length and the 3x3 rotation matrix (rows = local axes in global coordinates)."""
    d = np.asarray(xj, dtype=float) - np.asarray(xi, dtype=float)
    length = np.linalg.norm(d)
    if length == 0.0:
        raise ValueError("zero-length element")
    ex = d / length
    ey = np.cross(np.asarray(vecxz, dtype=float), ex)
    ny = np.linalg.norm(ey)
    if ny < 1e-12:
        raise ValueError("vecxz is parallel to the element axis")
    ey /= ny
    ez = np.cross(ex, ey)
    return length, np.vstack([ex, ey, ez])


def transformation(r3):
    """12x12 transformation matrix: u_local = T @ u_global."""
    t = np.zeros((12, 12))
    for k in range(4):
        t[3 * k:3 * k + 3, 3 * k:3 * k + 3] = r3
    return t


def local_stiffness(E, G, A, Iy, Iz, J, L):
    """Linear elastic stiffness in local coordinates (Iz: bending in the local x-y plane)."""
    k = np.zeros((12, 12))
    ea, gj = E * A / L, G * J / L
    k[0, 0] = k[6, 6] = ea
    k[0, 6] = -ea
    k[3, 3] = k[9, 9] = gj
    k[3, 9] = -gj

    # bending in the x-y plane (displacement v, rotation rz)
    a, b, c, d = 12 * E * Iz / L**3, 6 * E * Iz / L**2, 4 * E * Iz / L, 2 * E * Iz / L
    k[1, 1] = k[7, 7] = a
    k[1, 7] = -a
    k[1, 5] = k[1, 11] = b
    k[5, 7] = k[7, 11] = -b
    k[5, 5] = k[11, 11] = c
    k[5, 11] = d

    # bending in the x-z plane (displacement w, rotation ry = -dw/dx)
    a, b, c, d = 12 * E * Iy / L**3, 6 * E * Iy / L**2, 4 * E * Iy / L, 2 * E * Iy / L
    k[2, 2] = k[8, 8] = a
    k[2, 8] = -a
    k[2, 4] = k[2, 10] = -b
    k[4, 8] = k[8, 10] = b
    k[4, 4] = k[10, 10] = c
    k[4, 10] = d

    return k + np.triu(k, 1).T


def geometric_stiffness_local(N, L):
    """P-Delta geometric stiffness (chord rotation only), as in the OpenSees PDelta transformation.

    N is the axial force, tension positive.
    """
    kg = np.zeros((12, 12))
    s = N / L
    for a, b in ((1, 7), (2, 8)):
        kg[a, a] = kg[b, b] = s
        kg[a, b] = kg[b, a] = -s
    return kg


def fixed_end_local(w_local, L):
    """Equivalent nodal loads (local) for a uniform load (wx, wy, wz) per unit length in local axes."""
    wx, wy, wz = w_local
    f = np.zeros(12)
    f[0] = f[6] = wx * L / 2
    f[1] = f[7] = wy * L / 2
    f[5] = wy * L**2 / 12
    f[11] = -wy * L**2 / 12
    f[2] = f[8] = wz * L / 2
    f[4] = -wz * L**2 / 12
    f[10] = wz * L**2 / 12
    return f
