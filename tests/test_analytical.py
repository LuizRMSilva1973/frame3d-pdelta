"""Closed-form checks: beam theory results the solver must reproduce exactly."""

import numpy as np
import pytest

from frame3d import Model, Section, solve_linear, solve_pdelta

E, G = 200e6, 80e6
SEC = Section(A=0.01, Iy=2e-4, Iz=5e-5, J=1e-5)
L, P = 4.0, 10.0


def cantilever(direction=(1.0, 0.0, 0.0), vecxz=(0.0, 0.0, 1.0), n_el=1):
    m = Model()
    d = np.asarray(direction, dtype=float) / np.linalg.norm(direction)
    nodes = [m.add_node(*(d * L * k / n_el)) for k in range(n_el + 1)]
    m.fix(nodes[0])
    for a, b in zip(nodes, nodes[1:]):
        m.add_element(a, b, E, G, SEC, vecxz)
    return m, nodes[-1]


def test_cantilever_tip_load_both_planes():
    m, tip = cantilever()
    m.add_nodal_load(tip, fy=P, fz=-P)
    u = solve_linear(m).displacements[tip]
    # local y = global Y (bending with Iz), local z = global Z (bending with Iy)
    assert u[1] == pytest.approx(P * L**3 / (3 * E * SEC.Iz), rel=1e-12)
    assert u[2] == pytest.approx(-P * L**3 / (3 * E * SEC.Iy), rel=1e-12)
    assert u[5] == pytest.approx(P * L**2 / (2 * E * SEC.Iz), rel=1e-12)
    assert u[4] == pytest.approx(P * L**2 / (2 * E * SEC.Iy), rel=1e-12)


def test_axial_and_torsion():
    m, tip = cantilever()
    m.add_nodal_load(tip, fx=P, mx=P)
    u = solve_linear(m).displacements[tip]
    assert u[0] == pytest.approx(P * L / (E * SEC.A), rel=1e-12)
    assert u[3] == pytest.approx(P * L / (G * SEC.J), rel=1e-12)


def test_result_does_not_depend_on_member_orientation():
    d = np.array([1.0, 2.0, 2.0]) / 3.0
    m, tip = cantilever(direction=d, vecxz=(0.0, 0.0, 1.0))
    load = np.cross(d, [0.0, 0.0, 1.0])
    load = P * load / np.linalg.norm(load)          # perpendicular to the member, along local y
    m.add_nodal_load(tip, *load)
    u = solve_linear(m).displacements[tip, :3]
    assert np.linalg.norm(u) == pytest.approx(P * L**3 / (3 * E * SEC.Iz), rel=1e-12)
    assert np.dot(u, d) == pytest.approx(0.0, abs=1e-15)


def test_fixed_fixed_beam_uniform_load():
    w = -12.0
    m = Model()
    a, mid, b = (m.add_node(x, 0, 0) for x in (0.0, L / 2, L))
    m.fix(a)
    m.fix(b)
    for i, j in ((a, mid), (mid, b)):
        e = m.add_element(i, j, E, G, SEC, (0.0, 0.0, 1.0))
        m.add_uniform_load_global(e, wz=w)
    res = solve_linear(m)
    assert res.displacements[mid, 2] == pytest.approx(w * L**4 / (384 * E * SEC.Iy), rel=1e-12)
    end_moment = abs(res.local_forces[0, 4])
    assert end_moment == pytest.approx(abs(w) * L**2 / 12, rel=1e-12)
    assert res.reactions[a, 2] == pytest.approx(-w * L / 2, rel=1e-12)


def test_pdelta_single_element_cantilever_closed_form():
    """With chord-only P-Delta, one element gives k = 3EI/L³ - P/L exactly."""
    H, N = 1.0, 500.0
    m, tip = cantilever(direction=(0.0, 0.0, 1.0), vecxz=(1.0, 0.0, 0.0))
    m.add_nodal_load(tip, fx=H, fz=-N)
    u1 = solve_linear(m).displacements[tip, 0]
    u2 = solve_pdelta(m).displacements[tip, 0]
    EI = E * SEC.Iy                       # local z = global X: bending in X uses Iy
    shortening = N * L / (E * SEC.A)      # P-Delta uses the converged axial force
    assert u1 == pytest.approx(H * L**3 / (3 * EI), rel=1e-12)
    assert u2 / u1 == pytest.approx(1.0 / (1.0 - N * L**2 / (3 * EI)), rel=1e-9)
    assert solve_pdelta(m).displacements[tip, 2] == pytest.approx(-shortening, rel=1e-9)
