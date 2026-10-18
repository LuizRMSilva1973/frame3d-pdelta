"""gamma_z (NBR 6118, 15.5.3) against closed form and against the P-Delta amplification."""

import pytest

from examples.building import build
from frame3d import Model, Section, gamma_z, solve_linear, solve_pdelta


def test_gamma_z_single_column_closed_form():
    """Cantilever column, top loads H and N: dM/M1 = N·L²/(3EI)."""
    E, G, L, H, N = 30e6, 12.5e6, 6.0, 10.0, 800.0
    sec = Section(A=0.12, Iy=1.6e-3, Iz=6.25e-4, J=1e-3)
    m = Model()
    base, top = m.add_node(0, 0, 0), m.add_node(0, 0, L)
    m.fix(base)
    m.add_element(base, top, E, G, sec, (1, 0, 0))
    m.add_nodal_load(top, fx=H, fz=-N)
    gz = gamma_z(m, solve_linear(m), direction=(1, 0, 0))
    assert gz.M1 == pytest.approx(H * L, rel=1e-12)
    assert gz.gamma_z == pytest.approx(1 / (1 - N * L**2 / (3 * E * sec.Iy)), rel=1e-12)


def test_gamma_z_close_to_pdelta_amplification_in_building():
    model, ids = build()
    top = ids[1, 1, 6]
    u1 = solve_linear(model)
    u2 = solve_pdelta(model)
    gz = gamma_z(model, u1, direction=(1, 0, 0))
    amplification = u2.displacements[top, 0] / u1.displacements[top, 0]
    assert 1.0 < gz.gamma_z < 1.1          # non-sway structure for NBR 6118
    assert gz.gamma_z == pytest.approx(amplification, rel=0.01)
