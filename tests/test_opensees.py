"""Independent cross-check against OpenSees (elasticBeamColumn, Linear and PDelta transformations)."""

import numpy as np
import pytest

pytest.importorskip("openseespy.opensees")

from examples.building import build                     # noqa: E402
from frame3d import solve_linear, solve_pdelta           # noqa: E402
from frame3d.opensees_ref import run_opensees            # noqa: E402

TOL = 1e-9


def rel(a, b):
    return np.abs(a - b).max() / np.abs(b).max()


@pytest.mark.parametrize("pdelta", [False, True], ids=["linear", "pdelta"])
def test_building_matches_opensees(pdelta):
    model, _ = build()
    ours = solve_pdelta(model) if pdelta else solve_linear(model)
    ref = run_opensees(model, pdelta=pdelta)
    assert rel(ours.displacements, ref.displacements) < TOL
    assert rel(ours.reactions, ref.reactions) < TOL
    assert rel(ours.local_forces, ref.local_forces) < TOL


def test_skewed_members_match_opensees():
    """Members in arbitrary directions with mixed local/global member loads."""
    from frame3d import Model, rectangular_section

    m = Model()
    sec = rectangular_section(0.3, 0.5)
    a = m.add_node(0, 0, 0)
    b = m.add_node(3, 1, 4)
    c = m.add_node(6, -2, 4.5)
    d = m.add_node(7, 2, 0)
    m.fix(a)
    m.fix(d)
    e1 = m.add_element(a, b, 30e6, 12.5e6, sec, (1, 0, 0))
    e2 = m.add_element(b, c, 30e6, 12.5e6, sec, (0, 0, 1))
    e3 = m.add_element(c, d, 30e6, 12.5e6, sec, (0, 1, 0))
    m.add_uniform_load_global(e2, wz=-20.0, wy=3.0)
    m.add_uniform_load_local(e1, wy=-4.0, wx=1.5)
    m.add_uniform_load_global(e3, wx=5.0)
    m.add_nodal_load(b, fx=10, fy=-5, fz=-50, mz=3)
    m.add_nodal_load(c, fy=8, mx=-2)
    for pdelta in (False, True):
        ours = solve_pdelta(m) if pdelta else solve_linear(m)
        ref = run_opensees(m, pdelta=pdelta)
        assert rel(ours.displacements, ref.displacements) < TOL
        assert rel(ours.local_forces, ref.local_forces) < TOL
