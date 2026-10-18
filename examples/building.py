"""Multi-storey reinforced concrete frame: first-order, P-Delta and gamma_z.

Geometry: 3 x 3 column grid (2 bays of 5 m in each direction), storey height 3 m.
Material: concrete C30, Eci = 5600·sqrt(fck) (NBR 6118, 8.2.8, granite aggregate),
with the simplified secant stiffness 0.7·Eci·Ic for global analysis (NBR 6118, 15.7.3).
Loads are design values (already factored): gravity on beams and wind as storey forces
in +X, shared by the nine nodes of each floor.
"""

import math

from frame3d import Model, rectangular_section

FCK = 30.0                                   # MPa
ECI = 5600 * math.sqrt(FCK) * 1e3            # kN/m²
E = 0.7 * ECI
NU = 0.2
G = E / (2 * (1 + NU))

COLUMN = rectangular_section(0.25, 0.40)     # b along local y, h along local z
BEAM = rectangular_section(0.20, 0.50)


def build(storeys=6, bay=5.0, height=3.0, beam_load=-30.0, wind_per_node=3.0):
    m = Model()
    grid = range(3)
    ids = {}
    for k in range(storeys + 1):
        for i in grid:
            for j in grid:
                ids[i, j, k] = m.add_node(i * bay, j * bay, k * height)
    for i in grid:
        for j in grid:
            m.fix(ids[i, j, 0])

    for k in range(storeys):
        for i in grid:
            for j in grid:
                # columns: local x = +Z, local z = global X (strong axis resists wind in X)
                m.add_element(ids[i, j, k], ids[i, j, k + 1], E, G, COLUMN, (1.0, 0.0, 0.0))
    for k in range(1, storeys + 1):
        for i in grid:
            for j in grid:
                for di, dj in ((1, 0), (0, 1)):
                    if (i + di, j + dj, k) in ids:
                        e = m.add_element(ids[i, j, k], ids[i + di, j + dj, k], E, G, BEAM,
                                          (0.0, 0.0, 1.0))
                        m.add_uniform_load_global(e, wz=beam_load)
        for i in grid:
            for j in grid:
                m.add_nodal_load(ids[i, j, k], fx=wind_per_node)
    return m, ids


if __name__ == "__main__":
    from frame3d import gamma_z, solve_linear, solve_pdelta

    model, ids = build()
    top = ids[1, 1, 6]
    u1 = solve_linear(model)
    u2 = solve_pdelta(model)
    gz = gamma_z(model, u1, direction=(1, 0, 0))
    print(f"nodes {model.n_nodes}, elements {len(model.elements)}")
    print(f"top drift X: first order {u1.displacements[top, 0] * 1e3:.3f} mm, "
          f"P-Delta {u2.displacements[top, 0] * 1e3:.3f} mm, "
          f"ratio {u2.displacements[top, 0] / u1.displacements[top, 0]:.4f}")
    print(f"gamma_z = {gz.gamma_z:.4f} (M1 = {gz.M1:.1f} kN·m, dM = {gz.dM:.2f} kN·m)")
