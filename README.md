# frame3d-pdelta

Linear and P-Delta static analysis of 3D frames in Python, with the global stability
coefficient γz of ABNT NBR 6118, cross-checked against OpenSees.

[![tests](https://github.com/LuizRMSilva1973/frame3d-pdelta/actions/workflows/tests.yml/badge.svg)](https://github.com/LuizRMSilva1973/frame3d-pdelta/actions/workflows/tests.yml)

## What it does

* Assembles and solves 3D frames with 12-DOF Euler-Bernoulli beam-column elements
  (axial, torsion and biaxial bending), nodal loads and uniform member loads in local
  or global axes.
* Runs a second-order **P-Delta** analysis (geometric stiffness from the chord rotation,
  iterated on the axial forces until equilibrium).
* Computes **γz** (NBR 6118, item 15.5.3) from the first-order results and compares it with
  the amplification obtained by the P-Delta analysis.
* Rebuilds every model in **OpenSeesPy** (`elasticBeamColumn`, `Linear` and `PDelta`
  transformations) and compares displacements, reactions and element end forces.

## Method

**Element.** Local stiffness of a prismatic Euler-Bernoulli member, no shear deformation.
Local axes follow the OpenSees `geomTransf` convention: *x* from node *i* to node *j*,
`vecxz` in the local *x–z* plane, *y = vecxz × x*, *z = x × y*. `Iz` governs bending in the
local *x–y* plane and `Iy` bending in the *x–z* plane.

**Member loads.** Uniform loads are replaced by consistent fixed-end forces; element end
forces are recovered as *q = k·u − f₀* (forces acting on the element, local axes).

**P-Delta.** For each member, the geometric stiffness adds *N/L* to the transverse
translational terms (chord rotation only, no P-δ), with *N* positive in tension:

    (K + Kg(N)) U = F

solved by fixed-point iteration on *N* until ‖ΔU‖/‖U‖ < 10⁻¹². This is the equilibrium that
OpenSees reaches with the `PDelta` transformation and a Newton algorithm.

**γz (NBR 6118, 15.5.3).**

    γz = 1 / (1 − ΔMtot,d / M1,tot,d)

*M1,tot,d* is the overturning moment of the horizontal design forces about the base and
*ΔMtot,d* the sum of the vertical design forces times the first-order horizontal
displacements of their points of application. The code treats γz ≤ 1.1 as a non-sway
structure.

**Example building** (`examples/building.py`): six storeys of 3 m, a 3 × 3 column grid with
5 m bays, C30 concrete with *Eci = 5600·√fck* (NBR 6118, 8.2.8) and the simplified secant
stiffness *0.7·Eci·Ic* for global analysis (NBR 6118, 15.7.3); design gravity load on the
beams and wind as storey forces.

## Validation

Full report, regenerated from the code: [VALIDATION.md](VALIDATION.md).

| Check | Result |
|---|---|
| Building, first order: max relative difference to OpenSees (displacements / end forces) | 1.6·10⁻¹³ / 4.7·10⁻¹⁵ |
| Building, P-Delta: max relative difference to OpenSees (displacements / end forces) | 1.3·10⁻¹³ / 3.1·10⁻¹⁵ |
| Skewed frame with members in arbitrary directions and mixed member loads | < 10⁻⁹ (test) |
| γz vs P-Delta amplification of the top drift | 1.0556 vs 1.0578 (−0.21 %) |

Closed-form checks in `tests/test_analytical.py`: cantilever tip deflection and rotation in
both planes, axial and torsional stiffness, independence from member orientation, fixed-fixed
beam under uniform load (mid-span deflection, end moments, reactions), and the exact P-Delta
amplification of a single-element cantilever, *1 / (1 − N·L²/(3EI))*. `tests/test_gamma_z.py`
checks γz against the same closed form for a single column.

## Usage

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest                                   # analytical checks + OpenSees cross-checks
python -m examples.building              # drift, P-Delta amplification and gamma_z
python -m examples.validation_report     # regenerates VALIDATION.md
```

```python
from frame3d import Model, rectangular_section, solve_linear, solve_pdelta, gamma_z

m = Model()
base, top = m.add_node(0, 0, 0), m.add_node(0, 0, 6.0)
m.fix(base)
m.add_element(base, top, E=21.5e6, G=9.0e6, section=rectangular_section(0.25, 0.40), vecxz=(1, 0, 0))
m.add_nodal_load(top, fx=10.0, fz=-300.0)          # kN
u1, u2 = solve_linear(m), solve_pdelta(m)
print(gamma_z(m, u1).gamma_z, u2.displacements[top, 0] / u1.displacements[top, 0])
# 1.1436 1.1436 — for a single column both reduce to 1 / (1 - N·L²/(3EI))
```

Units are consistent but not enforced; the examples use kN and m.

## Scope and limitations

Linear elastic material, small displacements, P-Delta without P-δ (members are not
subdivided automatically), no shear deformation, no rigid end offsets or floor diaphragms,
loads taken as design values already factored. It is a verification and teaching tool,
not a design program.

## Authorship

Luiz Ricardo Mantovani da Silva, civil engineer (CREA-SP 5071695331). The implementation was
written with the help of an AI coding assistant; correctness rests on the automated checks
above (closed-form solutions and OpenSees), which run on every push.

## License

MIT — see [LICENSE](LICENSE).
