"""Builds the same model in OpenSeesPy and runs it, as an independent reference.

Requires ``openseespy``. Element: elasticBeamColumn with a Linear or PDelta
geometric transformation; member loads: eleLoad -beamUniform (local axes).
"""

import numpy as np

from .analysis import Result


def run_opensees(model, pdelta=False):
    import openseespy.opensees as ops

    ops.wipe()
    ops.model("basic", "-ndm", 3, "-ndf", 6)
    for n, (x, y, z) in enumerate(model.nodes):
        ops.node(n + 1, float(x), float(y), float(z))
    for n, dofs in model.fixed.items():
        ops.fix(n + 1, *[int(d) for d in dofs])

    transf = "PDelta" if pdelta else "Linear"
    for e, el in enumerate(model.elements):
        tag = e + 1
        ops.geomTransf(transf, tag, *map(float, el.vecxz))
        s = el.section
        ops.element("elasticBeamColumn", tag, el.i + 1, el.j + 1,
                    s.A, el.E, el.G, s.J, s.Iy, s.Iz, tag)

    ops.timeSeries("Linear", 1)
    ops.pattern("Plain", 1, 1)
    for n, load in model.nodal_loads.items():
        ops.load(n + 1, *map(float, load))
    for e, el in enumerate(model.elements):
        wx, wy, wz = el.w_local
        if wx or wy or wz:
            ops.eleLoad("-ele", e + 1, "-type", "-beamUniform", float(wy), float(wz), float(wx))

    ops.constraints("Plain")
    ops.numberer("RCM")
    ops.system("BandGeneral")
    ops.test("NormDispIncr", 1e-14, 100)
    ops.algorithm("Newton" if pdelta else "Linear")
    ops.integrator("LoadControl", 1.0)
    ops.analysis("Static")
    if ops.analyze(1) != 0:
        raise RuntimeError("OpenSees analysis failed")
    ops.reactions()

    nn, ne = model.n_nodes, len(model.elements)
    U = np.array([ops.nodeDisp(n + 1) for n in range(nn)])
    R = np.array([ops.nodeReaction(n + 1) for n in range(nn)])
    q = np.array([ops.eleResponse(e + 1, "localForce") for e in range(ne)])
    ops.wipe()
    return Result(U, R, q, q[:, 6].copy(), iterations=0)
