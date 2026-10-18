"""Linear (first-order) and P-Delta (second-order) static analysis."""

from dataclasses import dataclass

import numpy as np

from .element import fixed_end_local, geometric_stiffness_local, local_stiffness, transformation


@dataclass
class Result:
    displacements: np.ndarray   # (n_nodes, 6), global axes
    reactions: np.ndarray       # (n_nodes, 6), global axes, zero at free dofs
    local_forces: np.ndarray    # (n_elements, 12): end forces acting on the element, local axes
    axial: np.ndarray           # (n_elements,): axial force, tension positive
    iterations: int


def _element_data(model, e):
    el = model.elements[e]
    L, r3 = model.element_geometry(e)
    s = el.section
    k = local_stiffness(el.E, el.G, s.A, s.Iy, s.Iz, s.J, L)
    t = transformation(r3)
    f = fixed_end_local(el.w_local, L)
    dofs = np.r_[6 * el.i:6 * el.i + 6, 6 * el.j:6 * el.j + 6]
    return L, k, t, f, dofs


def load_vector(model):
    """Global load vector: nodal loads plus equivalent nodal loads of member loads."""
    F = np.zeros(6 * model.n_nodes)
    for node, load in model.nodal_loads.items():
        F[6 * node:6 * node + 6] += load
    for e in range(len(model.elements)):
        _, _, t, f, dofs = _element_data(model, e)
        F[dofs] += t.T @ f
    return F


def _assemble(model, axial=None):
    n = 6 * model.n_nodes
    K = np.zeros((n, n))
    for e in range(len(model.elements)):
        L, k, t, _, dofs = _element_data(model, e)
        if axial is not None:
            k = k + geometric_stiffness_local(axial[e], L)
        K[np.ix_(dofs, dofs)] += t.T @ k @ t
    return K


def _free_dofs(model):
    restrained = np.zeros(6 * model.n_nodes, dtype=bool)
    for node, dofs in model.fixed.items():
        restrained[6 * node:6 * node + 6] = dofs
    return np.flatnonzero(~restrained)


def _recover(model, U, F, K, iterations):
    n_el = len(model.elements)
    local_forces = np.zeros((n_el, 12))
    axial = np.zeros(n_el)
    for e in range(n_el):
        _, k, t, f, dofs = _element_data(model, e)
        q = k @ (t @ U[dofs]) - f
        local_forces[e] = q
        axial[e] = q[6]
    R = K @ U - F
    free = _free_dofs(model)
    R[free] = 0.0
    return Result(U.reshape(-1, 6), R.reshape(-1, 6), local_forces, axial, iterations)


def _solve(model, K, F):
    free = _free_dofs(model)
    U = np.zeros_like(F)
    U[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return U


def solve_linear(model):
    F = load_vector(model)
    K = _assemble(model)
    U = _solve(model, K, F)
    return _recover(model, U, F, K, iterations=1)


def solve_pdelta(model, tol=1e-12, max_iter=100):
    """Second-order analysis with the P-Delta geometric stiffness.

    Fixed-point iteration on the axial forces: solve (K + Kg(N)) U = F,
    update N from U, repeat until the displacement increment is below tol
    (relative to the displacement norm). The converged state is the same
    equilibrium that OpenSees reaches with the PDelta transformation and a
    Newton algorithm.
    """
    F = load_vector(model)
    first = solve_linear(model)
    axial = first.axial
    U_prev = first.displacements.ravel()
    for it in range(1, max_iter + 1):
        K = _assemble(model, axial)
        U = _solve(model, K, F)
        res = _recover(model, U, F, K, it)
        step = np.linalg.norm(U - U_prev) / max(np.linalg.norm(U), 1e-300)
        if step < tol:
            return res
        axial, U_prev = res.axial, U
    raise RuntimeError(f"P-Delta iteration did not converge in {max_iter} iterations")
