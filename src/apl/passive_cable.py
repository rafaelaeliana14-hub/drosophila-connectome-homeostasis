"""Núcleo experimental de cable pasivo.

No modifica los modelos Brian2 existentes.
Las unidades del ejemplo son arbitrarias.
"""

import numpy as np
from scipy.sparse import diags, coo_matrix
from scipy.sparse.linalg import spsolve


def build_laplacian(n, edges, conductances):
    """Matriz de acoplamiento de un árbol no dirigido."""
    edges = np.asarray(edges, dtype=int).reshape(-1, 2)
    g = np.asarray(conductances, dtype=float)

    if len(edges) != len(g):
        raise ValueError("Número incompatible de conexiones y conductancias")
    if np.any(g <= 0):
        raise ValueError("Las conductancias deben ser positivas")
    if np.any(edges < 0) or np.any(edges >= n):
        raise ValueError("Índice de compartimento inválido")
    if np.any(edges[:, 0] == edges[:, 1]):
        raise ValueError("Autoconexión no permitida")

    i, j = edges[:, 0], edges[:, 1]

    rows = np.concatenate([i, j, i, j])
    cols = np.concatenate([i, j, j, i])
    values = np.concatenate([g, g, -g, -g])

    return coo_matrix(
        (values, (rows, cols)),
        shape=(n, n)
    ).tocsr()


def passive_step(v, current, capacitance, leak, laplacian, dt):
    """Un paso de Euler implícito para voltajes relativos al reposo."""
    v = np.asarray(v, dtype=float)
    current = np.asarray(current, dtype=float)
    c = np.broadcast_to(np.asarray(capacitance, dtype=float), v.shape)
    gl = np.broadcast_to(np.asarray(leak, dtype=float), v.shape)

    if dt <= 0 or np.any(c <= 0) or np.any(gl < 0):
        raise ValueError("Parámetros pasivos inválidos")
    if current.shape != v.shape or laplacian.shape != (len(v), len(v)):
        raise ValueError("Dimensiones incompatibles")

    A = diags(c / dt + gl) + laplacian
    b = c / dt * v + current

    return spsolve(A.tocsr(), b)


if __name__ == "__main__":
    # Árbol artificial: 0 -- 1 -- 2
    L = build_laplacian(
        n=3,
        edges=[(0, 1), (1, 2)],
        conductances=[1.0, 1.0]
    )

    v = np.zeros(3)
    current = np.array([1.0, 0.0, 0.0])

    for _ in range(1000):
        v = passive_step(
            v=v,
            current=current,
            capacitance=1.0,
            leak=1.0,
            laplacian=L,
            dt=0.01
        )

    expected = np.linalg.solve(
        np.eye(3) + L.toarray(),
        current
    )

    assert np.allclose(v, expected, atol=1e-4)
    assert v[0] > v[1] > v[2] > 0

    print("Prueba del cable pasivo: CORRECTA")
    print("Voltajes simulados:", np.round(v, 6))
    print("Solución estacionaria:", np.round(expected, 6))
    print("Atenuación espacial: CORRECTA")
