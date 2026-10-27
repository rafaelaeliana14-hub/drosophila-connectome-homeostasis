"""Integrador pasivo con factorización reutilizable."""

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu


class FactorizedCable:
    def __init__(self, capacitance, leak, laplacian, dt=0.001):
        self.C = np.asarray(capacitance, dtype=float)
        self.G = np.asarray(leak, dtype=float)
        self.dt = float(dt)

        if self.C.ndim != 1 or self.G.shape != self.C.shape:
            raise ValueError("Dimensiones incompatibles")

        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("Paso temporal inválido")

        if not np.all(np.isfinite(self.C)) or np.any(self.C <= 0):
            raise ValueError("Capacitancias inválidas")

        if not np.all(np.isfinite(self.G)) or np.any(self.G < 0):
            raise ValueError("Conductancias de fuga inválidas")

        n = len(self.C)

        if laplacian.shape != (n, n):
            raise ValueError("Dimensiones del laplaciano incompatibles")

        self.C_over_dt = self.C / self.dt

        A = (
            diags(self.C_over_dt + self.G)
            + laplacian
        ).tocsc()

        self.solver = splu(A)

    def step(self, voltage, current):
        v = np.asarray(voltage, dtype=float)
        I = np.asarray(current, dtype=float)

        if v.shape != self.C.shape or I.shape != self.C.shape:
            raise ValueError("Dimensiones incompatibles")

        if not np.all(np.isfinite(v)) or not np.all(np.isfinite(I)):
            raise ValueError("Voltajes o corrientes no finitos")

        rhs = self.C_over_dt * v + I
        return self.solver.solve(rhs)
