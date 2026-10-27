"""Lectura espacial APL→KC desde voltajes nodales."""

import numpy as np
import pandas as pd


def nodal_kc_signal(voltage, mapping, segment_ids, edges):
    """
    Calcula el voltaje medio de APL recibido por cada KC.

    Cada segmento se evalúa mediante el promedio de sus
    dos voltajes nodales. Los contactos se ponderan usando
    distributed_connectivity, que conserva los pesos
    originales del conectoma funcional.
    """
    voltage = np.asarray(voltage, dtype=float)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)

    if voltage.ndim != 1 or not np.all(np.isfinite(voltage)):
        raise ValueError("Voltajes inválidos")

    if edges.shape != (len(segment_ids), 2):
        raise ValueError("Dimensiones de segmentos incompatibles")

    if np.any(edges < 0) or np.any(edges >= len(voltage)):
        raise ValueError("Índices nodales fuera de rango")

    if len(np.unique(segment_ids)) != len(segment_ids):
        raise ValueError("Segmentos duplicados")

    required = {
        "kc_root_id", "segment_id", "distributed_connectivity"
    }
    if not required.issubset(mapping.columns):
        raise ValueError("Faltan columnas del mapeo")

    index = pd.Series(np.arange(len(segment_ids)), index=segment_ids)
    idx = index.reindex(mapping["segment_id"].to_numpy(dtype=np.int64))

    if idx.isna().any():
        raise ValueError("Segmentos sin correspondencia anatómica")

    endpoints = edges[idx.to_numpy(dtype=int)]
    segment_voltage = (
        voltage[endpoints[:, 0]] + voltage[endpoints[:, 1]]
    ) / 2.0

    weights = mapping["distributed_connectivity"].to_numpy(dtype=float)

    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("Pesos sinápticos inválidos")

    result = pd.DataFrame({
        "kc_root_id": mapping["kc_root_id"].to_numpy(),
        "weighted_voltage": weights * segment_voltage,
        "weight": weights,
    })

    grouped = result.groupby("kc_root_id", sort=True).sum()

    if np.any(grouped["weight"].to_numpy() <= 0):
        raise ValueError("Hay KC con peso total no positivo")

    grouped["mean_apl_voltage"] = (
        grouped["weighted_voltage"] / grouped["weight"]
    )

    return grouped
