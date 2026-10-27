"""Operador eléctrico reducido de APL basado en el SWC v783."""

import numpy as np
import pandas as pd

from src.apl.passive_cable import build_laplacian


def build_anatomical_cable(tree_path, geometry_path,
                           ra=1.0, cm=0.01, gm=1.0):
    """
    ra: resistividad axial, ohm*m
    cm: capacitancia específica, F/m²
    gm: conductancia específica de fuga, S/m²

    Los parámetros son exploratorios, no calibrados para APL.
    """
    if not all(np.isfinite(x) and x > 0 for x in (ra, cm, gm)):
        raise ValueError("ra, cm y gm deben ser finitos y positivos")

    tree = pd.read_parquet(tree_path)
    geom = pd.read_parquet(geometry_path)

    if tree.segment_id.duplicated().any() or geom.segment_id.duplicated().any():
        raise ValueError("Hay identificadores de segmento duplicados")

    data = tree[
        ["segment_id", "start_node_id", "end_node_id"]
    ].merge(
        geom[
            ["segment_id", "lateral_area_m2",
             "axial_resistance_factor_m_inv"]
        ],
        on="segment_id",
        how="left",
        validate="one_to_one"
    )

    if len(data) != len(tree):
        raise ValueError("La geometría no cubre todos los segmentos")

    area = data.lateral_area_m2.to_numpy(dtype=float)
    factor = data.axial_resistance_factor_m_inv.to_numpy(dtype=float)

    if not (
        np.all(np.isfinite(area)) and np.all(area > 0)
        and np.all(np.isfinite(factor)) and np.all(factor > 0)
    ):
        raise ValueError("Geometría inválida: áreas o factores axiales")

    endpoints = data[
        ["start_node_id", "end_node_id"]
    ].to_numpy(dtype=np.int64)

    node_ids, inverse = np.unique(
        endpoints.ravel(), return_inverse=True
    )
    edges = inverse.reshape(-1, 2)

    axial_g = 1.0 / (ra * factor)

    laplacian = build_laplacian(
        len(node_ids), edges, axial_g
    )

    capacitance = np.zeros(len(node_ids), dtype=float)
    leak = np.zeros(len(node_ids), dtype=float)

    half_area = area / 2.0

    for side in (0, 1):
        np.add.at(capacitance, edges[:, side], cm * half_area)
        np.add.at(leak, edges[:, side], gm * half_area)

    if np.any(capacitance <= 0):
        raise ValueError("Nodos sin capacitancia")

    return {
        "node_ids": node_ids,
        "edges": edges,
        "laplacian": laplacian,
        "capacitance": capacitance,
        "leak": leak,
        "segment_ids": data.segment_id.to_numpy(dtype=np.int64),
    }


if __name__ == "__main__":
    base = "data/flywire783/apl_morphology/"
    prefix = "APL_720575940613583001"

    model = build_anatomical_cable(
        base + prefix + "_segment_tree_preliminary_v783.parquet",
        base + prefix + "_segment_geometry_exploratory_v783.parquet"
    )

    L = model["laplacian"]
    C = model["capacitance"]
    G = model["leak"]

    print("Nodos eléctricos:", len(C))
    print("Segmentos:", len(model["edges"]))
    print("Matriz axial:", L.shape)
    print("Elementos no nulos:", L.nnz)
    print("Capacitancia total (F):", C.sum())
    print("Fuga total (S):", G.sum())
    print("Simetría axial:", (L - L.T).nnz == 0)
    print("Suma máxima por fila:", np.max(np.abs(np.asarray(L.sum(axis=1)))))
