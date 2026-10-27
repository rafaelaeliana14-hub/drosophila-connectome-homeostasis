"""Distribución exploratoria de entradas KC→APL sobre nodos eléctricos."""

import numpy as np
import pandas as pd


def kc_events_to_nodal_current(
    active_kc_ids,
    mapping,
    segment_ids,
    edges,
    n_nodes,
    amplitude_per_contact,
    kc_gain=None,
):
    """
    Distribuye por mitades la corriente de cada contacto entre
    los extremos del segmento correspondiente.

    active_kc_ids puede contener identificadores repetidos:
    cada repetición representa un evento adicional.
    """
    if not np.isfinite(amplitude_per_contact):
        raise ValueError("Amplitud no finita")

    if n_nodes <= 0:
        raise ValueError("Número de nodos inválido")

    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    edges = np.asarray(edges, dtype=np.int64)

    if edges.shape != (len(segment_ids), 2):
        raise ValueError("Dimensiones incompatibles")

    if np.any(edges < 0) or np.any(edges >= n_nodes):
        raise ValueError("Índices nodales inválidos")

    if len(np.unique(segment_ids)) != len(segment_ids):
        raise ValueError("Segmentos duplicados")

    active = np.asarray(active_kc_ids, dtype=np.int64)

    current = np.zeros(n_nodes, dtype=float)

    if active.size == 0:
        return current

    counts = pd.Series(active).value_counts()

    contacts = mapping.loc[
        mapping["kc_root_id"].isin(counts.index),
        ["kc_root_id", "segment_id"]
    ].copy()

    if contacts.empty:
        return current

    contacts["multiplicity"] = (
        contacts["kc_root_id"].map(counts).astype(int)
    )

    segment_index = pd.Series(
        np.arange(len(segment_ids)), index=segment_ids
    )

    idx = segment_index.reindex(
        contacts["segment_id"].to_numpy(dtype=np.int64)
    )

    if idx.isna().any():
        raise ValueError("Hay contactos asignados a segmentos inexistentes")

    endpoints = edges[idx.to_numpy(dtype=int)]
    if kc_gain is None:
        gain = np.ones(len(contacts), dtype=float)
    else:
        gain = contacts["kc_root_id"].map(kc_gain).fillna(1.0).to_numpy(dtype=float)
        if not np.isfinite(gain).all() or np.any(gain < 0):
            raise ValueError("Factores KC→APL inválidos")

    weights = (
        contacts["multiplicity"].to_numpy(dtype=float)
        * gain
        * amplitude_per_contact / 2.0
    )

    np.add.at(current, endpoints[:, 0], weights)
    np.add.at(current, endpoints[:, 1], weights)

    return current
