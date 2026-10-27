"""Entrada espacial KC→APL basada en contactos FlyWire provisionales.

La amplitud por contacto es exploratoria y debe calibrarse.
"""

import numpy as np
import pandas as pd


def load_kc_apl_mapping(path):
    df = pd.read_parquet(path)
    required = {"direction", "kc_root_id", "segment_id"}
    if not required.issubset(df.columns):
        raise ValueError("Faltan columnas requeridas.")
    return df.loc[df["direction"] == "KC_to_APL"].copy()


def spatial_apl_input(active_kc_ids, mapping, amplitude_per_contact, n_segments=53635):
    """Vector de entradas por segmento para un intervalo temporal."""
    if not np.isfinite(amplitude_per_contact) or amplitude_per_contact < 0:
        raise ValueError("Amplitud inválida.")

    active = np.asarray(active_kc_ids, dtype=np.int64).reshape(-1)
    event_ids, event_counts = np.unique(active, return_counts=True)

    selected = mapping.loc[mapping["kc_root_id"].isin(event_ids)].copy()
    multiplicity = selected["kc_root_id"].map(
        dict(zip(event_ids.tolist(), event_counts.tolist()))
    ).to_numpy(dtype=np.int64)

    segments = selected["segment_id"].to_numpy(dtype=np.int64)

    if np.any(segments < 0) or np.any(segments >= n_segments):
        raise ValueError("Segmentos fuera de rango.")

    result = np.bincount(
        segments, weights=multiplicity, minlength=n_segments
    ).astype(float)
    return result * amplitude_per_contact
