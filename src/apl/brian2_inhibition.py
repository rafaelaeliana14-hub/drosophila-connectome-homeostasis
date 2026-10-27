"""Adaptación exploratoria de señales espaciales APL a Brian2."""

import numpy as np
import pandas as pd

from src.apl.graded_inhibition import relative_inhibition


def spatial_inhibitory_increment(
    apl_output,
    kc_root_ids,
    reference_voltage=0.001,
    inhibitory_scale_v=0.000275,
):
    """
    Devuelve incrementos negativos para la variable g de Brian2.

    La escala y la función de transferencia son exploratorias.
    Los KC no representados espacialmente reciben incremento cero;
    su inhibición original debe gestionarse por separado.
    """
    if not np.isfinite(inhibitory_scale_v) or inhibitory_scale_v < 0:
        raise ValueError("Escala inhibitoria inválida")

    if not np.isfinite(reference_voltage) or reference_voltage <= 0:
        raise ValueError("Voltaje de referencia inválido")

    ids = np.asarray(kc_root_ids, dtype=np.int64)

    if len(np.unique(ids)) != len(ids):
        raise ValueError("Identificadores KC duplicados")

    required = {"mean_apl_voltage"}
    if not required.issubset(apl_output.columns):
        raise ValueError("Falta la señal espacial de APL")

    required_output = {"mean_apl_voltage", "weight"}
    if not required_output.issubset(apl_output.columns):
        raise ValueError("Faltan voltajes o pesos originales")

    if not apl_output.index.is_unique:
        raise ValueError("Identificadores KC duplicados en la salida")

    signal = apl_output["mean_apl_voltage"].reindex(ids)
    weights = apl_output["weight"].reindex(ids)

    increments = np.zeros(len(ids), dtype=float)
    mapped = signal.notna().to_numpy()

    if mapped.any():
        factors = relative_inhibition(
            signal.to_numpy(dtype=float)[mapped],
            v_ref=reference_voltage,
        )
        original_weights = weights.to_numpy(dtype=float)[mapped]

        if not np.all(np.isfinite(original_weights)):
            raise ValueError("Pesos originales no finitos")
        if np.any(original_weights < 0):
            raise ValueError("Pesos originales negativos")

        increments[mapped] = (
            -inhibitory_scale_v * original_weights * factors
        )

    if not np.all(np.isfinite(increments)):
        raise ValueError("Incrementos no finitos")

    return pd.DataFrame({
        "kc_root_id": ids,
        "inhibitory_increment_v": increments,
        "spatially_mapped": mapped,
    })
