"""Transmisión APL→KC exploratoria dependiente del voltaje.

No representa una ley fisiológica calibrada.
No modifica el modelo Brian2 original.
"""

import numpy as np


def relative_inhibition(mean_apl_voltage, v_ref=0.001):
    """Señal inhibitoria relativa, adimensional.

    mean_apl_voltage y v_ref deben estar en voltios.
    """
    v = np.asarray(mean_apl_voltage, dtype=float)

    if not np.all(np.isfinite(v)):
        raise ValueError("Voltajes no finitos.")

    if not np.isfinite(v_ref) or v_ref <= 0:
        raise ValueError("v_ref debe ser positivo y finito.")

    return np.maximum(v, 0.0) / v_ref


if __name__ == "__main__":
    test = np.array([-0.001, 0.0, 0.0005, 0.001])
    expected = np.array([0.0, 0.0, 0.5, 1.0])

    result = relative_inhibition(test)

    assert np.allclose(result, expected)

    print("VALIDACIÓN DE TRANSMISIÓN EXPLORATORIA: CORRECTA")
    print("Voltajes:", test)
    print("Inhibición relativa:", result)
