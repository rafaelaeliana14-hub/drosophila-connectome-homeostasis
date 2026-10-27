"""Dinámica temporal pasiva de APL con geometría anatómica."""

import numpy as np

from src.apl.factorized_cable import FactorizedCable
from src.apl.nodal_synaptic_input import kc_events_to_nodal_current
from src.apl.nodal_synaptic_output import nodal_kc_signal


class SpatialAPL:
    def __init__(self, model, input_mapping, output_mapping,
                 amplitude_per_contact=1e-15, dt=0.001):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Paso temporal inválido")
        if not np.isfinite(amplitude_per_contact):
            raise ValueError("Amplitud inválida")

        self.model = model
        self.input_mapping = input_mapping
        self.output_mapping = output_mapping
        self.amplitude_per_contact = amplitude_per_contact
        self.dt = dt

        self.voltage = np.zeros(len(model["node_ids"]), dtype=float)
        self.integrator = FactorizedCable(
            model["capacitance"],
            model["leak"],
            model["laplacian"],
            dt=self.dt,
        )

    def step(self, active_kc_ids, kc_gain=None):
        current = kc_events_to_nodal_current(
            active_kc_ids,
            self.input_mapping,
            self.model["segment_ids"],
            self.model["edges"],
            len(self.voltage),
            self.amplitude_per_contact,
            kc_gain=kc_gain,
        )

        self.voltage = self.integrator.step(
            self.voltage,
            current,
        )

        output = nodal_kc_signal(
            self.voltage,
            self.output_mapping,
            self.model["segment_ids"],
            self.model["edges"],
        )

        return output

    def reset(self):
        self.voltage.fill(0.0)
