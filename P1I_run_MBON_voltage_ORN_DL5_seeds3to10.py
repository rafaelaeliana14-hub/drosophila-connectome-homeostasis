
from pathlib import Path
import numpy as np

import P1I_APL_spatial_MBON_voltage_validation as model

SOURCE = Path(
    "results/P1I_paired_input_ORN_DL5_seed2.npz"
)

CHECKPOINT = Path(
    "results/P1I_multilesion_training/"
    "functional_checkpoints/lesion_seed001/"
    "weights_eta0.030_cycle20.npz"
)

OUTPUT_DIR = Path("results")
STIMULUS = "ORN_DL5"
LESION_SEED = 1
SIMULATION_SEED = 1


def run_seed(input_seed):
    source_path = Path(
        f"results/P1I_paired_input_ORN_DL5_seed{input_seed}.npz"
    )
    with np.load(source_path, allow_pickle=False) as data:
        assert str(data["stimulus"]) == STIMULUS
        assert int(data["input_seed"]) == input_seed

        events = (
            np.asarray(data["input_indices"], dtype=np.int32).copy(),
            np.asarray(data["input_times_ms"], dtype=np.float64).copy(),
        )
        input_hash = str(data["input_sha256"])

    assert len(events[0]) == len(events[1]) > 0

    with np.load(CHECKPOINT, allow_pickle=False) as data:
        weights_P = np.asarray(
            data["weights_after_update_mV"],
            dtype=np.float64,
        ).copy()
        assert weights_P.shape == (460086,)
        assert np.all(np.isfinite(weights_P))
        assert int(data["cycle"]) == 20
        assert np.isclose(float(data["eta"]), 0.030)

    conditions = (
        ("B", None, None),
        ("L", LESION_SEED, None),
        ("P", LESION_SEED, weights_P),
    )

    outputs = {
        name: OUTPUT_DIR / f"P1I_MBON_voltage_ORN_DL5_seed{input_seed}_{name}.npz"
        for name, _, _ in conditions
    }

    for name, lesion, transferred in conditions:
        if outputs[name].exists():
            print(
                f"Semilla {input_seed}, condición {name}: "
                "resultado existente — omitido",
                flush=True,
            )
            continue
        print(f"\n=== VOLTAJE MBON: {name} ===", flush=True)

        result = model.run_condition(
            lesion_seed=lesion,
            stimulus=STIMULUS,
            simulation_seed=SIMULATION_SEED,
            p1e_current_weights_mV=transferred,
            silence_apl_to_kc=False,
            paired_input_events=events,
            return_mbon_voltage=True,
        )

        assert len(result) == 7
        rates = np.asarray(result[3], dtype=np.float64)
        voltage = result[6]

        v = np.asarray(voltage["voltage_mV"], dtype=np.float64)
        t = np.asarray(voltage["times_ms"], dtype=np.float64)
        baseline = np.asarray(voltage["baseline_mask"], dtype=bool)
        stimulus = np.asarray(voltage["stimulus_mask"], dtype=bool)

        assert v.shape[0] == 96
        assert v.shape[1] == len(t)
        assert baseline.any() and stimulus.any()
        assert np.all(np.isfinite(v))

        path = outputs[name]

        np.savez_compressed(
            path,
            voltage_mV=v,
            times_ms=t,
            baseline_mask=baseline,
            stimulus_mask=stimulus,
            mbon_rates_hz=rates,
            input_indices=events[0],
            input_times_ms=events[1],
            input_sha256=input_hash,
            condition=name,
            stimulus=STIMULUS,
            lesion_seed=-1 if lesion is None else lesion,
            simulation_seed=SIMULATION_SEED,
            checkpoint_path=str(CHECKPOINT),
        )

        print("Archivo guardado:", path, flush=True)
        print("MBON activas:", int(np.count_nonzero(rates)), flush=True)
        print(
            "Cambio medio de voltaje (mV):",
            float(np.mean(v[:, stimulus]) - np.mean(v[:, baseline])),
            flush=True,
        )



def main():
    for input_seed in range(3, 11):
        print(
            f"\\n{'=' * 60}\\n"
            f"ORN_DL5 — SEMILLA {input_seed}\\n"
            f"{'=' * 60}",
            flush=True,
        )
        run_seed(input_seed)


if __name__ == "__main__":
    main()
