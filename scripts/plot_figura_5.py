from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINTS = (
    ROOT / "results/P1I_multilesion_training/functional_checkpoints"
)
FIGURES = ROOT / "results/P1I_multilesion_training/figures"
REFERENCE = (
    ROOT / "results/multistimulus_intact/"
    "FULL_MBON_reference_by_stimulus.csv"
)
NEURONS = ROOT / "data/mb_subcircuit/mb_neurons_intact.parquet"

FIGURES.mkdir(parents=True, exist_ok=True)

# Reproducir el orden neuronal utilizado por P1I.
neurons = (
    pd.read_parquet(NEURONS)
    .drop_duplicates("root_id")
    .reset_index(drop=True)
)

mbon_ids = (
    neurons.loc[
        neurons["population"].astype(str) == "MBON",
        "root_id"
    ]
    .astype("int64")
    .tolist()
)

assert len(mbon_ids) == 96

# Referencia intacta, emparejada por root_id.
reference = pd.read_csv(REFERENCE)

assert not reference.duplicated(
    ["stimulus", "root_id"]
).any()

reference_by_stimulus = {}

for stimulus, group in reference.groupby("stimulus"):
    rates = group.set_index("root_id")["mean_rate_hz"]

    reference_by_stimulus[stimulus] = (
        rates.loc[mbon_ids].to_numpy(dtype=float)
    )

assert len(reference_by_stimulus) == 8


def calculate_distances(path):
    with np.load(path, allow_pickle=False) as data:
        mask = np.asarray(data["cycle"]) == 20

        stimuli = np.asarray(data["stimulus"])[mask]
        rates = np.asarray(data["mbon_rates"])[mask]

    assert rates.shape == (8, 96)
    assert len(set(stimuli.tolist())) == 8
    assert set(stimuli.tolist()) == set(reference_by_stimulus)

    intact = np.stack([
        reference_by_stimulus[str(stimulus)]
        for stimulus in stimuli
    ])

    assert np.isfinite(rates).all()
    assert np.isfinite(intact).all()

    absolute_errors = np.abs(rates - intact)

    global_distance = float(absolute_errors.mean())

    dm1_mask = stimuli == "ORN_DM1"
    assert dm1_mask.sum() == 1

    dm1_distance = float(
        absolute_errors[dm1_mask].mean()
    )

    return global_distance, dm1_distance


rows = []

for lesion in range(1, 21):
    folder = CHECKPOINTS / f"lesion_seed{lesion:03d}"

    control = folder / "functional_eta0.000_cycle20.npz"
    plastic = folder / "functional_eta0.030_cycle20.npz"

    if not control.exists() or not plastic.exists():
        raise FileNotFoundError(
            f"Faltan checkpoints para lesión {lesion}"
        )

    global_control, dm1_control = calculate_distances(control)
    global_plastic, dm1_plastic = calculate_distances(plastic)

    rows.append({
        "lesion": lesion,
        "global_control": global_control,
        "global_plastic": global_plastic,
        "dm1_control": dm1_control,
        "dm1_plastic": dm1_plastic,
        "delta_dm1": dm1_plastic - dm1_control,
    })

df = pd.DataFrame(rows)

# Comprobación contra los resultados previamente obtenidos.
assert np.isclose(df["global_control"].mean(), 0.038125, atol=1e-6)
assert np.isclose(df["global_plastic"].mean(), 0.046615, atol=1e-6)
assert np.isclose(df["dm1_control"].mean(), 0.305000, atol=1e-6)
assert np.isclose(df["dm1_plastic"].mean(), 0.372917, atol=1e-6)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    1, 3,
    figsize=(13.5, 4.6),
    gridspec_kw={"width_ratios": [1, 1, 1.25]},
)

color_control = "#54748A"
color_plastic = "#C06A45"
color_closer = "#387B68"
color_farther = "#BA654C"
color_unchanged = "#858585"

# Panel A: distancia global.
ax = axes[0]

for _, row in df.iterrows():
    ax.plot(
        [0, 1],
        [row["global_control"], row["global_plastic"]],
        color="#A8A8A8",
        alpha=0.65,
        linewidth=1,
        zorder=1,
    )

ax.scatter(
    np.zeros(len(df)),
    df["global_control"],
    color=color_control,
    s=27,
    zorder=3,
)
ax.scatter(
    np.ones(len(df)),
    df["global_plastic"],
    color=color_plastic,
    s=27,
    zorder=3,
)

ax.set_xticks([0, 1], ["Sin plasticidad", "Con plasticidad"])
ax.set_ylabel("Distancia absoluta media (Hz)")
ax.set_title("A  Ocho estímulos", loc="left", fontweight="bold")
ax.set_xlim(-0.3, 1.3)

# Panel B: ORN_DM1.
ax = axes[1]

for _, row in df.iterrows():
    ax.plot(
        [0, 1],
        [row["dm1_control"], row["dm1_plastic"]],
        color="#A8A8A8",
        alpha=0.65,
        linewidth=1,
        zorder=1,
    )

ax.scatter(
    np.zeros(len(df)),
    df["dm1_control"],
    color=color_control,
    s=27,
    zorder=3,
)
ax.scatter(
    np.ones(len(df)),
    df["dm1_plastic"],
    color=color_plastic,
    s=27,
    zorder=3,
)

ax.set_xticks([0, 1], ["Sin plasticidad", "Con plasticidad"])
ax.set_ylabel("Distancia absoluta media (Hz)")
ax.set_title("B  Estímulo ORN_DM1", loc="left", fontweight="bold")
ax.set_xlim(-0.3, 1.3)

# Panel C: cambio individual por lesión.
ax = axes[2]

colors = [
    color_farther if value > 1e-10
    else color_closer if value < -1e-10
    else color_unchanged
    for value in df["delta_dm1"]
]

ax.bar(
    df["lesion"],
    df["delta_dm1"],
    color=colors,
    width=0.75,
)

ax.axhline(0, color="#333333", linewidth=0.9)

ax.set_xticks(range(1, 21, 2))
ax.set_xlabel("Semilla de lesión")
ax.set_ylabel("Cambio de distancia, ΔD (Hz)")
ax.set_title(
    "C  Cambio por lesión · ORN_DM1",
    loc="left",
    fontweight="bold",
)

n_farther = int((df["delta_dm1"] > 1e-10).sum())
n_closer = int((df["delta_dm1"] < -1e-10).sum())
n_unchanged = int((df["delta_dm1"].abs() <= 1e-10).sum())

ax.text(
    0.98, 0.96,
    f"Alejamiento: {n_farther}/20\n"
    f"Aproximación: {n_closer}/20\n"
    f"Sin cambio: {n_unchanged}/20",
    transform=ax.transAxes,
    ha="right",
    va="top",
    fontsize=9,
    bbox={
        "boxstyle": "round,pad=0.4",
        "facecolor": "white",
        "edgecolor": "#D5D5D5",
    },
)

fig.suptitle(
    "Proximidad de las respuestas MBON a la referencia intacta",
    fontsize=13,
    fontweight="bold",
    y=1.02,
)

fig.tight_layout()

png = FIGURES / "Figura_5_distancia_MBON_intacto.png"
pdf = FIGURES / "Figura_5_distancia_MBON_intacto.pdf"

fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")

plt.close(fig)

print("\nFIGURA 5 GENERADA")
print("=" * 65)
print("PNG:", png)
print("PDF:", pdf)
print()
print("Distancia global sin plasticidad:",
      f"{df['global_control'].mean():.6f} Hz")
print("Distancia global con plasticidad:",
      f"{df['global_plastic'].mean():.6f} Hz")
print("Distancia ORN_DM1 sin plasticidad:",
      f"{df['dm1_control'].mean():.6f} Hz")
print("Distancia ORN_DM1 con plasticidad:",
      f"{df['dm1_plastic'].mean():.6f} Hz")
print()
print("Se alejan:", n_farther)
print("Se acercan:", n_closer)
print("Sin cambios:", n_unchanged)
