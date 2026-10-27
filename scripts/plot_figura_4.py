import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

BASE = Path("results/P1I_multilesion_training/functional_checkpoints")
SALIDA = Path("results/P1I_multilesion_training/figures")

OLORES = [
    "ORN_DA1", "ORN_DA2", "ORN_DC4", "ORN_DL5",
    "ORN_DM1", "ORN_DP1l", "ORN_VL2a", "ORN_VM4"
]

apl = np.zeros((20, 2, 2))
mbon_inicial = np.zeros((20, 8, 96))
mbon_final = np.zeros((20, 8, 96))

for lesion in range(1, 21):
    carpeta = BASE / f"lesion_seed{lesion:03d}"

    for momento, ciclo in enumerate([1, 20]):
        archivo = carpeta / f"functional_eta0.030_cycle{ciclo:02d}.npz"

        with np.load(archivo, allow_pickle=False) as datos:
            filtro = datos["cycle"] == ciclo
            olores = datos["stimulus"][filtro].astype(str)
            tasas_apl = datos["apl_rates"][filtro]
            tasas_mbon = datos["mbon_rates"][filtro]

            assert len(olores) == 8
            assert set(olores) == set(OLORES)

            orden = [list(olores).index(o) for o in OLORES]

            apl[lesion - 1, momento] = tasas_apl[orden].mean(axis=0)

            if momento == 0:
                mbon_inicial[lesion - 1] = tasas_mbon[orden]
            else:
                mbon_final[lesion - 1] = tasas_mbon[orden]

delta = mbon_final - mbon_inicial

positivos = np.sum(delta > 1e-10, axis=(1, 2))
negativos = np.sum(delta < -1e-10, axis=(1, 2))
cambios_por_olor = np.sum(np.abs(delta) > 1e-10, axis=(0, 2))

assert positivos.sum() == 8
assert negativos.sum() == 38
assert cambios_por_olor.sum() == 46
assert np.count_nonzero(positivos + negativos) == 19
assert np.allclose(
    apl[:, 1, :].mean(axis=0) - apl[:, 0, :].mean(axis=0),
    [0.4625, 0.8500]
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

fig, axes = plt.subplots(
    1, 3, figsize=(15, 5), constrained_layout=True
)

# A. Actividad de APL
ax = axes[0]
colores = ["#2676A8", "#C16A3B"]

for i in range(2):
    x0, x1 = i * 3, i * 3 + 1

    for lesion in range(20):
        ax.plot(
            [x0, x1],
            [apl[lesion, 0, i], apl[lesion, 1, i]],
            color=colores[i], alpha=0.3, linewidth=1
        )

    for momento, posicion in enumerate([x0, x1]):
        ax.scatter(
            np.full(20, posicion),
            apl[:, momento, i],
            color=colores[i], s=18, alpha=0.6
        )

    ax.plot(
        [x0, x1],
        [apl[:, 0, i].mean(), apl[:, 1, i].mean()],
        color="black", linewidth=2.5,
        marker="D", markersize=6
    )

ax.set_xticks([0, 1, 3, 4])
ax.set_xticklabels([
    "APL1\nInicial", "APL1\nFinal",
    "APL2\nInicial", "APL2\nFinal"
])
ax.set_ylabel("Tasa media de disparo (Hz)")
ax.set_title("A | Actividad de APL", loc="left", fontweight="bold")
ax.grid(axis="y", alpha=0.15)

# B. Cambios MBON por lesión
ax = axes[1]
x = np.arange(1, 21)

ax.bar(x, positivos, label="Aumentos", color="#2676A8")
ax.bar(x, -negativos, label="Disminuciones", color="#C16A3B")
ax.axhline(0, color="black", linewidth=0.8)

ax.set_xticks([1, 5, 10, 15, 20])
ax.set_xlabel("Configuración de lesión")
ax.set_ylabel("Número de respuestas olor–MBON modificadas")
ax.set_title(
    "B | Cambios de actividad MBON",
    loc="left", fontweight="bold"
)
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.15)

# C. Selectividad por estímulo
ax = axes[2]

ax.barh(
    OLORES,
    cambios_por_olor,
    color=[
        "#C16A3B" if o == "ORN_DM1" else "#A9B0B6"
        for o in OLORES
    ]
)

ax.invert_yaxis()
ax.set_xlabel("Cambios de respuesta acumulados entre lesiones")
ax.set_title(
    "C | Selectividad por estímulo",
    loc="left", fontweight="bold"
)

for i, valor in enumerate(cambios_por_olor):
    if valor > 0:
        ax.text(valor + 0.5, i, str(int(valor)), va="center")

ax.set_xlim(0, 52)
ax.grid(axis="x", alpha=0.15)

SALIDA.mkdir(parents=True, exist_ok=True)

for extension in ["pdf", "png"]:
    archivo = SALIDA / f"Figura_4_actividad_APL_MBON.{extension}"
    opciones = {"dpi": 300} if extension == "png" else {}
    fig.savefig(archivo, bbox_inches="tight", **opciones)

plt.close(fig)

print("Figura 4 regenerada correctamente.")
print("20 lesiones | 46 cambios MBON | 8 aumentos | 38 disminuciones")
