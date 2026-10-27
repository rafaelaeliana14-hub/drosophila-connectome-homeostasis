from pathlib import Path
import json
import numpy as np
import pandas as pd

from brian2 import (
    NeuronGroup,
    Synapses,
    PoissonGroup,
    SpikeMonitor,
    StateMonitor,
    Network,
    mV,
    ms,
    Hz,
    seed,
    prefs,
    defaultclock,
)

# ============================================================
# 1. CONFIGURACIÓN GENERAL
# ============================================================


# ===== P1-E FROZEN REFERENCE =====

P1E_REFERENCE_FILE = (
    Path(__file__).resolve().parent
    / "results"
    / "P1E_reference"
    / "P1E_reference_Gstar.npz"
)

P1E_EXPECTED_SHA256 = (
    "a71920d7301567cacff30767005ef85aa"
    "3d75f8f467afd70225ebd6af6623140"
)

# Parámetros P1-E preespecificados.
P1E_ETA_GRID = (0.0, 0.01, 0.03, 0.10)
P1E_N_CYCLES = 20
P1E_ERROR_CLIP = 1.0
P1E_WEIGHT_MIN_FACTOR = 0.5
P1E_WEIGHT_MAX_FACTOR = 2.0

# IMPORTANTE:
# Estos parámetros quedan declarados aquí ANTES de ejecutar
# cualquier simulación de recuperación.
#
# Este bloque todavía NO modifica pesos.
# =======================================

ROOT = Path(__file__).resolve().parent

NEURONS_FILE = (
    ROOT
    / "data"
    / "mb_subcircuit"
    / "mb_neurons_intact.parquet"
)

EDGES_FILE = (
    ROOT
    / "data"
    / "mb_subcircuit"
    / "mb_connectivity_intact.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "results"
    / "P1I_multilesion_training"
)

CHECKPOINT_DIR = (
    OUTPUT_DIR
    / "checkpoints"
)

LESION_IDS_DIR = (
    OUTPUT_DIR
    / "lesion_ids"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

LESION_IDS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RUN_SUMMARY_FILE = (
    OUTPUT_DIR
    / "P0_run_summary.csv"
)

KC_LONG_FILE = (
    OUTPUT_DIR
    / "P0_KC_activity_long.csv"
)

APL_LONG_FILE = (
    OUTPUT_DIR
    / "P0_APL_activity_long.csv"
)

MBON_LONG_FILE = (
    OUTPUT_DIR
    / "P0_MBON_activity_long.csv"
)

DESCRIPTIVE_FILE = (
    OUTPUT_DIR
    / "P0_descriptive_by_stimulus.csv"
)

CONFIG_FILE = (
    OUTPUT_DIR
    / "P0_experiment_config.json"
)

PROGRESS_FILE = (
    OUTPUT_DIR
    / "P0_progress.csv"
)


# ============================================================
# 2. DISEÑO EXPERIMENTAL PREDECLARADO
# ============================================================

STIMULI = [
    "ORN_DM1",
    "ORN_DL5",
    "ORN_DA2",
    "ORN_DA1",
    "ORN_DC4",
    "ORN_DP1l",
    "ORN_VM4",
    "ORN_VL2a",
]

# Selección numérica predefinida.
# NO se eligen según resultados del P0 anterior.
LESION_SEEDS = list(range(1, 21))

# Variabilidad estocástica independiente de la identidad
# de la lesión.
SIMULATION_SEEDS = [1]

LESION_FRACTION = 0.25

# ============================================================
# 3. TEMPORALIDAD DEL EXPERIMENTO
# ============================================================

WARMUP_MS = 200
BASELINE_MS = 500
STIMULUS_MS = 500
POST_MS = 500

# IMPORTANTE:
# Esto es la tasa del drive Poisson POR ORN seleccionada.
# NO significa que la ORN resultante dispare a 100 Hz.
STIMULUS_RATE_HZ = 100


# ============================================================
# 4. PARÁMETROS LIF
#    COPIADOS DEL P0 ORIGINAL
# ============================================================

V_0 = -52 * mV
V_RST = -52 * mV
V_TH = -45 * mV

T_MBR = 20 * ms
TAU = 5 * ms
T_RFC = 2.2 * ms
T_DLY = 1.8 * ms

W_SYN = 0.275 * mV
F_POI = 250

prefs.codegen.target = "numpy"


# ============================================================
# 5. INICIO
# ============================================================

print("\n" + "=" * 72)
print(" P0 MULTIESTÍMULO")
print(" LESIÓN FUNCIONAL DE SALIDA DEL 25% DE KC")
print(" PLASTICIDAD: NO")
print("=" * 72)


# ============================================================
# 6. CARGAR DATOS
# ============================================================

print("\nCargando neuronas...")

neurons_df = pd.read_parquet(
    NEURONS_FILE
)

print("Cargando conectividad...")

edges_df = pd.read_parquet(
    EDGES_FILE
)

neurons_df = (
    neurons_df
    .drop_duplicates("root_id")
    .reset_index(drop=True)
)

neuron_ids = (
    neurons_df["root_id"]
    .astype(np.int64)
    .to_numpy()
)

N = len(neuron_ids)

id_to_idx = {
    int(root_id): idx
    for idx, root_id
    in enumerate(neuron_ids)
}

population_array = (
    neurons_df["population"]
    .astype(str)
    .to_numpy()
)

population_by_id = (
    neurons_df
    .set_index("root_id")[
        "population"
    ]
)


# ============================================================
# 7. VERIFICAR POBLACIONES
# ============================================================

expected_counts = {
    "ORN": 2279,
    "ALPN": 685,
    "KC": 5177,
    "APL": 2,
    "MBON": 96,
}

observed_counts = (
    neurons_df["population"]
    .astype(str)
    .value_counts()
    .to_dict()
)

print("\nControl de poblaciones:")

for population, expected in expected_counts.items():

    observed = int(
        observed_counts.get(
            population,
            0,
        )
    )

    print(
        f"  {population}: "
        f"{observed:,}"
    )

    if observed != expected:

        raise RuntimeError(
            f"Población {population}: "
            f"esperaba {expected:,}, "
            f"pero encontré {observed:,}."
        )

if N != 8239:

    raise RuntimeError(
        f"Esperaba 8,239 neuronas, "
        f"pero encontré {N:,}."
    )

print(
    f"  TOTAL: {N:,}"
)

print(
    "OK: poblaciones verificadas."
)


# ============================================================
# 8. CONECTIVIDAD
# ============================================================

pre_idx = (
    edges_df["Presynaptic_ID"]
    .map(id_to_idx)
    .to_numpy()
)

post_idx = (
    edges_df["Postsynaptic_ID"]
    .map(id_to_idx)
    .to_numpy()
)

if (
    np.isnan(pre_idx).any()
    or np.isnan(post_idx).any()
):

    raise RuntimeError(
        "Hay IDs sin mapear en la conectividad."
    )

pre_idx = pre_idx.astype(
    np.int32
)

post_idx = post_idx.astype(
    np.int32
)

weights = (
    edges_df[
        "Excitatory x Connectivity"
    ]
    .astype(np.float64)
    .to_numpy()
)

# Control adicional.
signed_check = (
    edges_df["Excitatory"]
    .astype(float)
    .to_numpy()
    *
    edges_df["Connectivity"]
    .astype(float)
    .to_numpy()
)

if not np.allclose(
    signed_check,
    weights,
    rtol=1e-10,
    atol=1e-12,
):

    raise RuntimeError(
        "Excitatory × Connectivity "
        "no coincide con "
        "'Excitatory x Connectivity'."
    )

print(
    "\nPares conectados: "
    f"{len(edges_df):,}"
)

print(
    "Contactos sinápticos: "
    f"{edges_df['Connectivity'].sum():,}"
)

print(
    "OK: conectividad firmada verificada."
)


# ============================================================
# 9. FUNCIÓN PARA OBTENER ÍNDICES DE UNA POBLACIÓN
# ============================================================

def indices_for_population(
    population,
):

    mask = (
        neurons_df["population"]
        .astype(str)
        .eq(population)
        .to_numpy()
    )

    return np.flatnonzero(
        mask
    ).astype(
        np.int32
    )


# ============================================================
# 10. IDENTIFICAR KC, APL Y MBON
# ============================================================

alpn_indices = (
    indices_for_population("ALPN")
)

kc_indices = (
    indices_for_population("KC")
)

apl_indices = (
    indices_for_population("APL")
)

mbon_indices = (
    indices_for_population("MBON")
)

kc_ids = (
    neuron_ids[
        kc_indices
    ]
)

apl_ids = (
    neuron_ids[
        apl_indices
    ]
)

mbon_ids = (
    neuron_ids[
        mbon_indices
    ]
)

n_kc = len(
    kc_indices
)


# ===== LOAD AND VERIFY P1-E GSTAR =====

if not P1E_REFERENCE_FILE.exists():
    raise FileNotFoundError(
        f"No existe la referencia P1-E: {P1E_REFERENCE_FILE}"
    )

with np.load(
    P1E_REFERENCE_FILE,
    allow_pickle=False,
) as p1e_ref:

    p1e_reference_kc_ids = np.asarray(
        p1e_ref["kc_root_ids"],
        dtype=np.int64,
    )

    P1E_GSTAR_MV = np.asarray(
        p1e_ref["Gstar_mV"],
        dtype=np.float64,
    )

    P1E_GAMMA_MV = float(
        np.asarray(
            p1e_ref["gamma_mV"]
        )
    )

    p1e_stored_sha256 = str(
        np.asarray(
            p1e_ref["sha256"]
        ).item()
    )

# Controles estrictos de integridad.
if n_kc != 5177:
    raise RuntimeError(
        f"P1-E esperaba 5177 KC; encontró {n_kc}."
    )

if P1E_GSTAR_MV.shape != (n_kc,):
    raise RuntimeError(
        "Shape G* incompatible: "
        f"{P1E_GSTAR_MV.shape}; esperaba {(n_kc,)}."
    )

if p1e_reference_kc_ids.shape != (n_kc,):
    raise RuntimeError(
        "Shape root_id de referencia incompatible: "
        f"{p1e_reference_kc_ids.shape}."
    )

if not np.all(
    np.isfinite(P1E_GSTAR_MV)
):
    raise RuntimeError(
        "G* contiene NaN o Inf."
    )

if (
    not np.isfinite(P1E_GAMMA_MV)
    or P1E_GAMMA_MV <= 0
):
    raise RuntimeError(
        f"gamma P1-E inválido: {P1E_GAMMA_MV}"
    )

if p1e_stored_sha256 != P1E_EXPECTED_SHA256:
    raise RuntimeError(
        "El SHA256 de la referencia P1-E "
        "no coincide con el valor congelado."
    )

if not np.array_equal(
    p1e_reference_kc_ids,
    kc_ids.astype(np.int64),
):
    raise RuntimeError(
        "El orden root_id de las KC no coincide "
        "con la referencia P1-E congelada."
    )

print()
print("P1-E frozen reference:")
print(
    f"  KC: {n_kc:,}"
)
print(
    f"  gamma: {P1E_GAMMA_MV:.9f} mV"
)
print(
    f"  SHA256: {p1e_stored_sha256}"
)
print(
    "  root_id order: OK"
)
print(
    "✓ REFERENCIA P1-E CARGADA Y VERIFICADA"
)

# ===== END LOAD AND VERIFY P1-E GSTAR =====

n_lesioned_fixed = int(
    round(
        n_kc
        * LESION_FRACTION
    )
)

if n_lesioned_fixed != 1294:

    raise RuntimeError(
        "Esperaba 1,294 KC lesionadas, "
        f"pero obtuve {n_lesioned_fixed:,}."
    )

print(
    f"\nKC totales: {n_kc:,}"
)

print(
    "KC lesionadas por identidad: "
    f"{n_lesioned_fixed:,}"
)


# ============================================================
# 11. IDENTIFICAR LOS OCHO ESTÍMULOS
# ============================================================

stimulus_indices_by_name = {}

print(
    "\nCanales ORN:"
)

for stimulus in STIMULI:

    stimulus_mask = (
        neurons_df["cell_type"]
        .astype(str)
        .eq(stimulus)
        .to_numpy()
    )

    stimulus_indices = (
        np.flatnonzero(
            stimulus_mask
        )
        .astype(np.int32)
    )

    if len(stimulus_indices) == 0:

        raise RuntimeError(
            f"No se encontraron neuronas "
            f"para {stimulus}."
        )

    stimulus_indices_by_name[
        stimulus
    ] = stimulus_indices

    print(
        f"  {stimulus}: "
        f"{len(stimulus_indices):,} ORN"
    )


# ============================================================
# 12. CREAR LAS 20 IDENTIDADES DE LESIÓN
# ============================================================

lesion_indices_by_seed = {}
lesion_ids_by_seed = {}

print(
    "\nGenerando identidades de lesión..."
)

for lesion_seed in LESION_SEEDS:

    lesion_rng = (
        np.random.default_rng(
            lesion_seed
        )
    )

    lesioned_indices = (
        lesion_rng.choice(
            kc_indices,
            size=n_lesioned_fixed,
            replace=False,
        )
        .astype(np.int32)
    )

    lesioned_ids = (
        neuron_ids[
            lesioned_indices
        ]
    )

    lesion_indices_by_seed[
        lesion_seed
    ] = lesioned_indices

    lesion_ids_by_seed[
        lesion_seed
    ] = lesioned_ids

    lesion_file = (
        LESION_IDS_DIR
        / (
            "lesion_seed"
            f"{lesion_seed:03d}.csv"
        )
    )

    if lesion_file.exists():

        previous = pd.read_csv(
            lesion_file
        )

        previous_ids = (
            previous["root_id"]
            .astype(np.int64)
            .to_numpy()
        )

        if (
            set(previous_ids.tolist())
            != set(lesioned_ids.tolist())
        ):

            raise RuntimeError(
                "La lesión previamente guardada "
                f"para seed {lesion_seed} "
                "no coincide con la lesión "
                "generada actualmente."
            )

    else:

        pd.DataFrame({
            "root_id":
                lesioned_ids
        }).to_csv(
            lesion_file,
            index=False,
        )

print(
    "OK: identidades de lesión "
    "fijadas y verificadas."
)


# ============================================================
# 13. MÉTRICAS ESTRUCTURALES DE CADA LESIÓN
# ============================================================

structural_metrics_by_lesion = {}

for lesion_seed in LESION_SEEDS:

    lesioned_indices = (
        lesion_indices_by_seed[
            lesion_seed
        ]
    )

    output_mask = np.isin(
        pre_idx,
        lesioned_indices,
    )

    affected_edges = (
        edges_df.loc[
            output_mask
        ]
        .copy()
    )

    affected_edges[
        "post_population"
    ] = (
        affected_edges[
            "Postsynaptic_ID"
        ]
        .map(
            population_by_id
        )
    )

    def count_edges_to(
        population,
    ):

        return int(
            affected_edges[
                "post_population"
            ]
            .astype(str)
            .eq(population)
            .sum()
        )

    def count_contacts_to(
        population,
    ):

        target_mask = (
            affected_edges[
                "post_population"
            ]
            .astype(str)
            .eq(population)
        )

        return int(
            affected_edges.loc[
                target_mask,
                "Connectivity",
            ].sum()
        )

    mbon_mask = (
        affected_edges[
            "post_population"
        ]
        .astype(str)
        .eq("MBON")
    )

    structural_metrics_by_lesion[
        lesion_seed
    ] = {

        "affected_connections":
            int(
                output_mask.sum()
            ),

        "affected_contacts":
            int(
                edges_df.loc[
                    output_mask,
                    "Connectivity",
                ].sum()
            ),

        "absolute_removed_weight":
            float(
                np.abs(
                    weights[
                        output_mask
                    ]
                ).sum()
            ),

        "signed_removed_weight":
            float(
                weights[
                    output_mask
                ].sum()
            ),

        "KC_MBON_connections_removed":
            count_edges_to("MBON"),

        "KC_APL_connections_removed":
            count_edges_to("APL"),

        "KC_KC_connections_removed":
            count_edges_to("KC"),

        "KC_MBON_contacts_removed":
            count_contacts_to("MBON"),

        "KC_APL_contacts_removed":
            count_contacts_to("APL"),

        "KC_KC_contacts_removed":
            count_contacts_to("KC"),

        "MBON_targets_affected":
            int(
                affected_edges.loc[
                    mbon_mask,
                    "Postsynaptic_ID",
                ].nunique()
            ),
    }


# ============================================================
# 14. GUARDAR CONFIGURACIÓN
# ============================================================

config = {

    "experiment":
        "P0_multistimulus",

    "stimuli":
        STIMULI,

    "lesion_fraction":
        LESION_FRACTION,

    "n_total_KC":
        n_kc,

    "n_lesioned_KC":
        n_lesioned_fixed,

    "lesion_seeds":
        LESION_SEEDS,

    "simulation_seeds":
        SIMULATION_SEEDS,

    "n_lesion_identities":
        len(LESION_SEEDS),

    "n_simulation_seeds":
        len(SIMULATION_SEEDS),

    "n_stimuli":
        len(STIMULI),

    "n_total_lesioned_runs":
        (
            len(LESION_SEEDS)
            * len(SIMULATION_SEEDS)
            * len(STIMULI)
        ),

    "lesion_definition":
        (
            "Outgoing synaptic weights "
            "of selected KC neurons "
            "were set to zero. "
            "Lesioned KCs remained able "
            "to receive input and spike."
        ),

    "same_lesion_identity_across_stimuli":
        True,

    "plasticity":
        False,

    "new_connections":
        False,

    "stimulus_rate_hz_per_selected_ORN":
        STIMULUS_RATE_HZ,

    "stimulus_rate_note":
        (
            "100 Hz is the Poisson input "
            "rate per selected ORN. "
            "It is not the resulting ORN "
            "firing rate and does not "
            "equalize total afferent input "
            "across ORN channels."
        ),

    "warmup_ms":
        WARMUP_MS,

    "baseline_ms":
        BASELINE_MS,

    "stimulus_ms":
        STIMULUS_MS,

    "post_ms":
        POST_MS,

    "V0_mV":
        -52,

    "Vreset_mV":
        -52,

    "Vthreshold_mV":
        -45,

    "tau_membrane_ms":
        20,

    "tau_synaptic_ms":
        5,

    "refractory_ms":
        2.2,

    "delay_ms":
        1.8,

    "W_SYN_mV":
        0.275,

    "F_POI":
        F_POI,

    "baseline_reference":
        (
            "Frozen intact multistimulus "
            "B_s from the previous "
            "20-seed calibration."
        ),

    "baseline_modified":
        False,

    "checkpointing":
        True,
}

with open(
    CONFIG_FILE,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        config,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# 15. FUNCIONES AUXILIARES
# ============================================================

def checkpoint_path(
    lesion_seed,
    stimulus,
    simulation_seed,
):

    return (
        CHECKPOINT_DIR
        / (
            f"L{lesion_seed:03d}_"
            f"{stimulus}_"
            f"S{simulation_seed:02d}.npz"
        )
    )


def population_rate(
    spike_indices,
    population_indices,
    duration_s,
):

    if len(
        population_indices
    ) == 0:

        return np.nan

    count = int(
        np.isin(
            spike_indices,
            population_indices,
        ).sum()
    )

    return (
        count
        / len(population_indices)
        / duration_s
    )


def individual_rates(
    spike_indices,
    population_indices,
    duration_s,
):

    counts = np.bincount(
        spike_indices,
        minlength=N,
    )

    return (
        counts[
            population_indices
        ]
        .astype(np.float64)
        / duration_s
    )


# ============================================================
# 16. FUNCIÓN PRINCIPAL DE SIMULACIÓN
# ============================================================

def run_condition(
    lesion_seed,
    stimulus,
    simulation_seed,
    p1e_current_weights_mV=None,
):

    lesioned_indices = (
        lesion_indices_by_seed[
            lesion_seed
        ]
    )

    stimulus_indices = (
        stimulus_indices_by_name[
            stimulus
        ]
    )

    # ========================================================
    # SEMILLA ESTOCÁSTICA
    # ========================================================

    seed(
        simulation_seed
    )

    np.random.seed(
        simulation_seed
    )

    # ========================================================
    # NEURONAS
    # ========================================================

    eqs = """
    dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
    dg/dt = -g / tau : volt (unless refractory)
    rfc : second
    """

    namespace = {
        "v_0": V_0,
        "v_rst": V_RST,
        "v_th": V_TH,
        "t_mbr": T_MBR,
        "tau": TAU,
    }

    brain = NeuronGroup(
        N=N,
        model=eqs,
        method="linear",
        threshold="v > v_th",
        reset=(
            "v = v_rst; "
            "g = 0 * mV"
        ),
        refractory="rfc",
        namespace=namespace,
    )

    brain.v = V_0
    brain.g = 0 * mV
    brain.rfc = T_RFC

    # ========================================================
    # CONECTOMA
    # ========================================================

    synapses = Synapses(
        brain,
        brain,
        model="w : volt",
        on_pre="g += w",
        delay=T_DLY,
    )

    synapses.connect(
        i=pre_idx,
        j=post_idx,
    )

    # ===== P1-I PLASTIC SYNAPSE DOMAIN =====
    #
    # w0: pesos conectómicos originales. Esta copia NO debe
    # cambiar durante la plasticidad.
    #
    # synaptic_weights: pesos efectivos de la simulación.
    # Sobre este vector se aplica primero la lesión y,
    # posteriormente, P1-I.
    # ========================================================

    synaptic_weights_w0 = (
        weights
        * W_SYN
    ).copy()

    synaptic_weights = (
        synaptic_weights_w0.copy()
    )

    # Dominio estructural permitido para P1-I:
    #
    #   KC superviviente -> APL
    #   peso original positivo
    #
    # No se crean aristas nuevas.
    # No se modifican conexiones APL -> KC.
    # Las KC lesionadas quedan excluidas como presinápticas
    # porque su output funcional permanece silenciado.

    p1i_kc_to_apl_positive_mask = (
        np.isin(
            pre_idx,
            kc_indices,
        )
        &
        np.isin(
            post_idx,
            apl_indices,
        )
        &
        (
            np.asarray(
                synaptic_weights_w0 / mV,
                dtype=np.float64,
            )
            > 0.0
        )
    )

    # ========================================================
    # APLICAR LESIÓN
    #
    # Igual que en P0/P1-E:
    # pesos SALIENTES de KC lesionadas = 0.
    #
    # Las KC lesionadas siguen recibiendo input y pueden
    # seguir disparando internamente.
    # ========================================================

    lesioned_output_mask = np.isin(
        pre_idx,
        lesioned_indices,
    )

    # En P1-I la KC es PRESINÁPTICA.
    # Solo outputs de KC supervivientes pueden modificarse.
    p1i_surviving_pre_mask = (
        ~np.isin(
            pre_idx,
            lesioned_indices,
        )
    )

    p1i_plastic_mask = (
        p1i_kc_to_apl_positive_mask
        &
        p1i_surviving_pre_mask
    )

    synaptic_weights[
        lesioned_output_mask
    ] = 0 * mV

    # ===== P1-E EXTERNAL WEIGHT STATE =====
    #
    # Si se proporciona un estado plástico externo, únicamente
    # las sinapsis autorizadas por p1i_plastic_mask reciben
    # esos valores. El resto conserva exactamente el peso
    # conectómico/lesionado correspondiente.
    #
    # p1e_current_weights_mV está expresado en mV y tiene una
    # entrada por conexión del conectoma.

    if p1e_current_weights_mV is not None:

        p1e_current_weights_mV = np.asarray(
            p1e_current_weights_mV,
            dtype=np.float64,
        )

        if p1e_current_weights_mV.shape != (
            len(pre_idx),
        ):
            raise RuntimeError(
                "Shape de pesos P1-E incompatible: "
                f"{p1e_current_weights_mV.shape}; "
                f"esperaba {(len(pre_idx),)}."
            )

        if not np.all(
            np.isfinite(
                p1e_current_weights_mV
            )
        ):
            raise RuntimeError(
                "Pesos P1-E contienen NaN/Inf."
            )

        synaptic_weights_mV = np.asarray(
            synaptic_weights / mV,
            dtype=np.float64,
        ).copy()

        synaptic_weights_mV[
            p1i_plastic_mask
        ] = (
            p1e_current_weights_mV[
                p1i_plastic_mask
            ]
        )

        synaptic_weights = (
            synaptic_weights_mV
            * mV
        )

        # La lesión tiene prioridad absoluta:
        # ningún estado plástico puede resucitar outputs
        # de KC lesionadas.
        synaptic_weights[
            lesioned_output_mask
        ] = 0 * mV

    synapses.w = (
        synaptic_weights
    )

    # ========================================================
    # INPUT POISSON
    # ========================================================

    n_input = len(
        stimulus_indices
    )

    input_group = PoissonGroup(
        n_input,
        rates=(
            np.zeros(
                n_input
            )
            * Hz
        ),
    )

    input_synapses = Synapses(
        input_group,
        brain,
        model="w : volt",
        on_pre="g += w",
    )

    input_synapses.connect(
        i=np.arange(
            n_input
        ),
        j=stimulus_indices,
    )

    input_synapses.w = (
        W_SYN
        * F_POI
    )

    # Igual que P0 original.
    brain.rfc[
        stimulus_indices
    ] = 0 * ms

    # ========================================================
    # MONITOR
    # ========================================================

    spike_monitor = SpikeMonitor(
        brain,
        record=True,
    )

    # ========================================================
    # RED
    # ========================================================

    net = Network(
        brain,
        synapses,
        input_group,
        input_synapses,
        spike_monitor,
    )

    # ========================================================
    # SIMULACIÓN
    # ========================================================

    input_group.rates = (
        0 * Hz
    )

    net.run(
        WARMUP_MS
        * ms
    )

    net.run(
        BASELINE_MS
        * ms
    )

    stimulus_start_ms = (
        WARMUP_MS
        + BASELINE_MS
    )

    # ========================================================
    # P1-E2: MONITOR POSTSINÁPTICO KC
    #
    # Se crea DESPUÉS de warmup + baseline.
    # Registra únicamente durante los 500 ms de estímulo.
    # ========================================================

    kc_state_monitor = StateMonitor(
        brain,
        variables=["g", "v"],
        record=kc_indices,
        dt=defaultclock.dt,
    )

    net.add(
        kc_state_monitor
    )

    input_group.rates = (
        STIMULUS_RATE_HZ
        * Hz
    )

    net.run(
        STIMULUS_MS
        * ms
    )

    # No registrar durante POST.
    net.remove(
        kc_state_monitor
    )

    stimulus_end_ms = (
        stimulus_start_ms
        + STIMULUS_MS
    )

    input_group.rates = (
        0 * Hz
    )

    net.run(
        POST_MS
        * ms
    )

    # ========================================================
    # EXTRAER SPIKES
    # ========================================================

    spike_indices = np.asarray(
        spike_monitor.i,
        dtype=np.int32,
    )

    spike_times_ms = np.asarray(
        spike_monitor.t / ms,
        dtype=np.float64,
    )

    stimulus_mask = (
        (
            spike_times_ms
            >= stimulus_start_ms
        )
        &
        (
            spike_times_ms
            < stimulus_end_ms
        )
    )

    stimulus_spike_indices = (
        spike_indices[
            stimulus_mask
        ]
    )

    duration_s = (
        STIMULUS_MS
        / 1000.0
    )

    # ========================================================
    # TASAS POBLACIONALES
    # ========================================================

    rates = {}

    for population in [
        "ORN",
        "ALPN",
        "KC",
        "APL",
        "MBON",
    ]:

        population_indices = (
            indices_for_population(
                population
            )
        )

        rates[
            population
        ] = population_rate(
            stimulus_spike_indices,
            population_indices,
            duration_s,
        )

    # ========================================================
    # KC LESIONADAS VS SUPERVIVIENTES
    # ========================================================

    surviving_kc_indices = (
        np.setdiff1d(
            kc_indices,
            lesioned_indices,
            assume_unique=False,
        )
        .astype(np.int32)
    )

    lesioned_kc_rate = (
        population_rate(
            stimulus_spike_indices,
            lesioned_indices,
            duration_s,
        )
    )

    surviving_kc_rate = (
        population_rate(
            stimulus_spike_indices,
            surviving_kc_indices,
            duration_s,
        )
    )

    # ========================================================
    # VECTORES INDIVIDUALES
    # ========================================================

    kc_rates = individual_rates(
        stimulus_spike_indices,
        kc_indices,
        duration_s,
    )

    apl_rates = individual_rates(
        stimulus_spike_indices,
        apl_indices,
        duration_s,
    )

    mbon_rates = individual_rates(
        stimulus_spike_indices,
        mbon_indices,
        duration_s,
    )

    kc_active = (
        kc_rates > 0
    )

    # ========================================================
    # P1-E2: SENSORES POSTSINÁPTICOS
    # ========================================================

    kc_g_mV = np.asarray(
        kc_state_monitor.g / mV,
        dtype=np.float64,
    )

    kc_v_mV = np.asarray(
        kc_state_monitor.v / mV,
        dtype=np.float64,
    )

    if kc_g_mV.shape[0] != len(kc_indices):
        raise RuntimeError(
            f"Monitor g: {kc_g_mV.shape[0]} KC; "
            f"esperadas {len(kc_indices)}."
        )

    if kc_v_mV.shape != kc_g_mV.shape:
        raise RuntimeError(
            "Shapes de g y v no coinciden."
        )

    if kc_g_mV.shape[1] == 0:
        raise RuntimeError(
            "StateMonitor no registró muestras."
        )

    # --------------------------------------------------------
    # SENSOR PRIMARIO
    #
    # G_i = <g_i(t)> durante el estímulo.
    # --------------------------------------------------------

    kc_G_mean_mV = np.mean(
        kc_g_mV,
        axis=1,
    )

    # --------------------------------------------------------
    # SENSOR SECUNDARIO
    #
    # D_i = <v_i(t) - v0> durante el estímulo.
    # --------------------------------------------------------

    kc_D_mean_mV = np.mean(
        kc_v_mV
        - float(V_0 / mV),
        axis=1,
    )

    G_nonzero_fraction = float(
        np.mean(
            np.abs(kc_G_mean_mV) > 1e-12
        )
    )

    D_nonzero_fraction = float(
        np.mean(
            np.abs(kc_D_mean_mV) > 1e-12
        )
    )

    # ========================================================
    # RESUMEN DE LA CONDICIÓN
    # ========================================================

    structural = (
        structural_metrics_by_lesion[
            lesion_seed
        ]
    )

    metadata = {

        "condition":
            "lesion",

        "lesion_seed":
            int(
                lesion_seed
            ),

        "simulation_seed":
            int(
                simulation_seed
            ),

        "stimulus":
            stimulus,

        "lesion_fraction":
            float(
                LESION_FRACTION
            ),

        "n_total_KC":
            int(
                n_kc
            ),

        "n_lesioned_KC":
            int(
                len(
                    lesioned_indices
                )
            ),

        "n_surviving_KC":
            int(
                len(
                    surviving_kc_indices
                )
            ),

        "n_stimulus_ORN":
            int(
                len(
                    stimulus_indices
                )
            ),

        "ORN_rate_hz":
            float(
                rates["ORN"]
            ),

        "ALPN_rate_hz":
            float(
                rates["ALPN"]
            ),

        "KC_rate_hz":
            float(
                rates["KC"]
            ),

        "KC_lesioned_rate_hz":
            float(
                lesioned_kc_rate
            ),

        "KC_surviving_rate_hz":
            float(
                surviving_kc_rate
            ),

        "KC_active_count":
            int(
                kc_active.sum()
            ),

        "KC_active_fraction":
            float(
                kc_active.mean()
            ),

        "APL_rate_hz":
            float(
                rates["APL"]
            ),

        "MBON_rate_hz":
            float(
                rates["MBON"]
            ),

        "stimulus_spike_count":
            int(
                len(
                    stimulus_spike_indices
                )
            ),

        "P1E2_state_samples":
            int(
                kc_g_mV.shape[1]
            ),

        "P1E2_G_nonzero_fraction":
            float(
                G_nonzero_fraction
            ),

        "P1E2_D_nonzero_fraction":
            float(
                D_nonzero_fraction
            ),

        **structural,
    }

    return (
        metadata,
        kc_rates,
        apl_rates,
        mbon_rates,
        kc_G_mean_mV,
        kc_D_mean_mV,
    )


# ============================================================
# LEGACY P0/P1-E2 EXECUTION — PRESERVED BUT DISABLED
# ============================================================
if False:  # P1-E: legacy execution disabled
    # ============================================================
    # 17. EJECUTAR EXPERIMENTO
    # ============================================================

    total_runs = (
        len(LESION_SEEDS)
        * len(STIMULI)
        * len(SIMULATION_SEEDS)
    )

    print(
        "\n" + "=" * 72
    )

    print(
        " EJECUCIÓN P0 MULTIESTÍMULO"
    )

    print(
        "=" * 72
    )

    print(
        f"\nTotal de simulaciones: "
        f"{total_runs:,}"
    )

    print(
        "20 lesiones × "
        "8 estímulos × "
        "5 semillas = 800"
    )

    print(
        "\nPlasticidad: NO"
    )

    print(
        "Si una condición ya tiene checkpoint, "
        "será omitida.\n"
    )


    completed_now = 0
    skipped_existing = 0


    for lesion_seed in LESION_SEEDS:

        print(
            "\n" + "#" * 72
        )

        print(
            f"LESIÓN "
            f"{lesion_seed:02d}/"
            f"{len(LESION_SEEDS)}"
        )

        print(
            "#" * 72
        )

        for stimulus in STIMULI:

            for simulation_seed in SIMULATION_SEEDS:

                checkpoint = (
                    checkpoint_path(
                        lesion_seed,
                        stimulus,
                        simulation_seed,
                    )
                )

                if checkpoint.exists():

                    skipped_existing += 1

                    print(
                        f"[SKIP] "
                        f"L={lesion_seed:02d} | "
                        f"{stimulus} | "
                        f"S={simulation_seed:02d}"
                    )

                    continue

                print(
                    f"[RUN ] "
                    f"L={lesion_seed:02d} | "
                    f"{stimulus} | "
                    f"S={simulation_seed:02d}"
                )

                (
                    metadata,
                    kc_rates,
                    apl_rates,
                    mbon_rates,
                    kc_G_mean_mV,
                    kc_D_mean_mV,
                ) = run_condition(
                    lesion_seed,
                    stimulus,
                    simulation_seed,
                )

                # =================================================
                # CHECKPOINT ATÓMICO
                # =================================================

                temporary_checkpoint = (
                    checkpoint.with_name(
                        checkpoint.name
                        + ".tmp.npz"
                    )
                )

                np.savez_compressed(
                    temporary_checkpoint,

                    metadata_json=(
                        json.dumps(
                            metadata,
                            ensure_ascii=False,
                        )
                    ),

                    kc_rates=
                        kc_rates,

                    apl_rates=
                        apl_rates,

                    mbon_rates=
                        mbon_rates,

                    kc_G_mean_mV=
                        kc_G_mean_mV,

                    kc_D_mean_mV=
                        kc_D_mean_mV,
                )

                temporary_checkpoint.replace(
                    checkpoint
                )

                completed_now += 1

                print(
                    "       "
                    f"KC={metadata['KC_rate_hz']:.6f} Hz | "
                    f"KC surv="
                    f"{metadata['KC_surviving_rate_hz']:.6f} Hz | "
                    f"APL={metadata['APL_rate_hz']:.6f} Hz | "
                    f"MBON={metadata['MBON_rate_hz']:.6f} Hz"
                )

                n_checkpoints = len(
                    list(
                        CHECKPOINT_DIR.glob(
                            "*.npz"
                        )
                    )
                )

                progress = {

                    "completed_checkpoints":
                        n_checkpoints,

                    "expected_total":
                        total_runs,

                    "last_lesion_seed":
                        lesion_seed,

                    "last_stimulus":
                        stimulus,

                    "last_simulation_seed":
                        simulation_seed,
                }

                pd.DataFrame(
                    [progress]
                ).to_csv(
                    PROGRESS_FILE,
                    index=False,
                )


    # ============================================================
    # 18. VERIFICAR CHECKPOINTS
    # ============================================================

    expected_paths = []

    for lesion_seed in LESION_SEEDS:

        for stimulus in STIMULI:

            for simulation_seed in SIMULATION_SEEDS:

                expected_paths.append(
                    checkpoint_path(
                        lesion_seed,
                        stimulus,
                        simulation_seed,
                    )
                )


    missing = [
        path
        for path in expected_paths
        if not path.exists()
    ]


    print(
        "\n" + "=" * 72
    )

    print(
        " CONTROL DE CHECKPOINTS"
    )

    print(
        "=" * 72
    )

    print(
        "Nuevas simulaciones completadas: "
        f"{completed_now:,}"
    )

    print(
        "Checkpoints existentes omitidos: "
        f"{skipped_existing:,}"
    )

    print(
        "Checkpoints esperados: "
        f"{len(expected_paths):,}"
    )

    print(
        "Checkpoints faltantes: "
        f"{len(missing):,}"
    )


    if len(missing) > 0:

        print(
            "\nEl experimento quedó "
            "parcialmente completado."
        )

        print(
            "Vuelve a ejecutar exactamente "
            "el mismo script."
        )

        print(
            "Las condiciones ya terminadas "
            "serán omitidas."
        )

        raise RuntimeError(
            f"Faltan {len(missing)} "
            "condiciones."
        )


    # ============================================================
    # 19. CONSOLIDAR RESULTADOS
    # ============================================================

    print(
        "\n" + "=" * 72
    )

    print(
        " CONSOLIDANDO RESULTADOS"
    )

    print(
        "=" * 72
    )


    summary_rows = []

    kc_frames = []
    apl_frames = []
    mbon_frames = []


    for lesion_seed in LESION_SEEDS:

        lesioned_id_set = set(
            lesion_ids_by_seed[
                lesion_seed
            ].tolist()
        )

        for stimulus in STIMULI:

            for simulation_seed in SIMULATION_SEEDS:

                checkpoint = (
                    checkpoint_path(
                        lesion_seed,
                        stimulus,
                        simulation_seed,
                    )
                )

                with np.load(
                    checkpoint,
                    allow_pickle=False,
                ) as data:

                    metadata = (
                        json.loads(
                            str(
                                data[
                                    "metadata_json"
                                ]
                            )
                        )
                    )

                    kc_rates = (
                        data[
                            "kc_rates"
                        ]
                        .astype(
                            np.float64
                        )
                    )

                    apl_rates = (
                        data[
                            "apl_rates"
                        ]
                        .astype(
                            np.float64
                        )
                    )

                    mbon_rates = (
                        data[
                            "mbon_rates"
                        ]
                        .astype(
                            np.float64
                        )
                    )


                if len(kc_rates) != len(kc_ids):

                    raise RuntimeError(
                        "Longitud KC inválida "
                        f"en {checkpoint.name}."
                    )


                if len(apl_rates) != len(apl_ids):

                    raise RuntimeError(
                        "Longitud APL inválida "
                        f"en {checkpoint.name}."
                    )


                if len(mbon_rates) != len(mbon_ids):

                    raise RuntimeError(
                        "Longitud MBON inválida "
                        f"en {checkpoint.name}."
                    )


                summary_rows.append(
                    metadata
                )


                kc_frames.append(

                    pd.DataFrame({

                        "condition":
                            "lesion",

                        "lesion_seed":
                            lesion_seed,

                        "simulation_seed":
                            simulation_seed,

                        "stimulus":
                            stimulus,

                        "root_id":
                            kc_ids,

                        "rate_hz":
                            kc_rates,

                        "active":
                            kc_rates > 0,

                        "lesioned":
                            [
                                int(root_id)
                                in lesioned_id_set

                                for root_id
                                in kc_ids
                            ],
                    })
                )


                apl_frames.append(

                    pd.DataFrame({

                        "condition":
                            "lesion",

                        "lesion_seed":
                            lesion_seed,

                        "simulation_seed":
                            simulation_seed,

                        "stimulus":
                            stimulus,

                        "root_id":
                            apl_ids,

                        "rate_hz":
                            apl_rates,
                    })
                )


                mbon_frames.append(

                    pd.DataFrame({

                        "condition":
                            "lesion",

                        "lesion_seed":
                            lesion_seed,

                        "simulation_seed":
                            simulation_seed,

                        "stimulus":
                            stimulus,

                        "root_id":
                            mbon_ids,

                        "rate_hz":
                            mbon_rates,
                    })
                )


    # ============================================================
    # 20. TABLA DE RESUMEN
    # ============================================================

    summary_df = pd.DataFrame(
        summary_rows
    )

    summary_df = (
        summary_df
        .sort_values(
            [
                "lesion_seed",
                "stimulus",
                "simulation_seed",
            ]
        )
        .reset_index(
            drop=True
        )
    )


    if len(summary_df) != total_runs:

        raise RuntimeError(
            "Resumen final: esperaba "
            f"{total_runs:,} filas, "
            f"pero obtuve "
            f"{len(summary_df):,}."
        )


    if summary_df.duplicated(
        subset=[
            "lesion_seed",
            "stimulus",
            "simulation_seed",
        ]
    ).any():

        raise RuntimeError(
            "Hay condiciones duplicadas "
            "en el resumen."
        )


    # ============================================================
    # 21. TABLAS LONG
    # ============================================================

    kc_long_df = pd.concat(
        kc_frames,
        ignore_index=True,
    )

    apl_long_df = pd.concat(
        apl_frames,
        ignore_index=True,
    )

    mbon_long_df = pd.concat(
        mbon_frames,
        ignore_index=True,
    )


    expected_kc_rows = (
        total_runs
        * len(kc_ids)
    )

    expected_apl_rows = (
        total_runs
        * len(apl_ids)
    )

    expected_mbon_rows = (
        total_runs
        * len(mbon_ids)
    )


    print(
        "\nControl de integridad:"
    )

    print(
        f"  Resumen: "
        f"{len(summary_df):,} / "
        f"{total_runs:,}"
    )

    print(
        f"  KC: "
        f"{len(kc_long_df):,} / "
        f"{expected_kc_rows:,}"
    )

    print(
        f"  APL: "
        f"{len(apl_long_df):,} / "
        f"{expected_apl_rows:,}"
    )

    print(
        f"  MBON: "
        f"{len(mbon_long_df):,} / "
        f"{expected_mbon_rows:,}"
    )


    if len(kc_long_df) != expected_kc_rows:

        raise RuntimeError(
            "Número incorrecto "
            "de filas KC."
        )


    if len(apl_long_df) != expected_apl_rows:

        raise RuntimeError(
            "Número incorrecto "
            "de filas APL."
        )


    if len(mbon_long_df) != expected_mbon_rows:

        raise RuntimeError(
            "Número incorrecto "
            "de filas MBON."
        )


    # ============================================================
    # 22. GUARDAR ARCHIVOS FINALES
    # ============================================================

    print(
        "\nGuardando P0_run_summary.csv..."
    )

    summary_df.to_csv(
        RUN_SUMMARY_FILE,
        index=False,
    )


    print(
        "Guardando "
        "P0_KC_activity_long.csv..."
    )

    kc_long_df.to_csv(
        KC_LONG_FILE,
        index=False,
    )


    print(
        "Guardando "
        "P0_APL_activity_long.csv..."
    )

    apl_long_df.to_csv(
        APL_LONG_FILE,
        index=False,
    )


    print(
        "Guardando "
        "P0_MBON_activity_long.csv..."
    )

    mbon_long_df.to_csv(
        MBON_LONG_FILE,
        index=False,
    )


    # ============================================================
    # 23. RESUMEN DESCRIPTIVO
    # ============================================================

    descriptive = (

        summary_df

        .groupby(
            "stimulus"
        )

        .agg(

            n_runs=(
                "simulation_seed",
                "size",
            ),

            KC_rate_mean_hz=(
                "KC_rate_hz",
                "mean",
            ),

            KC_rate_sd_hz=(
                "KC_rate_hz",
                "std",
            ),

            KC_surviving_mean_hz=(
                "KC_surviving_rate_hz",
                "mean",
            ),

            KC_surviving_sd_hz=(
                "KC_surviving_rate_hz",
                "std",
            ),

            APL_rate_mean_hz=(
                "APL_rate_hz",
                "mean",
            ),

            APL_rate_sd_hz=(
                "APL_rate_hz",
                "std",
            ),

            MBON_rate_mean_hz=(
                "MBON_rate_hz",
                "mean",
            ),

            MBON_rate_sd_hz=(
                "MBON_rate_hz",
                "std",
            ),
        )

        .reset_index()
    )


    descriptive.to_csv(
        DESCRIPTIVE_FILE,
        index=False,
    )


    # ============================================================
    # 24. MOSTRAR RESUMEN
    # ============================================================

    print(
        "\n" + "=" * 72
    )

    print(
        " P0 MULTIESTÍMULO COMPLETADO"
    )

    print(
        "=" * 72
    )


    print(
        f"\nSimulaciones: "
        f"{len(summary_df):,}"
    )

    print(
        f"KC long: "
        f"{len(kc_long_df):,} filas"
    )

    print(
        f"APL long: "
        f"{len(apl_long_df):,} filas"
    )

    print(
        f"MBON long: "
        f"{len(mbon_long_df):,} filas"
    )


    print(
        "\nResumen descriptivo:"
    )

    print(
        descriptive.to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.6f}"
            ),
        )
    )


    print(
        "\nArchivos:"
    )

    print(
        f"  {RUN_SUMMARY_FILE}"
    )

    print(
        f"  {KC_LONG_FILE}"
    )

    print(
        f"  {APL_LONG_FILE}"
    )

    print(
        f"  {MBON_LONG_FILE}"
    )

    print(
        f"  {DESCRIPTIVE_FILE}"
    )

    print(
        f"  {CONFIG_FILE}"
    )

    print(
        f"  {LESION_IDS_DIR}"
    )

    print(
        f"  {CHECKPOINT_DIR}"
    )


    print(
        "\nPlasticidad: NO aplicada."
    )

    print(
        "Lesión: pesos sinápticos "
        "salientes de las 1,294 KC "
        "seleccionadas = 0."
    )

    print(
        "Las KC lesionadas siguen "
        "recibiendo input y pueden disparar."
    )

    print(
        "B_s intacto: NO modificado "
        "ni recalibrado."
    )


# ============================================================
# P1-E TECHNICAL PILOT CONTROLLER
# ============================================================
#
# Piloto de implementación, NO análisis de recuperación.
#
# Objetivo:
#   verificar mecánicamente la regla P1-E congelada.
#
# No utiliza MBON, recovery, ni ninguna variable global de
# resultado para calcular o seleccionar actualizaciones.
# ============================================================

P1E_ADAPTATION_STIMULI = (
    "ORN_DM1",
    "ORN_DL5",
    "ORN_DA2",
    "ORN_DA1",
    "ORN_DC4",
    "ORN_DP1l",
    "ORN_VM4",
    "ORN_VL2a",
)

P1E_PILOT_LESION_SEED = 1
P1E_PILOT_SIMULATION_SEED = 1

# eta=0 = control nulo.
# eta=0.03 = comprobación mecánica de actualización.
# No se selecciona por recuperación.
P1E_PILOT_ETAS = (
    0.0,
    0.03,
)


def p1e_initial_weight_state_mV():
    """
    Estado conectómico original en mV, antes de lesión
    y antes de plasticidad.
    """

    w0_mV = np.asarray(
        (weights * W_SYN) / mV,
        dtype=np.float64,
    ).copy()

    if w0_mV.shape != (len(pre_idx),):
        raise RuntimeError(
            "P1-E: shape de w0 incompatible."
        )

    if not np.all(np.isfinite(w0_mV)):
        raise RuntimeError(
            "P1-E: w0 contiene NaN/Inf."
        )

    return w0_mV



def p1i_lesion_and_plastic_masks(lesion_seed, w0_mV):
    """
    P1-I: únicamente conexiones excitatorias existentes
    de KC supervivientes hacia APL.
    """

    lesioned_indices_local = lesion_indices_by_seed[
        lesion_seed
    ].copy()

    lesioned_output_mask_local = np.isin(
        pre_idx,
        lesioned_indices_local,
    )

    kc_to_apl_positive = (
        np.isin(pre_idx, kc_indices)
        & np.isin(post_idx, apl_indices)
        & (w0_mV > 0.0)
    )

    surviving_pre = ~np.isin(
        pre_idx,
        lesioned_indices_local,
    )

    plastic_mask = (
        kc_to_apl_positive
        & surviving_pre
    )

    if np.any(plastic_mask & lesioned_output_mask_local):
        raise RuntimeError(
            "P1-I autorizó una salida de KC lesionada."
        )

    if np.any(
        plastic_mask
        & ~np.isin(post_idx, apl_indices)
    ):
        raise RuntimeError(
            "P1-I autorizó una conexión fuera de APL."
        )

    return (
        lesioned_indices_local,
        lesioned_output_mask_local,
        plastic_mask,
    )



# Referencia P1-I congelada.
# Drive KC->APL ponderado por conectividad.
P1I_ISTAR = np.array(
    [2209.8125, 8254.075],
    dtype=np.float64,
)


def p1i_apl_input_drive(kc_rates, current_weights_mV,
                        lesioned_indices_local):
    """Entrada KC→APL ponderada por conectividad."""
    rates = np.asarray(kc_rates, dtype=float)
    w = np.asarray(current_weights_mV, dtype=float)

    if rates.shape != (n_kc,) or w.shape != (len(pre_idx),):
        raise RuntimeError("Dimensiones P1-I incorrectas.")
    if not np.isfinite(rates).all() or not np.isfinite(w).all():
        raise RuntimeError("Valores no finitos.")
    if len(apl_indices) != 2:
        raise RuntimeError("Se esperaban dos APL.")

    kc_map = np.full(len(neuron_ids), -1, dtype=int)
    kc_map[kc_indices] = np.arange(n_kc)

    apl_map = np.full(len(neuron_ids), -1, dtype=int)
    apl_map[apl_indices] = np.arange(2)

    mask = (
        np.isin(pre_idx, kc_indices)
        & np.isin(post_idx, apl_indices)
        & (weights > 0)
        & ~np.isin(pre_idx, lesioned_indices_local)
    )

    positions = np.flatnonzero(mask)
    source = kc_map[pre_idx[positions]]
    target = apl_map[post_idx[positions]]

    if (source < 0).any() or (target < 0).any():
        raise RuntimeError("Mapeo KC/APL incorrecto.")

    scale = float(W_SYN / mV)
    if scale <= 0:
        raise RuntimeError("Escala sináptica inválida.")

    drive = np.bincount(
        target,
        weights=rates[source] * w[positions] / scale,
        minlength=2,
    ).astype(float)

    if drive.shape != (2,) or not np.isfinite(drive).all():
        raise RuntimeError("Drive APL inválido.")

    return drive


def p1i_apply_update(current_weights_mV, w0_mV,
                     plastic_mask, Ibar, eta):
    """Actualiza exclusivamente las conexiones KC→APL permitidas."""
    current = np.asarray(current_weights_mV, dtype=float)
    original = np.asarray(w0_mV, dtype=float)
    mask = np.asarray(plastic_mask, dtype=bool)
    Ibar = np.asarray(Ibar, dtype=float)

    if current.shape != original.shape or mask.shape != current.shape:
        raise RuntimeError("Dimensiones de pesos incompatibles.")
    if Ibar.shape != (2,):
        raise RuntimeError("Se esperaban dos entradas APL.")
    if not np.isfinite(current).all() or not np.isfinite(Ibar).all():
        raise RuntimeError("Valores no finitos en P1-I.")

    if not np.isfinite(original).all():
        raise RuntimeError("Pesos originales no finitos.")
    if not np.isfinite(eta) or eta < 0:
        raise RuntimeError("Tasa de plasticidad inválida.")
    if np.any(original[mask] <= 0):
        raise RuntimeError("Conexión plástica no positiva.")

    epsilon = np.clip(
        (P1I_ISTAR - Ibar) / P1I_ISTAR,
        -1.0, 1.0
    )

    apl_map = np.full(len(neuron_ids), -1, dtype=int)
    apl_map[apl_indices] = np.arange(2)

    positions = np.flatnonzero(mask)
    targets = apl_map[post_idx[positions]]

    if np.any(targets < 0):
        raise RuntimeError("Destino plástico fuera de APL.")

    updated = current.copy()

    if positions.size:
        factors = np.exp(eta * epsilon[targets])
        updated[positions] = np.clip(
            current[positions] * factors,
            0.5 * original[positions],
            2.0 * original[positions]
        )

    if not np.isfinite(updated).all():
        raise RuntimeError("Pesos actualizados no finitos.")

    if not np.array_equal(updated[~mask], current[~mask]):
        raise RuntimeError("Se modificaron conexiones prohibidas.")

    return updated, epsilon


def p1e_apply_update(
    current_weights_mV,
    w0_mV,
    plastic_mask,
    Gbar_mV,
    eta,
):
    """
    Regla P1-E congelada:

        epsilon_i =
            clip(
                (G*_i - Gbar_i)
                / (|G*_i| + gamma),
                -1,
                +1
            )

        w_ji(c+1) =
            clip(
                w_ji(c) * exp(eta * epsilon_i),
                0.5*w0_ji,
                2.0*w0_ji
            )

    Solo se actualizan conexiones incluidas en plastic_mask.
    """

    current_weights_mV = np.asarray(
        current_weights_mV,
        dtype=np.float64,
    ).copy()

    Gbar_mV = np.asarray(
        Gbar_mV,
        dtype=np.float64,
    )

    if Gbar_mV.shape != (n_kc,):
        raise RuntimeError(
            "P1-E: Gbar tiene shape incompatible."
        )

    epsilon = (
        (
            P1E_GSTAR_MV
            - Gbar_mV
        )
        /
        (
            np.abs(
                P1E_GSTAR_MV
            )
            + P1E_GAMMA_MV
        )
    )

    epsilon = np.clip(
        epsilon,
        -P1E_ERROR_CLIP,
        P1E_ERROR_CLIP,
    )

    if not np.all(np.isfinite(epsilon)):
        raise RuntimeError(
            "P1-E: epsilon contiene NaN/Inf."
        )

    # Cada conexión hereda el error de su KC postsináptica.
    #
    # post_idx contiene índices globales de neuronas.
    # Necesitamos convertir cada KC global a posición 0..5176
    # en el vector G*.
    kc_global_to_local = np.full(
        len(neuron_ids),
        -1,
        dtype=np.int32,
    )

    kc_global_to_local[
        kc_indices
    ] = np.arange(
        n_kc,
        dtype=np.int32,
    )

    plastic_positions = np.flatnonzero(
        plastic_mask
    )

    post_global = post_idx[
        plastic_positions
    ]

    post_kc_local = kc_global_to_local[
        post_global
    ]

    if np.any(post_kc_local < 0):
        raise RuntimeError(
            "P1-E intentó actualizar una conexión "
            "cuyo target no es KC."
        )

    old = current_weights_mV[
        plastic_positions
    ]

    original = w0_mV[
        plastic_positions
    ]

    factor = np.exp(
        eta
        * epsilon[
            post_kc_local
        ]
    )

    proposed = (
        old
        * factor
    )

    lower = (
        P1E_WEIGHT_MIN_FACTOR
        * original
    )

    upper = (
        P1E_WEIGHT_MAX_FACTOR
        * original
    )

    updated = np.clip(
        proposed,
        lower,
        upper,
    )

    new_weights_mV = (
        current_weights_mV.copy()
    )

    new_weights_mV[
        plastic_positions
    ] = updated

    return (
        new_weights_mV,
        epsilon,
    )


def run_p1e_technical_pilot():
    """
    Piloto exclusivamente técnico.

    NO evalúa recuperación.
    NO selecciona eta.
    NO utiliza MBON como señal plástica.
    """

    print()
    print("=" * 72)
    print(" P1-E TECHNICAL PILOT")
    print("=" * 72)

    w0_mV = (
        p1e_initial_weight_state_mV()
    )

    (
        lesioned_indices_local,
        lesioned_output_mask_local,
        plastic_mask,
    ) = p1i_lesion_and_plastic_masks(
        P1E_PILOT_LESION_SEED,
        w0_mV,
    )

    print(
        f"KC lesionadas: "
        f"{len(lesioned_indices_local):,}"
    )

    print(
        f"Sinapsis P1-E autorizadas: "
        f"{int(plastic_mask.sum()):,}"
    )

    if len(lesioned_indices_local) != 1294:
        raise RuntimeError(
            "P1-E: número de KC lesionadas incorrecto."
        )

    if plastic_mask.sum() == 0:
        raise RuntimeError(
            "P1-E: dominio plástico vacío."
        )

    # --------------------------------------------------------
    # Ejecutar cada eta independientemente desde w0.
    # --------------------------------------------------------

    for eta in P1E_PILOT_ETAS:

        print()
        print("-" * 72)
        print(
            f"eta = {eta:.3f}"
        )
        print("-" * 72)

        current_weights_mV = (
            w0_mV.copy()
        )

        # La lesión no se almacena dentro de current_weights:
        # run_condition() la reaplica con prioridad absoluta.
        #
        # current_weights contiene el estado plástico de las
        # conexiones autorizadas.

        for cycle in range(
            P1E_N_CYCLES
        ):

            G_by_stimulus = []

            for stimulus in (
                P1E_ADAPTATION_STIMULI
            ):

                (
                    metadata,
                    kc_rates,
                    apl_rates,
                    mbon_rates,
                    kc_G_mean_mV,
                    kc_D_mean_mV,
                ) = run_condition(
                    P1E_PILOT_LESION_SEED,
                    stimulus,
                    P1E_PILOT_SIMULATION_SEED,
                    p1e_current_weights_mV=(
                        current_weights_mV
                    ),
                )

                # Única variable utilizada por la regla:
                # estado sináptico efectivo postsináptico KC.
                G_by_stimulus.append(
                    np.asarray(
                        kc_G_mean_mV,
                        dtype=np.float64,
                    )
                )

            G_by_stimulus = np.stack(
                G_by_stimulus,
                axis=0,
            )

            if G_by_stimulus.shape != (
                len(P1E_ADAPTATION_STIMULI),
                n_kc,
            ):
                raise RuntimeError(
                    "P1-E: matriz G por estímulo "
                    "tiene shape inesperado."
                )

            Gbar_mV = np.mean(
                G_by_stimulus,
                axis=0,
            )

            previous_weights_mV = (
                current_weights_mV.copy()
            )

            (
                current_weights_mV,
                epsilon,
            ) = p1e_apply_update(
                previous_weights_mV,
                w0_mV,
                plastic_mask,
                Gbar_mV,
                eta,
            )

            # =================================================
            # INVARIANTES MECÁNICOS
            # =================================================

            # 1. Fuera del dominio plástico nada cambia.
            if not np.array_equal(
                current_weights_mV[
                    ~plastic_mask
                ],
                w0_mV[
                    ~plastic_mask
                ],
            ):
                raise RuntimeError(
                    "P1-E modificó pesos fuera "
                    "del dominio permitido."
                )

            # 2. Ningún peso plástico cruza límites.
            wp = current_weights_mV[
                plastic_mask
            ]

            w0p = w0_mV[
                plastic_mask
            ]

            tol = 1e-12

            if np.any(
                wp
                <
                (
                    P1E_WEIGHT_MIN_FACTOR
                    * w0p
                    - tol
                )
            ):
                raise RuntimeError(
                    "P1-E violó límite inferior."
                )

            if np.any(
                wp
                >
                (
                    P1E_WEIGHT_MAX_FACTOR
                    * w0p
                    + tol
                )
            ):
                raise RuntimeError(
                    "P1-E violó límite superior."
                )

            # 3. No aparecen conexiones nuevas.
            if np.any(
                (
                    w0_mV == 0.0
                )
                &
                (
                    current_weights_mV != 0.0
                )
            ):
                raise RuntimeError(
                    "P1-E creó una conexión nueva."
                )

            # 4. eta=0 debe ser identidad exacta.
            if eta == 0.0:

                if not np.array_equal(
                    current_weights_mV,
                    w0_mV,
                ):
                    raise RuntimeError(
                        "eta=0 modificó pesos."
                    )

            # 5. La ecuación implementada debe coincidir
            # exactamente con la actualización esperada.
            plastic_positions = np.flatnonzero(
                plastic_mask
            )

            kc_global_to_local = np.full(
                len(neuron_ids),
                -1,
                dtype=np.int32,
            )

            kc_global_to_local[
                kc_indices
            ] = np.arange(
                n_kc,
                dtype=np.int32,
            )

            target_local = (
                kc_global_to_local[
                    post_idx[
                        plastic_positions
                    ]
                ]
            )

            expected = np.clip(
                previous_weights_mV[
                    plastic_positions
                ]
                *
                np.exp(
                    eta
                    * epsilon[
                        target_local
                    ]
                ),
                P1E_WEIGHT_MIN_FACTOR
                * w0_mV[
                    plastic_positions
                ],
                P1E_WEIGHT_MAX_FACTOR
                * w0_mV[
                    plastic_positions
                ],
            )

            if not np.allclose(
                current_weights_mV[
                    plastic_positions
                ],
                expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(
                    "P1-E no coincide con la "
                    "ecuación congelada."
                )

            if cycle in (
                0,
                1,
                4,
                9,
                19,
            ):

                changed = int(
                    np.sum(
                        np.abs(
                            current_weights_mV[
                                plastic_mask
                            ]
                            -
                            w0_mV[
                                plastic_mask
                            ]
                        )
                        > 1e-12
                    )
                )

                print(
                    f"cycle={cycle + 1:02d} | "
                    f"epsilon mean="
                    f"{np.mean(epsilon):.6f} | "
                    f"changed={changed:,}/"
                    f"{int(plastic_mask.sum()):,}"
                )

        # ----------------------------------------------------
        # Controles finales de eta
        # ----------------------------------------------------

        changed_final = int(
            np.sum(
                np.abs(
                    current_weights_mV[
                        plastic_mask
                    ]
                    -
                    w0_mV[
                        plastic_mask
                    ]
                )
                > 1e-12
            )
        )

        if eta == 0.0:

            if changed_final != 0:
                raise RuntimeError(
                    "Control eta=0 falló."
                )

            print(
                "✓ eta=0: identidad exacta"
            )

        else:

            if changed_final == 0:
                raise RuntimeError(
                    "eta>0 no modificó ningún "
                    "peso autorizado."
                )

            print(
                f"✓ eta={eta:.3f}: "
                f"{changed_final:,} pesos "
                "autorizados cambiaron"
            )

        print(
            "✓ límites 0.5x–2.0x respetados"
        )

        print(
            "✓ fuera de P1-E: pesos intactos"
        )

        print(
            "✓ cero aristas nuevas"
        )

    print()
    print("=" * 72)
    print("✓ P1-E TECHNICAL PILOT PASSED")
    print("✓ MBON NO FUE USADO COMO SEÑAL DE ACTUALIZACIÓN")
    print("=" * 72)




# ============================================================
# P1-E PILOT ENTRY POINT
# ============================================================

def save_p1i_functional_records(records, eta, cycle):
    """Guarda las respuestas neuronales acumuladas."""
    folder = OUTPUT_DIR / "functional_checkpoints" / f"lesion_seed{P1E_PILOT_LESION_SEED:03d}"
    folder.mkdir(parents=True, exist_ok=True)

    filename = folder / (
        f"functional_eta{eta:.3f}_cycle{cycle:02d}.npz"
    )

    np.savez_compressed(
        filename,
        eta=np.array([r["eta"] for r in records]),
        cycle=np.array([r["cycle"] for r in records]),
        stimulus=np.array([r["stimulus"] for r in records]),
        lesion_seed=np.array([r["lesion_seed"] for r in records]),
        simulation_seed=np.array([r["simulation_seed"] for r in records]),
        I_APL1=np.array([r["I_APL1"] for r in records]),
        I_APL2=np.array([r["I_APL2"] for r in records]),
        kc_rates=np.stack([r["kc_rates"] for r in records]),
        apl_rates=np.stack([r["apl_rates"] for r in records]),
        mbon_rates=np.stack([r["mbon_rates"] for r in records]),
    )


def run_p1i_technical_pilot():
    """Piloto técnico P1-I, sin evaluar recuperación."""
    print("\n" + "=" * 72)
    print("P1-I TECHNICAL PILOT")
    print("=" * 72)

    w0_mV = p1e_initial_weight_state_mV()

    (
        lesioned_indices_local,
        lesioned_output_mask_local,
        plastic_mask,
    ) = p1i_lesion_and_plastic_masks(
        P1E_PILOT_LESION_SEED, w0_mV
    )

    if len(lesioned_indices_local) != 1294:
        raise RuntimeError("Número incorrecto de KC lesionadas.")
    if not np.any(plastic_mask):
        raise RuntimeError("Dominio plástico P1-I vacío.")

    for eta in P1E_PILOT_ETAS:
        print(f"\nP1-I: eta = {eta:.3f}")
        functional_records = []
        current_weights_mV = w0_mV.copy()

        for cycle in range(P1E_N_CYCLES):
            I_by_stimulus = []

            for stimulus in P1E_ADAPTATION_STIMULI:
                (
                    metadata, kc_rates, apl_rates,
                    mbon_rates, kc_G_mean_mV,
                    kc_D_mean_mV,
                ) = run_condition(
                    P1E_PILOT_LESION_SEED,
                    stimulus,
                    P1E_PILOT_SIMULATION_SEED,
                    p1e_current_weights_mV=current_weights_mV,
                )

                I_apl = p1i_apl_input_drive(
                    kc_rates,
                    current_weights_mV,
                    lesioned_indices_local,
                )
                I_by_stimulus.append(I_apl)

                functional_records.append({
                    "eta": float(eta),
                    "cycle": int(cycle + 1),
                    "stimulus": str(stimulus),
                    "lesion_seed": int(P1E_PILOT_LESION_SEED),
                    "simulation_seed": int(P1E_PILOT_SIMULATION_SEED),
                    "I_APL1": float(I_apl[0]),
                    "I_APL2": float(I_apl[1]),
                    "kc_rates": np.asarray(kc_rates, dtype=float).copy(),
                    "apl_rates": np.asarray(apl_rates, dtype=float).copy(),
                    "mbon_rates": np.asarray(mbon_rates, dtype=float).copy(),
                })

            I_by_stimulus = np.stack(
                I_by_stimulus, axis=0
            )

            if I_by_stimulus.shape != (8, 2):
                raise RuntimeError(
                    "Dimensiones incorrectas de entrada APL."
                )

            Ibar = np.mean(I_by_stimulus, axis=0)

            previous_weights_mV = current_weights_mV.copy()

            current_weights_mV, epsilon = p1i_apply_update(
                previous_weights_mV,
                w0_mV,
                plastic_mask,
                Ibar,
                eta,
            )

            if not np.array_equal(
                current_weights_mV[~plastic_mask],
                w0_mV[~plastic_mask],
            ):
                raise RuntimeError(
                    "P1-I modificó conexiones prohibidas."
                )

            positions = np.flatnonzero(plastic_mask)
            apl_map = np.full(len(neuron_ids), -1, dtype=int)
            apl_map[apl_indices] = np.arange(2)
            targets = apl_map[post_idx[positions]]

            if np.any(targets < 0):
                raise RuntimeError("Destino APL inválido.")

            expected = np.clip(
                previous_weights_mV[positions]
                * np.exp(eta * epsilon[targets]),
                0.5 * w0_mV[positions],
                2.0 * w0_mV[positions],
            )

            if not np.allclose(
                current_weights_mV[positions],
                expected, rtol=1e-12, atol=1e-12
            ):
                raise RuntimeError("Actualización P1-I incorrecta.")

            max_change = np.max(np.abs(
                current_weights_mV[plastic_mask]
                - w0_mV[plastic_mask]
            ))

            print(
                f"eta={eta:.3f} | ciclo={cycle + 1:02d} | "
                f"I_APL1={Ibar[0]:.3f} | "
                f"I_APL2={Ibar[1]:.3f} | "
                f"error=({epsilon[0]:+.4f}, "
                f"{epsilon[1]:+.4f}) | "
                f"cambio_max={max_change:.6f} mV"
            )

            save_p1i_functional_records(
                functional_records, eta, cycle + 1
            )

            folder = OUTPUT_DIR / "functional_checkpoints" / f"lesion_seed{P1E_PILOT_LESION_SEED:03d}"
            np.savez_compressed(
                folder / (
                    f"weights_eta{eta:.3f}_cycle{cycle+1:02d}.npz"
                ),
                weights_before_update_mV=previous_weights_mV,
                weights_after_update_mV=current_weights_mV,
                eta=float(eta),
                cycle=int(cycle + 1),
            )

    print("\nPiloto técnico P1-I finalizado.")


if __name__ == "__main__":
    for lesion_seed in LESION_SEEDS:
        P1E_PILOT_LESION_SEED = lesion_seed
        print(f"\nENTRENAMIENTO MULTILESION: lesion_seed{lesion_seed:03d}")
        run_p1i_technical_pilot()
