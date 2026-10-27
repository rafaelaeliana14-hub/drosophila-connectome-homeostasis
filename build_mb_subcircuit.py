from pathlib import Path
import pandas as pd


# ============================================================
# 1. RUTAS
# ============================================================

ROOT = Path(__file__).resolve().parent

ANNOTATIONS = ROOT / "data" / "flywire783" / "neuron_annotations_v783.tsv"
CONNECTIVITY = ROOT.parent / "brain-model" / "Connectivity_783.parquet"
COMPLETENESS = ROOT.parent / "brain-model" / "Completeness_783.csv"

OUTPUT_DIR = ROOT / "data" / "mb_subcircuit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CARGAR DATOS
# ============================================================

print("\n=== CONSTRUCCIÓN DEL SUBCIRCUITO OLFATIVO–MUSHROOM BODY ===\n")

print("Cargando anotaciones...")
annotations = pd.read_csv(
    ANNOTATIONS,
    sep="\t",
    low_memory=False
)

print("Cargando conectividad...")
connectivity = pd.read_parquet(CONNECTIVITY)

print("Cargando listado de neuronas del modelo...")
completeness = pd.read_csv(COMPLETENESS)

print("Datos cargados.\n")


# ============================================================
# 3. IDENTIFICAR LAS NEURONAS PRESENTES EN EL MODELO
# ============================================================

# En Connectivity_783, los índices corresponden al orden utilizado
# por el modelo. Usamos los IDs explícitos para evitar depender
# únicamente de la posición de las filas.

model_ids = set(connectivity["Presynaptic_ID"]).union(
    set(connectivity["Postsynaptic_ID"])
)

print(f"Neuronas únicas encontradas en Connectivity_783: {len(model_ids):,}")


# ============================================================
# 4. DEFINIR POBLACIONES DEL CIRCUITO
# ============================================================

population_conditions = {
    "ORN": annotations["cell_class"].eq("olfactory"),
    "ALPN": annotations["cell_class"].eq("ALPN"),
    "KC": annotations["cell_class"].eq("Kenyon_Cell"),
    "APL": annotations["cell_type"].eq("APL"),
    "MBON": annotations["cell_class"].eq("MBON"),
}

population_frames = []

print("\nPoblaciones anotadas:")

for population, condition in population_conditions.items():

    neurons = annotations.loc[condition].copy()
    neurons["population"] = population

    total_annotated = len(neurons)

    # Comprobar cuáles están presentes en el conectoma usado
    # por nuestro modelo.
    neurons["in_model"] = neurons["root_id"].isin(model_ids)

    total_model = int(neurons["in_model"].sum())

    print(
        f"{population:5s}: "
        f"{total_annotated:6,d} anotadas | "
        f"{total_model:6,d} presentes en el modelo"
    )

    population_frames.append(neurons)


neurons = pd.concat(
    population_frames,
    ignore_index=True
)


# ============================================================
# 5. CONSERVAR SOLO NEURONAS PRESENTES EN EL MODELO
# ============================================================

neurons_model = neurons.loc[
    neurons["in_model"]
].copy()

circuit_ids = set(neurons_model["root_id"])

print(
    f"\nTotal de neuronas del subcircuito presentes en el modelo: "
    f"{len(circuit_ids):,}"
)


# ============================================================
# 6. EXTRAER CONECTIVIDAD INTERNA
# ============================================================

print("\nExtrayendo conexiones internas del subcircuito...")

edges = connectivity.loc[
    connectivity["Presynaptic_ID"].isin(circuit_ids)
    & connectivity["Postsynaptic_ID"].isin(circuit_ids)
].copy()

print(f"Pares neuronales conectados: {len(edges):,}")
print(f"Contactos sinápticos totales: {edges['Connectivity'].sum():,}")


# ============================================================
# 7. AÑADIR LA POBLACIÓN PRE Y POSTSINÁPTICA
# ============================================================

population_map = (
    neurons_model
    .drop_duplicates("root_id")
    .set_index("root_id")["population"]
)

edges["pre_population"] = edges["Presynaptic_ID"].map(population_map)
edges["post_population"] = edges["Postsynaptic_ID"].map(population_map)


# ============================================================
# 8. RESUMEN DE CONEXIONES ENTRE POBLACIONES
# ============================================================

summary = (
    edges
    .groupby(
        ["pre_population", "post_population"],
        dropna=False
    )
    .agg(
        neuron_pairs=("Connectivity", "size"),
        synaptic_contacts=("Connectivity", "sum"),
        presynaptic_neurons=("Presynaptic_ID", "nunique"),
        postsynaptic_neurons=("Postsynaptic_ID", "nunique"),
    )
    .reset_index()
    .sort_values(
        "synaptic_contacts",
        ascending=False
    )
)

print("\n=== CONECTIVIDAD ENTRE POBLACIONES ===\n")
print(summary.to_string(index=False))


# ============================================================
# 9. RESUMEN DE NEUROTRANSMISORES
# ============================================================

nt_summary = (
    neurons_model
    .groupby(
        ["population", "top_nt"],
        dropna=False
    )
    .size()
    .reset_index(name="neurons")
    .sort_values(
        ["population", "neurons"],
        ascending=[True, False]
    )
)

print("\n=== NEUROTRANSMISORES ANOTADOS ===\n")
print(nt_summary.to_string(index=False))


# ============================================================
# 10. GUARDAR CIRCUITO INTACTO
# ============================================================

neurons_file = OUTPUT_DIR / "mb_neurons_intact.parquet"
edges_file = OUTPUT_DIR / "mb_connectivity_intact.parquet"
summary_file = OUTPUT_DIR / "mb_connectivity_summary.csv"
nt_file = OUTPUT_DIR / "mb_neurotransmitters_summary.csv"

neurons_model.to_parquet(
    neurons_file,
    index=False
)

edges.to_parquet(
    edges_file,
    index=False
)

summary.to_csv(
    summary_file,
    index=False
)

nt_summary.to_csv(
    nt_file,
    index=False
)


# ============================================================
# 11. FINAL
# ============================================================

print("\n=== ARCHIVOS CREADOS ===\n")

print(neurons_file)
print(edges_file)
print(summary_file)
print(nt_file)

print(
    "\nSubcircuito intacto construido correctamente."
)

print(
    "\nIMPORTANTE: todavía NO se ha aplicado ninguna lesión "
    "ni mecanismo de plasticidad."
)