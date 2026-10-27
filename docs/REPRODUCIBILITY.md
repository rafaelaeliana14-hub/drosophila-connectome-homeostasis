# Reproducibilidad y procedencia de los resultados

## 1. Alcance

Este repositorio documenta un modelo computacional del cuerpo pedunculado de Drosophila melanogaster restringido por información conectómica.

Se distinguen tres niveles de evidencia:

1. Resultados reproducidos mediante la ejecución de scripts archivados.
2. Resultados preservados mediante archivos originales y verificaciones de integridad.
3. Análisis derivados cuya procedencia o reproducción completa todavía requiere documentación.

La disponibilidad de un archivo no equivale, por sí misma, a la reproducción independiente de su resultado.

## 2. Experimento principal: plasticidad KC-APL

El experimento principal considera 20 máscaras de lesión independientes, cada una con 1294 células Kenyon cuya salida sináptica fue silenciada.

Se comparan condiciones sin aprendizaje (eta = 0) y con actualización multiplicativa de pesos KC-APL (eta = 0.03), utilizando una referencia P1-E previamente fijada.

Archivos principales:

- P1I_multilesion_training.py
- results/P1E_reference/P1E_reference_Gstar.npz
- results/P1I_multilesion_training/functional_checkpoints/

Se han preservado 60 checkpoints funcionales seleccionados: tres por lesión, para las 20 lesiones.

Las copias de estos 60 checkpoints fueron comparadas con sus originales mediante SHA-256, sin discrepancias.

No se ha repetido el entrenamiento multilesión completo desde cero en este repositorio.

## 3. Figuras reproducidas

### Figura 4

Script: scripts/plot_figura_4.py

La figura fue regenerada satisfactoriamente desde la copia del repositorio.

El análisis recuperó 46 cambios de actividad MBON: 8 aumentos y 38 disminuciones, a través de 19 lesiones.

### Figura 5

Script: scripts/plot_figura_5.py

La figura fue regenerada satisfactoriamente desde la copia del repositorio.

Distancia global respecto al estado intacto:
- Sin aprendizaje: 0.038125 Hz
- Con aprendizaje: 0.046615 Hz

Distancia para DM1:
- Sin aprendizaje: 0.305 Hz
- Con aprendizaje: 0.372917 Hz

En DM1, la distancia aumentó en 16 lesiones, disminuyó en 3 y permaneció sin cambios en 1.

La mejora del objetivo local APL no debe interpretarse como recuperación global de la actividad MBON.

## 4. Análisis subumbrales ORN_DL5

Se conserva el código del modelo espacial APL y los scripts de evaluación del voltaje MBON.

Las diez realizaciones utilizan una misma identidad de lesión (lesion_seed = 1) y distintas semillas de entrada olfativa.

Fuentes de entrada:
- Semilla 1: results/P1I_spatial_BLP_paired_input_ORN_DL5_seed1.npz
- Semillas 2-10: results/P1I_paired_input_ORN_DL5_seed2.npz a seed10.npz

Los identificadores de entrada SHA-256 coincidieron con los registrados en las trazas de voltaje de la condición intacta para las diez semillas (10/10).

El checkpoint de pesos aprendidos de la lesión 1 también fue copiado y verificado mediante SHA-256.

Las trazas completas de voltaje permanecen fuera del conjunto previsto para Git y están consideradas para un depósito de archivo separado.

Estos análisis no constituyen diez lesiones biológicamente independientes.

## 5. Controles mecanísticos y análisis derivados

Los archivos results/subthreshold/APL_MBON_KC_* corresponden a evaluaciones de sensibilidad de la vía MBON-KC. No deben confundirse con una demostración de recuperación funcional mediante plasticidad KC-APL.

El resumen del control DP1l describe una evaluación de una lesión con diez semillas internas de simulación. Su archivo JSON preserva resultados resumidos, pero no sustituye las trazas originales.

Las tablas de geometría representacional están preservadas como resultados derivados. La identificación y verificación de todos sus archivos generadores sigue pendiente.

## 6. Datos anatómicos y licencias

Los archivos anatómicos derivados de FlyWire se mantienen localmente en data/ y están excluidos del seguimiento de Git mientras se revisan sus condiciones de redistribución y citación.

El código que utiliza estos archivos puede requerir rutas y datos externos adicionales.

Por tanto, la regeneración integral del modelo a partir de los datos conectómicos originales todavía no está demostrada en esta copia independiente.

## 7. Limitaciones de reproducibilidad

- No se ha ejecutado nuevamente el entrenamiento multilesión completo.
- No se han regenerado todas las figuras del manuscrito desde este repositorio.
- No se ha verificado todavía la ejecución integral del modelo espacial desde un entorno nuevo.
- La procedencia completa del análisis de geometría representacional sigue pendiente.
- Los datos anatómicos excluidos requieren documentación de origen, versión y condiciones de acceso.
- Los scripts y resultados disponibles no deben interpretarse como validación experimental in vivo.

## 8. Estado de publicación

El repositorio permanece privado durante la preparación del material.

No se ha realizado todavía un depósito público definitivo de datos ni se ha asignado un DOI a este conjunto de reproducción.

## 9. Construcción del subcircuito conectómico

El script `build_mb_subcircuit.py` construye el subcircuito a partir de archivos asociados a la versión 783 del conectoma y sus anotaciones.

### Dependencias de entrada

- `data/flywire783/neuron_annotations_v783.tsv`: anotaciones neuronales.
- `../brain-model/Connectivity_783.parquet`: conectividad dirigida y número de contactos.
- `../brain-model/Completeness_783.csv`: archivo cargado por el script, pero no utilizado posteriormente para filtrar neuronas o conexiones.

Las rutas son relativas a la ubicación del script. Los archivos fuente no están incluidos actualmente en el repositorio remoto.

### Selección de poblaciones

Se utilizan los siguientes criterios sobre las anotaciones:

- ORN: `cell_class == "olfactory"`
- ALPN: `cell_class == "ALPN"`
- KC: `cell_class == "Kenyon_Cell"`
- APL: `cell_type == "APL"`
- MBON: `cell_class == "MBON"`

Solo se conservan las neuronas cuyo `root_id` aparece como `Presynaptic_ID` o `Postsynaptic_ID` en el archivo de conectividad.

### Selección de conexiones

Se conservan únicamente las filas en que ambos identificadores, presináptico y postsináptico, pertenecen al conjunto neuronal seleccionado.

Se mantienen las direcciones originales de las conexiones.

El número de filas se informa como pares conectados, mientras que la suma de la columna `Connectivity` proporciona el número de contactos sinápticos.

La unicidad de los pares dirigidos se verificó sobre mb_connectivity_intact.parquet: 460086 filas, 460086 pares presináptico-postsináptico únicos, 0 duplicados y 1486479 contactos sinápticos. Esta comprobación corresponde al archivo derivado conservado localmente.

### Resúmenes generados

El script produce:

- `data/mb_subcircuit/mb_neurons_intact.parquet`
- `data/mb_subcircuit/mb_connectivity_intact.parquet`
- `data/mb_subcircuit/mb_connectivity_summary.csv`
- `data/mb_subcircuit/mb_neurotransmitters_summary.csv`

El resumen de conectividad agrupa pares, contactos y números de neuronas presinápticas y postsinápticas por combinación dirigida de poblaciones.

El resumen de neurotransmisores contabiliza la anotación `top_nt` por población; no constituye una validación fisiológica de la identidad funcional de las sinapsis.

### Conteos del subcircuito utilizado

- Neuronas: 8239.
- Filas de conectividad dirigida: 460086.
- Contactos sinápticos: 1486479.

Estos conteos corresponden a la construcción utilizada previamente en el proyecto y no se han regenerado durante esta auditoría.

## 10. Procedencia de los archivos conectómicos

### Fuentes de descarga identificadas

Según el historial de preparación del proyecto del 20 de septiembre de 2026, los archivos de entrada se obtuvieron de las siguientes fuentes:

**Modelo cerebral de Shiu y colaboradores**

Repositorio:
https://github.com/philshiu/Drosophila_brain_model

Archivos:
- `Connectivity_783.parquet`
- `Completeness_783.csv`

El README de este proyecto distingue entre FlyWire v630, utilizado en su artículo original, y la configuración alternativa para la versión pública 783.

**Anotaciones neuronales de FlyWire**

Repositorio:
https://github.com/flyconnectome/flywire_annotations

Archivo utilizado:
- `neuron_annotations_v783.tsv`

El archivo proporciona las anotaciones neuronales empleadas para seleccionar las poblaciones ORN, ALPN, KC, APL y MBON.

### Cadena de procesamiento del presente estudio

1. Obtención de archivos de conectividad y anotaciones identificados como versión 783.
2. Selección de las poblaciones neuronales mediante `build_mb_subcircuit.py`.
3. Extracción de conexiones dirigidas cuyos extremos pertenecen al subcircuito.
4. Construcción del modelo computacional P1-I y sus experimentos de lesión y plasticidad.

### Limitaciones de procedencia

Las fuentes de descarga se identificaron mediante el historial del proyecto y la documentación del modelo cerebral.

Todavía está pendiente verificar:
- Las revisiones o commits exactos de los repositorios fuente.
- Los hashes de los archivos descargados frente a copias publicadas.
- Las transformaciones anteriores a la distribución de `Connectivity_783.parquet`.
- Las condiciones de licencia, atribución y redistribución de cada conjunto de datos.

Por tanto, se identifica la procedencia de descarga, pero todavía no se afirma una trazabilidad completa hasta los datos sinápticos primarios.

Los archivos anatómicos derivados permanecen excluidos de Git mientras se completa esta revisión.
