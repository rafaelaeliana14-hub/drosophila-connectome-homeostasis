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
