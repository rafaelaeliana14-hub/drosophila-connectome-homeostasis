# Connectome-constrained homeostatic plasticity in Drosophila

Computational investigation of local KC-to-APL homeostatic plasticity and functional responses following partial Kenyon cell output silencing in a connectome-constrained model of the Drosophila melanogaster mushroom body.

## Overview

The model comprises 8,239 neurons:
- 2,279 olfactory receptor neurons (ORN)
- 685 antennal lobe projection neurons (ALPN)
- 5,177 Kenyon cells (KC)
- 2 APL neurons
- 96 mushroom body output neurons (MBON)

The network contains 460,086 directed neuron pairs representing 1,486,479 anatomical synaptic contacts.

## Experimental design

The principal experiment evaluates 20 distinct lesion configurations, each involving functional silencing of the outgoing connections of 1,294 Kenyon cells (approximately 25%).

Homeostatic adaptation is restricted to surviving positive KC-to-APL connections.

The experiment compares:
- No plasticity (eta = 0)
- Homeostatic plasticity (eta = 0.03)

Each configuration undergoes 20 recorded training cycles involving eight olfactory stimuli.

## Main findings

Homeostatic plasticity consistently reduced the normalized KC-to-APL input error across all 20 lesion configurations.

Mean error decreased:
- APL1: 22.71% to 15.04%
- APL2: 20.63% to 14.24%

However, improved local homeostatic regulation did not produce a global recovery of MBON firing rates.

Complementary analyses of one lesion configuration identified partial recovery of selected MBON subthreshold responses and olfactory representational geometry. These findings should not be generalized to all lesion configurations.

## Core files

- `build_mb_subcircuit.py`: constructs the anatomical subcircuit.
- `P1I_multilesion_training.py`: runs the principal multilesion homeostatic plasticity experiment.
- `results/P1E_reference/P1E_reference_Gstar.npz`: frozen P1-E reference.

## Data provenance

The subcircuit was extracted using FlyWire v783 neuronal annotations and connectivity files from the modeling dataset.

The builder requires:
- `data/flywire783/neuron_annotations_v783.tsv`
- `../brain-model/Connectivity_783.parquet`
- `../brain-model/Completeness_783.csv`

The completeness file is loaded by the original builder but is not used in its neuronal selection criteria.

Redistribution permissions and full dataset provenance remain under review.

## Reproducibility status

This repository is under preparation. The complete dependencies, input datasets, analysis scripts, figure-generation procedures, and execution instructions have not yet been deposited.

The principal model and frozen experimental outputs are being preserved without retrospective parameter changes.

## Interpretation

The results distinguish:
1. Residual network robustness without adaptation.
2. Homeostatic compensation of a defined local input variable.
3. Functional recovery assessed independently through MBON activity.

Local homeostatic compensation does not necessarily imply recovery of distributed circuit output.

## Citation

Manuscript in preparation. Author list, publication details, and archival DOI pending.

## License

Pending confirmation of code ownership and third-party data licensing.
