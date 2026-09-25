#!/bin/sh
# Download the FlyWire v783 connectome as packaged for the Shiu et al. and Eon Systems models,
# plus the FlyWire cell type annotations. About 135 MB in total.
set -e
mkdir -p data
curl -L -o data/Completeness_783.csv https://raw.githubusercontent.com/eonsystemspbc/fly-brain/main/data/2025_Completeness_783.csv
curl -L -o data/Connectivity_783.parquet https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/main/Connectivity_783.parquet
curl -L -o data/neuron_annotations.tsv https://raw.githubusercontent.com/flyconnectome/flywire_annotations/main/supplemental_files/Supplemental_file1_neuron_annotations.tsv
