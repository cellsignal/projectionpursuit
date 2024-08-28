The script used to process the data on the bwForCluster Helix is provided in `run_mcmicro.sh` using the [MCMICRO pipeline](https://mcmicro.org) (Schapiro et al. 2022). The script refers to a config file provided in `run_mcmicro.config` and the parameter `yml` file provided in `params.yml`.

Nextflow version: 24.04.2.5914
Singularity version: 3.11.3

Input data was structured as follows:
```
./data/
├── markers.csv
└── staging
    ├── SCC_T37C_T1_1_Day_1
    │   ├── SCC_T37C_T1_Scan1_[10838,39481]_component_data.tif
    │   ├── SCC_T37C_T1_Scan1_[10838,40177]_component_data.tif
    │   ├── SCC_T37C_T1_Scan1_[10838,40874]_component_data.tif
    │   └── ...
    └── SCC_T37C_T1_1_Day_2
        ├── SCC_T37C_T1_Day 2_Scan1_[10072,39468]_component_data.tif
        ├── SCC_T37C_T1_Day 2_Scan1_[10072,40165]_component_data.tif
        ├── SCC_T37C_T1_Day 2_Scan1_[10072,40861]_component_data.tif
        └── ...
```

Parameter file content explanation:
```
workflow:
  start-at: staging                  --> start with phenoimager-specific staging
  stop-at: quantification            --> stop at quantification (steps to run: staging, registration, segmentation, quantification)
  illumination: false                --> skips illumination
  staging-method: phenoimager2mc
  segmentation: mesmer               --> selects the DeepCell Mesmer segmentation tool 
  segmentation-recyze: true          --> extract only meaningful channels for segmentation
  segmentation-max-projection: true 
  segmentation-channel: 5 11         --> all channels to be taken into account for segmentation
  segmentation-nuclear-channel: 5 11 --> run maximum projection on nuclear channels (indexing starts at 1) to account for grid artifacts
options:
  phenoimager2mc: -m 6 --normalization max   --> number of channels per cycle is 6, float32 to uint16 conversion done by normalizing to the maximum value per channel across all tiles for the given cycle
  ashlar: --align-channel 4 --flip-y         --> nuclear channel is channel 4 (indexing starts at 0 for ashlar), to account for tile layour, y axis has to be flipped.
  mesmer: --image-mpp 0.49884663906387205 --nuclear-channel 0 --compartment "nuclear"     --> provides pixel size, segmentation method and the nuclear channel (should be 0 in the image extracted for segmentation)
```

Data folder structure after pipeline completion:
```
./data/
├── markers.csv
├── qc
│   ├── metadata.yml
│   ├── params.yml
│   └── provenance
├── quantification
│   └── data--mesmer_cell.csv
├── raw
│   ├── SCC_T37C_T1_1_Day_1.ome.tif
│   └── SCC_T37C_T1_1_Day_2.ome.tif
├── registration
│   └── data.ome.tif
├── segmentation
│   └── mesmer-data
│       └── cell.tif
└── staging
    ├── SCC_T37C_T1_1_Day_1
    │   ├── SCC_T37C_T1_Scan1_[10838,39481]_component_data.tif
    │   ├── SCC_T37C_T1_Scan1_[10838,40177]_component_data.tif
    │   ├── SCC_T37C_T1_Scan1_[10838,40874]_component_data.tif
    │   └── ...
    └── SCC_T37C_T1_1_Day_2
        ├── SCC_T37C_T1_Day 2_Scan1_[10072,39468]_component_data.tif
        ├── SCC_T37C_T1_Day 2_Scan1_[10072,40165]_component_data.tif
        ├── SCC_T37C_T1_Day 2_Scan1_[10072,40861]_component_data.tif
        └── ...
```

The provided execution report (`execution_report.html`) states in-depth details about the pipeline run.

To ensure reproducibility, the data was processed using Docker containers in a Linux environment (Nextflow version 23.10.0.5889) with the same parameters and data and the final quantification files had an identical `md5sum` (d1f37a429cd0ab72fcb332991eb68ede).