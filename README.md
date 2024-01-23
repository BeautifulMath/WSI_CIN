# Deep Learning-based Histologic Predictions of Chromosomal Instability in Colorectal Cancer 

Folder Structures
```bash
├── tile
│   ├── TCGA-3L-AA1B-01Z-00-DX2
│   ├── TCGA-4N-A93T-01Z-00-DX1
...
│   └── TCGA-WS-AB45-01Z-00-DX1
├── nucleus
│   ├── TCGA-A6-A56B-01Z-00-DX1
│   ├── TCGA-4N-A93T-01Z-00-DX1
...
│   └── TCGA-WS-AB45-01Z-00-DX1
├── features
│   ├── tile
│   │   ├──resnet18
│   │   ├──resnet18-ssl
│   │   ├──densenet121
│   │   └──vgg11
│   └──nucleus
│       ├──morphology
│       ├──resnet18
│       ├──resnet18-ssl
│       ├──densenet121
│       └──vgg11
├── slide_level_features
└── MLP
``` 

1. Download WSIs: <https://portal.gdc.cancer.gov/repository>
2. Download tumor annotations: <https://zenodo.org/records/5320076>
3. Get tiles from tumor regions: QuPath
4. Extract tumor cell nuclei: QuPath StarDist extension
5. Generate 100x100-pixel tumor nucleus images
6. Extract features: extract_deep_features.py, extract_morphologic_features.py
