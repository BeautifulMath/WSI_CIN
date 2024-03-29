# Integrating Histologic Data and Machine Learning to Predict Chromosomal Instability in Colorectal Cancer 

![alt text](https://github.com/BeautifulMath/WSI_CIN/blob/main/Image/Figure1.png?raw=true)

**Data**  
  
We provide the slide-level nuclear morphological features for 315 TCGA-CRC slides, which can be found in the [data](https://github.com/BeautifulMath/WSI_CIN/tree/main/data) folder under the file name *CRC_nucleus_morphology.csv*.

**Code**
1. Download WSIs: <https://portal.gdc.cancer.gov/repository>
2. Download tumor annotations (.csv files): <https://zenodo.org/records/5320076>
3. Reading csv in QuPath: *csv2annotation.groovy*
4. Get tiles from tumor regions: QuPath (Analyze > Tiles & superpixels > Create tiles)
5. Extract tumor cell nuclei: QuPath StarDist extension
6. Generate 100x100-pixel tumor nucleus images
7. Extract features: *extract_deep_features.py*, *extract_morphologic_features.py*
8. Slide-level feature aggregation: *feature_aggregation.py*
9. Get AS(Aneuploidy Score), WGD(Whole Genome Doubling) labels: *get_label.py*
10. Data split for 5-fold cross validation: *5_fold_split.py*
11. Train model: *MLP.py*
12. Calculate the correlation between nuclear morphology features and copy number signatures: *signature_morphology_correlation.py*
13. Extract the feature importances from random forest classifier: *random_forest.ipynb*

**Hover-Net nuclei segmentation results for tiles of 3 WSIs**

[Hover-Net outputs](http://naver.me/IgDGZjtN)

AS prediction scores were predicted at the tile-level by feeding individual tile features into trained MLP models.
Patches were categorized based on a tile-level AS prediction score difference of 0.1 units.

(red labels are neoplastic cells, blue labels are connective tissue cells, green labels are inflammatory cells, orange labels are
non-neoplastic cells, and yellow labels are necrotic cells.)

**Folder Structures**
```bash
├── tile
│   ├── TCGA-3L-AA1B-01Z-00-DX2
│   ├── TCGA-4N-A93T-01Z-00-DX1
│   ...
│   └── TCGA-WS-AB45-01Z-00-DX1
├── nucleus
│   ├── TCGA-3L-AA1B-01Z-00-DX2
│   ├── TCGA-4N-A93T-01Z-00-DX1
│   ...
│   └── TCGA-WS-AB45-01Z-00-DX1
├── features
│   ├── tile
│   │   ├── resnet18
│   │   ├── resnet18-ssl
│   │   ├── densenet121
│   │   └── vgg11
│   └── nucleus
│       ├── morphology
│       ├── resnet18
│       ├── resnet18-ssl
│       ├── densenet121
│       └── vgg11
├── slide_level_features
└── MLP
    ├── 5_fold_split
    ├── metrics
    └── model_weights
``` 
