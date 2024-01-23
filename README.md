# Deep Learning-based Histologic Predictions of Chromosomal Instability in Colorectal Cancer 


Folder Structure
tile/
-------TCGA-A6-A56B-01Z-00-DX1/
--------------

1. Download WSIs: https://portal.gdc.cancer.gov/repository
2. Download tumor annotations: https://zenodo.org/records/5320076
3. Get tiles from tumor regions: QuPath
4. Extract tumor cell nuclei: QuPath StarDist extension
5. Generate 100x100-pixel tumor nucleus images
6. Extract features: extract_deep_features.py, extract_morphologic_features.py
