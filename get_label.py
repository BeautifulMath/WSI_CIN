# Download AS label: https://ars.els-cdn.com/content/image/1-s2.0-S1535610818301119-mmc2.xlsx

import pandas as pd

slide_level_feature = pd.read_csv('./slide_level_features/CRC_nucleus_resnet18-ssl.csv')
label_path = '1-s2.0-S1535610818301119-mmc2.xlsx'
labels_df = pd.read_excel(label_path, header=1, index_col='Sample')
WSI_label = []
for i in range(len(slide_level_feature)):
    patient_name = slide_level_feature.Barcode[i][:15]
    try:
        AS = labels_df.loc[patient_name, 'AneuploidyScore(AS)']
        Genome_doublings = labels_df.loc[patient_name, 'Genome_doublings']
    except: 
        AS = None
        Genome_doublings = None
    WSI_label.append([slide_level_feature.Barcode[i], AS, Genome_doublings])

feature_df = pd.DataFrame(WSI_label, columns = ['Barcode', 'AS', 'Genome_doublings'])
feature_df.to_csv('./slide_level_features/CRC_AS_labels.csv', index=False)
