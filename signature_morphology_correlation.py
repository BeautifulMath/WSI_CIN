## Download CIN signature activities: 'https://www.nature.com/articles/s41586-022-04789-9' 
## 'Supplementary Tables 15-22.xlsx'

import pandas as pd, numpy as np
import scipy.stats

activities = pd.read_excel('Supplementary Tables 15-22.xlsx', sheet_name='ST_18_TCGA_Activities_raw') 
morp_feat = pd.read_csv('CRC_nucleus_morphology.csv') 

# Find intersection of patients in two files

idx = []
for i in range(len(morp_feat)):
    for j in range(len(activities)):
        if morp_feat.Barcode[i][:12] == activities['Unnamed: 0'][j][:12]:
            idx.append(i)
            break 

morp_feat = morp_feat.loc[idx]
morp_feat.reset_index(inplace=True, drop=True) 

idx = []
for i in range(len(morp_feat)):
    for j in range(len(activities)):
        if morp_feat.Barcode[i][:12] == activities['Unnamed: 0'][j][:12]:
            idx.append(j) 

activities = activities.loc[idx]
activities.reset_index(inplace=True, drop=True) 

# Calculate correlation coefficients
cor_30 = []
for i in range(17):
    cor = []
    for j in range(30):
        spe_r = scipy.stats.pearsonr(morp_feat.iloc[:,j+1], activities.iloc[:,i+1])
        cor.append(str(round(spe_r[0],4)) + ' (' + str(round(spe_r[1],4)) + ')')
    cor_30.append(cor) 

signature_cor = pd.DataFrame(cor_30, index=activities.columns[1:], columns=morp_feat.columns[1:])
signature_cor.to_csv('signature_morphology_correlation.csv', index=False)