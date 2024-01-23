import numpy as np
import pandas as pd
import os

slide_level_feature_dir = './slide_level_features'
if not os.path.exists(slide_level_feature_dir):
    os.makedirs(slide_level_feature_dir)

nuc_feature_dir = './features/nucleus'
tile_feature_dir = './features/tile'

# nuc_feature
backbones = os.listdir(nuc_feature_dir)
for j in range(len(backbones)):
    nuc_feature_list_dir = os.listdir(os.path.join(nuc_feature_dir,backbones[j]))
    slide_level_feature = []
    for i in range((len(nuc_feature_list_dir))):
        csv_path = os.path.join(nuc_feature_dir,backbones[j], nuc_feature_list_dir[i])
        feature_df = pd.read_csv(csv_path)
        mean_feature = list(feature_df.mean())
        std_feature = list(feature_df.std())
        slide_level_feature.append([nuc_feature_list_dir[i][:23]] + mean_feature + std_feature)

    features = ['Barcode'] + list(map(lambda x: x+'_mean', feature_df.columns)) + list(map(lambda x: x+'_sd', feature_df.columns))
    slide_level_feature_df = pd.DataFrame(slide_level_feature, columns = features) 
    slide_level_feature_df.to_csv(os.path.join(slide_level_feature_dir,'CRC_nuc_'+ backbones[j] +'.csv'), index=False)

# tile_feature
backbones = os.listdir(tile_feature_dir)
for j in range(len(backbones)):
    tile_feature_list_dir = os.listdir(os.path.join(tile_feature_dir,backbones[j]))
    slide_level_feature = []
    for i in range((len(tile_feature_list_dir))):
        csv_path = os.path.join(tile_feature_dir,backbones[j], tile_feature_list_dir[i])
        feature_df = pd.read_csv(csv_path)
        max_feature = list(feature_df.max())
        slide_level_feature.append([tile_feature_list_dir[i][:23]] + max_feature)

    features = ['Barcode'] + list(map(lambda x: x+'_max', feature_df.columns))
    slide_level_feature_df = pd.DataFrame(slide_level_feature, columns = features) 
    slide_level_feature_df.to_csv(os.path.join(slide_level_feature_dir,'CRC_tile_'+ backbones[j] +'.csv'), index=False)