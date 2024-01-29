import pandas as pd, numpy as np
import os
from sklearn.model_selection import StratifiedKFold
SPLITS = 5
skf = StratifiedKFold(n_splits = SPLITS, shuffle = True) 

split_dir = './MLP/5_fold_split'
if not os.path.exists(split_dir):
    os.makedirs(split_dir)

feature_df = pd.read_csv('./slide_level_features/CRC_nuc_resnet18-ssl.csv')
label_df = pd.read_csv('./slide_level_features/CRC_AS_labels.csv')
concat_df = pd.concat([label_df, feature_df], axis=1)
concat_df = concat_df.dropna(axis=0, subset=['AS'])
concat_df.reset_index(drop=True, inplace=True)

X = concat_df.iloc[:, 4:]
concat_df['AS_10'] = np.where(concat_df["AS"] > 10, 1, 0)
y = concat_df['AS_10'] 

n_iter = 0
for train_idx, test_idx in skf.split(X, y):
    n_iter += 1
    print(f'--------------------{n_iter}번째 KFold-------------------')
    print(f'train_idx_len : {len(train_idx)} / test_idx_len : {len(test_idx)}')
    np.save(os.path.join(split_dir, 'AS_test_idx_fold_{}.npy'.format(n_iter)), test_idx)
    np.save(os.path.join(split_dir, 'AS_train_idx_fold_{}.npy'.format(n_iter)), train_idx)