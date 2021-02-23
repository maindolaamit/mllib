import pandas as pd
import numpy as np
import os
from sklearn.model_selection import StratifiedKFold, KFold

# Keep randomness same
np.random.seed(2210)


def create_folds(input_file, target_cols, shuffle, stratify=False):
    # Read the dataset
    df = pd.read_csv(input_file)
    # Create index for fold
    df['kfold'] = -1

    # Shuffle the dataset
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    # Select Fold strategy
    cv = None
    if stratify:
        cv = StratifiedKFold(n_split=5, shuffle=False)
    else:
        cv = KFold(n_split=5, shuffle=shuffle)

    # Loop in folds 
    for fold, (train_idx, test_idx) in enumerate(cv.split(X=df, y=df[target_cols].values)):
        df.iloc[test_idx, 'kfold'] = fold

    input_dir = input_file.split(os.path.sep)[0:-1]
    df.to_csv(os.path.join(input_dir, 'train.csv'), index=False)

