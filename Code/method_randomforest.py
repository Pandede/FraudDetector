import pickle

import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

from metrics import *

# Dataset
train_data = pd.read_csv('../Data/train.csv', sep="|")
feature_cols = train_data.columns[train_data.columns != 'fraud']

# Settings
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
ros = RandomOverSampler(random_state=0)

# Oversampling
train_feature_resampled, train_target_resampled = ros.fit_resample(train_data[feature_cols], train_data['fraud'])

# Initializing before loops
profit = 0
f1 = 0
params = {
}

for n_fold, (train_idx, validation_idx) in enumerate(folds.split(train_feature_resampled, train_target_resampled)):
    train_x = train_feature_resampled.iloc[train_idx].values
    train_y = train_target_resampled.iloc[train_idx].values
    validation_x = train_feature_resampled.iloc[validation_idx].values
    validation_y = train_target_resampled.iloc[validation_idx].values

    model = RandomForestClassifier(**params)
    model.fit(train_x, train_y)

    # Cross Validation
    prediction = model.predict(validation_x)
    prediction = np.round(prediction)
    profit += retailer_profit(validation_y, prediction) / folds.n_splits
    f1 += f1_score(validation_y, prediction) / folds.n_splits

    # Save
    with open('../Model/RandomForest/model_%02d.pkl' % n_fold, 'wb') as f:
        pickle.dump(model, f)

print(f'Profit: {profit}, f1-score: {f1}')
