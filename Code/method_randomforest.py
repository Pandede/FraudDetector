import pickle

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from metrics import *

# Dataset
train_data = pd.read_csv('../Data/train_modified.csv', sep="|")
feature_cols = train_data.columns[train_data.columns != 'fraud']

X = train_data[feature_cols].values
Y = train_data['fraud'].values

# Settings
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

# Oversampling is discarded, further information is written on hackMD
# Oversampling
# ros = RandomOverSampler(random_state=0, sampling_strategy='minority')
# train_feature_resampled, train_target_resampled = ros.fit_resample(train_data[feature_cols], train_data['fraud'])

# Initializing before loops
profit = 0
f1 = 0
n_estimators = 100

# for n_fold, (train_idx, validation_idx) in enumerate(folds.split(train_feature_resampled, train_target_resampled)):
print('[Validating with K-Fold ...]')
for n_fold, (train_idx, validation_idx) in enumerate(folds.split(X, Y)):
    train_x, validation_x = X[train_idx], X[validation_idx]
    train_y, validation_y = Y[train_idx], Y[validation_idx]

    model = RandomForestClassifier(n_estimators=n_estimators)
    model.fit(train_x, train_y)

    # Cross Validation
    prediction = model.predict(validation_x)
    profit += retailer_profit(validation_y, prediction) / folds.n_splits
    f1 += f1_score(validation_y, prediction) / folds.n_splits

    # Save
    with open('../Model/RandomForest/model_%02d.pkl' % n_fold, 'wb') as f:
        pickle.dump(model, f)

print(f'[With K-Fold] Profit: {profit}, f1-score: {f1:.4f}')

# Whole Dataset
print('[Training with entire dataset ...]')
model = RandomForestClassifier(n_estimators=n_estimators)
model.fit(X, Y)
with open('../Model/RandomForest/model_reg_entire.pkl', 'wb') as f:
    pickle.dump(model, f)
print('[Model trained with entire dataset is saved]')
