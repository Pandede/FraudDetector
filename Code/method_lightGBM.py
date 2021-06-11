import lightgbm as lgb
import pandas as pd
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
params = {
    'nthread': 4,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'xentropy',
    'learning_rate': 0.1,
    'n_estimator': 150,
    'num_leaves': 4,
    'max_depth': 8,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.7,
    'bagging_freq': 1,
}

print('[Validating with K-Fold ...]')
for n_fold, (train_idx, validation_idx) in enumerate(folds.split(X, Y)):
    train_x, validation_x = X[train_idx], X[validation_idx]
    train_y, validation_y = Y[train_idx], Y[validation_idx]

    train_dataset = lgb.Dataset(train_x, label=train_y)
    validation_dataset = lgb.Dataset(validation_x, label=validation_y, reference=train_dataset)
    booster = lgb.train(params, train_dataset, num_boost_round=50000, valid_sets=[train_dataset, validation_dataset],
                        early_stopping_rounds=1000, verbose_eval=False)

    # Cross validation
    prediction = booster.predict(validation_x, num_iteration=booster.best_iteration)
    prediction = np.round(prediction)
    profit += retailer_profit(validation_y, prediction) / folds.n_splits
    f1 += f1_score(validation_y, prediction) / folds.n_splits

    # Save
    booster.save_model('../Model/LightGBM/model_%02d.txt' % n_fold)

print(f'[With K-Fold] Profit: {profit}, f1-score: {f1}')

# Whole Dataset
print('[Training with entire dataset ...]')
train_dataset = lgb.Dataset(X, label=Y)
booster = lgb.train(params, train_dataset, num_boost_round=50000, verbose_eval=False)
booster.save_model('../Model/LightGBM/model_entire.txt')
print('[Model trained with entire dataset is saved]')
