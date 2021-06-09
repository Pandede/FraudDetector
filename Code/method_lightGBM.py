import lightgbm as lgb
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from metrics import *

# Dataset
train_data = pd.read_csv('../Data/train.csv', sep="|")
feature_cols = train_data.columns[train_data.columns != 'fraud']

# Settings
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
ros = RandomOverSampler(random_state=0)

# Oversampling
train_feature_resampled, train_target_resampled = ros.fit_resample(train_data[feature_cols], train_data['fraud'])

# Initialize before loops
profit = 0
f1 = 0
params = {
    'nthread': 4,
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'xentropy',
    'learning_rate': 0.01,
    'num_leaves': 4,
    'max_depth': 6,
    'bagging_fraction': 0.8,
    'feature_fraction': 0.7,
    'bagging_freq': 1,
}

for n_fold, (train_idx, validation_idx) in enumerate(folds.split(train_feature_resampled, train_target_resampled)):
    train_x = train_feature_resampled.iloc[train_idx].values
    train_y = train_target_resampled.iloc[train_idx].values
    validation_x = train_feature_resampled.iloc[validation_idx].values
    validation_y = train_target_resampled.iloc[validation_idx].values

    train_dataset = lgb.Dataset(train_x, label=train_y)
    validation_dataset = lgb.Dataset(validation_x, label=validation_y, reference=train_dataset)
    booster = lgb.train(params, train_dataset, num_boost_round=50000, valid_sets=[train_dataset, validation_dataset],
                        early_stopping_rounds=1000, verbose_eval=100)

    # Cross validation
    prediction = booster.predict(validation_x[feature_cols], num_iteration=booster.best_iteration)
    prediction = np.round(prediction)
    profit += retailer_profit(validation_y, prediction) / folds.n_splits
    f1 += f1_score(validation_y, prediction) / folds.n_splits

    # Save
    booster.save_model('../Model/LightGBM/model_%02d.txt' % n_fold)

print(f'Profit: {profit}, f1-score: {f1}')
