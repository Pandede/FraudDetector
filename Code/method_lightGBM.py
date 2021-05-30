# from handler import *
from metrics import *
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# Dataset
train = pd.read_csv('../Data/train.csv', sep="|")
test = pd.read_csv('../Data/test.csv', sep="|")
label = pd.read_csv('../Data/realclass.csv', sep="|").to_numpy().reshape(-1)

# Training
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

feats = [i for i in train.columns if i != "fraud"]
sub_preds = np.zeros(test.shape[0])
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train[feats], train['fraud'])):
    train_x, train_y = train[feats].iloc[train_idx], train['fraud'].iloc[train_idx]
    valid_x, valid_y = train[feats].iloc[valid_idx], train['fraud'].iloc[valid_idx]
    print("Train Index:",train_idx,",Val Index:",valid_idx)

    params = {
        'nthread': 4,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'xentropy',
        'learning_rate': 0.01,
        'num_leaves': 64,
        'max_depth': 6,
        'bagging_fraction': 0.8,
        'feature_fraction': 0.7,
        'bagging_freq': 1,
    }
    if n_fold >= 0:
        dtrain = lgb.Dataset(train_x, label=train_y)
        dval = lgb.Dataset(valid_x, label=valid_y, reference=dtrain)
        bst = lgb.train(params, dtrain, num_boost_round=50000,valid_sets=[dtrain,dval], early_stopping_rounds=1000, verbose_eval=100)
        sub_preds += bst.predict(test[feats], num_iteration=bst.best_iteration) / folds.n_splits
# Testing
# print(label)
preds = [1 if i > 0.5 else 0 for i in sub_preds]
# print(preds)
profit = retailer_profit(label, preds)
print(f"profit:{profit}")
f1 = f1_score(label, preds)
print(f"f1:{f1}")
