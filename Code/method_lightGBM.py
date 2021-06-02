# from handler import *
from metrics import *
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler

# class ImportantSamplingDataset:
#     def __init__(self, x, y, sep: str = '|'):
#         self.x = x
#         self.y = y
#         self.pos_index, self.neg_index = self.__get_pos_neg_index()

#     def __get_pos_neg_index(self):
#         pos_index = np.argwhere(self.y == 1)
#         neg_index = np.argwhere(self.y == 0)
#         return pos_index.squeeze(), neg_index.squeeze()

#     def sampling(self, n: int, max_iter: int):
#         half = n // 2
#         for _ in range(max_iter):
#             pos_choices = np.random.choice(self.pos_index, n - half)
#             neg_choices = np.random.choice(self.neg_index, half)
#             pos_x = self.x[pos_choices]
#             pos_y = self.y[pos_choices]
#             neg_x = self.x[neg_choices]
#             neg_y = self.y[neg_choices]
#             yield np.hstack((pos_x, neg_x)), np.hstack((pos_y, neg_y))

# Dataset
train = pd.read_csv('../Data/train.csv', sep="|")
# test = pd.read_csv('../Data/test.csv', sep="|")
# label = pd.read_csv('../Data/realclass.csv', sep="|").to_numpy().reshape(-1)

# Training
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
feats = [i for i in train.columns if i != "fraud"]

ros = RandomOverSampler(random_state=0)
train_feature_resampled, train_target_resampled = ros.fit_resample(train[feats], train['fraud'])

profit = 0
f1 = 0

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_feature_resampled, train_target_resampled)):
    train_x, train_y = train_feature_resampled.iloc[train_idx], train_target_resampled.iloc[train_idx]
    valid_x, valid_y = train_feature_resampled.iloc[valid_idx], train_target_resampled.iloc[valid_idx]

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
        # cross validation
        valid_preds = bst.predict(valid_x[feats], num_iteration=bst.best_iteration)
        valid_preds = [1 if i > 0.5 else 0 for i in valid_preds]
        profit += retailer_profit(valid_y, valid_preds) / folds.n_splits
        f1 += f1_score(valid_y, valid_preds) / folds.n_splits
        # break
print(f"profit:{profit}")
print(f"f1:{f1}")



# Testing
# print(label)
# preds = [1 if i > 0.5 else 0 for i in sub_preds]
# print(preds)
# profit = retailer_profit(label, preds)
# print(f"profit:{profit}")
# f1 = f1_score(label, preds)
# print(f"f1:{f1}")
