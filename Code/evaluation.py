import pickle

import catboost as cb
import lightgbm as lgb
import pandas as pd
from sklearn.metrics import f1_score

from metrics import *

# Dataset
test_x = pd.read_csv('../Data/test_modified.csv', sep='|', chunksize=10000)
test_y = pd.read_csv('../Data/realclass.csv', sep='|', chunksize=10000)

# Model
with open('../Model/RandomForest/model_reg_entire.pkl', 'rb') as f:
    model_rf = pickle.load(f)
model_lgb = lgb.Booster(model_file='../Model/LightGBM/model_entire.txt')
model_cb = cb.CatBoostClassifier().load_model('../Model/CatBoost/model_reg_entire.cbm')

profit = 0
f1 = 0
for x, y in zip(test_x, test_y):
    x, y = x.values, y.values.squeeze()
    pred_rf = model_rf.predict_proba(x)[:, 1]
    pred_lgb = model_lgb.predict(x)
    pred_cb = model_cb.predict_proba(x)[:, 1]
    prediction = (pred_rf + pred_lgb + pred_cb) >= 2
    profit += retailer_profit(y, prediction)
    f1 += f1_score(y, prediction) / 50

print(f'Profit: {profit}, f1-score: {f1}')
