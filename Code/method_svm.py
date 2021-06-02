from sklearn.svm import SVC
from handler import *
from metrics import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import StratifiedKFold

# Initialize the dataset
train_dataset = pd.read_csv('../Data/train.csv', sep="|")
# train_dataset = ImportantSamplingDataset('../Data/train.csv', sep='|')

# Training begin

folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
feats = [i for i in train_dataset.columns if i != "fraud"]

profit = 0
f1 = 0
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_dataset[feats], train_dataset['fraud'])):
    train_x, train_y = train_dataset[feats].iloc[train_idx], train_dataset['fraud'].iloc[train_idx]
    valid_x, valid_y = train_dataset[feats].iloc[valid_idx], train_dataset['fraud'].iloc[valid_idx]
    # print("Train Index:",train_idx,",Val Index:",valid_idx)
    model = SVC(kernel='rbf')
    model.fit(train_x, train_y)
    valid_preds = model.predict(valid_x)

    # cross validation
    valid_preds = [1 if i > 0.5 else 0 for i in valid_preds]
    profit += retailer_profit(valid_y, valid_preds) / folds.n_splits
    f1 += f1_score(valid_y, valid_preds) / folds.n_splits


# Plot the accuracy curve
# plt.plot(running_acc)
# plt.show()

# Testing
# test = pd.read_csv('../Data/test.csv', sep="|")
# label = pd.read_csv('../Data/realclass.csv', sep="|").to_numpy().reshape(-1)

# y_pred = model.predict(test)
# profit = retailer_profit(label, y_pred)
print(f"profit:{profit}")

# f1 = f1_score(label, y_pred)
print(f"f1:{f1}")