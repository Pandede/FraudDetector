import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from handler import *
from metrics import *
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import RandomOverSampler

# Initialize the dataset
# train_dataset = ImportantSamplingDataset('../Data/train.csv', sep='|')
# Initialize the dataset
train_dataset = pd.read_csv('../Data/train.csv', sep="|")

# Configure parameters
epochs = 300
batch_size = 128
running_acc = np.zeros(epochs)
lr = 0.005

# Network
class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        self.layer_1 = nn.Linear(9, 64) 
        self.layer_2 = nn.Linear(64, 64)
        self.layer_out = nn.Linear(64, 1) 
        
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(32)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        # x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        # x = self.batchnorm2(x)
        # x = self.dropout(x)
        x = self.layer_out(x)
        
        return x

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))

    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    
    return acc




folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
feats = [i for i in train_dataset.columns if i != "fraud"]

# ros = RandomOverSampler(random_state=0)
# train_feature_resampled, train_target_resampled = ros.fit_resample(train_dataset[feats], train_dataset['fraud'])

train_feature_resampled = train_dataset[feats]
train_target_resampled = train_dataset['fraud']
profit = 0
f1 = 0

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(train_feature_resampled, train_target_resampled)):
    train_x, train_y = train_feature_resampled.iloc[train_idx], train_target_resampled.iloc[train_idx]
    valid_x, valid_y = train_feature_resampled.iloc[valid_idx], train_target_resampled.iloc[valid_idx]
    
    train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(train_x.values), torch.FloatTensor(train_y.values))
    valid_ds = torch.utils.data.TensorDataset(torch.FloatTensor(valid_x.values), torch.FloatTensor(valid_y.values))
    train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=False, drop_last=False)


    model = binaryClassification()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Training
    for e in range(epochs):
        print('Epoch %03d' % e)
        for _, (x, y) in enumerate(train_dataloader):
            optimizer.zero_grad()
            y_pred = model(x)

            loss = criterion(y_pred, y.unsqueeze(1))
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        for _, (x, y) in enumerate(valid_dataloader):
            valid_preds = model(x)
            valid_preds = torch.round(torch.sigmoid(valid_preds)).int().squeeze()
            # print(valid_preds.size())
            # cross validation
            # valid_preds = [1 if torch.sigmoid(i) > 0.5 else 0 for i in valid_preds]
            profit += retailer_profit(y.int().tolist(), valid_preds.tolist())
            f1 += f1_score(y, valid_preds)


# Testing
print(f"profit:{profit/folds.n_splits}")
print(f"f1:{f1/folds.n_splits/len(valid_dataloader)}")

