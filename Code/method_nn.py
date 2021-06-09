import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from handler import *
from metrics import *


# Network
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer_1 = nn.Linear(9, 4)
        self.layer_2 = nn.Linear(4, 8)
        self.layer_3 = nn.Linear(8, 16)
        self.layer_4 = nn.Linear(16, 32)
        self.layer_out = nn.Linear(32, 1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.relu(self.layer_2(x))
        x = self.relu(self.layer_3(x))
        x = self.relu(self.layer_4(x))
        x = self.layer_out(x)

        return x


def calc_binary_acc(y_pred, y_test):
    correct_results_sum = (y_pred == y_test).sum().float()
    return correct_results_sum / len(y_test)


# Dataset
train_data = pd.read_csv('../Data/train.csv', sep="|")
feature_cols = train_data.columns[train_data.columns != 'fraud']

# Settings
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
ros = RandomOverSampler(random_state=0)

# Oversampling
train_feature_resampled, train_target_resampled = ros.fit_resample(train_data[feature_cols], train_data['fraud'])

# Configure neural network
epochs = 150
batch_size = 512
running_acc = np.zeros(epochs)
lr = 0.001
criterion = nn.BCEWithLogitsLoss()

# Initializing before loops
profit = 0
f1 = 0

for n_fold, (train_idx, validation_idx) in enumerate(folds.split(train_feature_resampled, train_target_resampled)):
    train_x = train_feature_resampled.iloc[train_idx].values
    train_y = train_target_resampled.iloc[train_idx].values
    validation_x = train_feature_resampled.iloc[validation_idx].values
    validation_y = train_target_resampled.iloc[validation_idx].values

    # Training
    train_x = torch.FloatTensor(train_x)
    train_y = torch.FloatTensor(train_y)
    validation_x = torch.FloatTensor(validation_x)
    validation_y = torch.FloatTensor(validation_y)

    train_dataset = TensorDataset(train_x, train_y)
    validation_dataset = TensorDataset(validation_x, validation_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    model = BinaryClassifier()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    for e in range(epochs):
        with tqdm(total=len(train_loader), ncols=130) as progress:
            for b, (x, y) in enumerate(train_loader):
                y = y.unsqueeze(1)
                prediction = model(x)
                loss = criterion(prediction, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                prediction = torch.sigmoid(prediction)
                prediction = torch.round(prediction).int()
                accuracy = calc_binary_acc(prediction, y)
                progress.set_description(
                    f'[Fold {n_fold:02d}][Epoch {e:03d}][Iteration {b:04d}][Loss = {loss:.4f}][Acc. = {accuracy * 100:.2f}%]'
                )
                progress.update(1)
    with torch.no_grad():
        for _, (x, y) in enumerate(validation_loader):
            prediction = torch.sigmoid(model(x))
            prediction = torch.round(prediction).int().squeeze()
            # Cross validation
            profit += retailer_profit(y.numpy(), prediction.numpy()) / folds.n_splits
            f1 += f1_score(y.numpy(), prediction.numpy()) / folds.n_splits / len(validation_loader)

# Testing
print(f'Profit: {profit}, f1-score: {f1}')
