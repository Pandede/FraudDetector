from sklearn.svm import SVC
from handler import *
from metrics import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

# Initialize the dataset
train_dataset = ImportantSamplingDataset('../Data/train.csv', sep='|')
model = SVC(kernel='rbf')

# Configure parameters
epochs = 10
batch_size = 100
max_iter = 100
running_acc = np.zeros(epochs)

# Training begin
for e in range(epochs):
    print('Epoch %03d' % e)
    for b, (x, y) in enumerate(train_dataset.get(batch_size, max_iter=max_iter)):
        model.fit(x, y)
        pred = model.predict(x)
        running_acc[e] += np.mean(pred == y) / max_iter
    print('Average acc. = %.04f' % running_acc[e])

# Plot the accuracy curve
plt.plot(running_acc)
plt.show()

# Testing
test = pd.read_csv('../Data/test.csv', sep="|")
label = pd.read_csv('../Data/realclass.csv', sep="|").to_numpy().reshape(-1)

y_pred = model.predict(test)
profit = retailer_profit(label, y_pred)
print(f"profit:{profit}")

f1 = f1_score(label, y_pred)
print(f"f1:{f1}")