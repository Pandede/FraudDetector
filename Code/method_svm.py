from sklearn.svm import SVC
from Code.handler import *
import numpy as np
import matplotlib.pyplot as plt

# Initialize the dataset
train_dataset = ImportantSamplingDataset('../Data/train.csv', sep='|')
model = SVC(kernel='rbf')

# Configure parameters
epochs = 100
batch_size = 30
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