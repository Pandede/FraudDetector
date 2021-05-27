import abc

import numpy as np
import pandas as pd


class ParentDataset(abc.ABC):
    def __init__(self, file: str = None, sep: str = '|'):
        self.data = None
        self.rows = 0

        if file is not None:
            self.read(file, sep=sep)

    def read(self, file: str, sep: str = '|'):
        self.data = pd.read_csv(file, sep=sep)
        self.rows = len(self.data)

    def get(self, n: int = 1, max_iter: int = 1e6):
        for indices in self.sampling(n, max_iter):
            yield self.data.values[indices]

    @abc.abstractmethod
    def sampling(self, n: int, max_iter: int):
        raise NotImplemented


class SequentialDataset(ParentDataset):
    def __init__(self, file: str = None, sep: str = '|'):
        super().__init__(file, sep)

    def sampling(self, n: int, max_iter: int):
        indices = np.arange(self.rows)
        max_iter = min(max_iter, self.rows // n + 1)
        for i in range(max_iter):
            yield indices[i * n:(i + 1) * n]


class ShuffleDataset(ParentDataset):
    def __init__(self, file: str = None, sep: str = '|'):
        super().__init__(file, sep)

    def sampling(self, n: int, max_iter: int):
        indices = np.random.permutation(self.rows)
        max_iter = min(max_iter, self.rows // n + 1)
        for i in range(max_iter):
            yield indices[i * n:(i + 1) * n]


class ImportantSamplingDataset(ParentDataset):
    def __init__(self, file: str = None, sep: str = '|'):
        super().__init__(file, sep)
        self.pos_index, self.neg_index = self.__get_pos_neg_index()

    def __get_pos_neg_index(self):
        labels = self.data['fraud'].values
        pos_index = np.argwhere(labels == 1)
        neg_index = np.argwhere(labels == 0)
        return pos_index.squeeze(), neg_index.squeeze()

    def sampling(self, n: int, max_iter: int):
        half = n // 2
        for _ in range(max_iter):
            pos_choices = np.random.choice(self.pos_index, n - half)
            neg_choices = np.random.choice(self.neg_index, half)
            yield np.hstack((pos_choices, neg_choices))
