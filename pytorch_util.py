import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


def load_trial(dataset_file, trial, full_load=True):
    trial = dataset_file[trial]
    if full_load:
        X = trial['X'][:]
        y = trial['y'][:]
    else:
        X = trial['X']
        y = trial['y']

    return X, y, trial.attrs['srate'], trial.attrs['gameType']


class ECoGDatast(Dataset):
    def __init__(self, X, y, window=1, stride=1, x2y_ratio=1):
        '''
        param: X is an nd array with first dim is time axis
        y is
        '''
        self._xwindow = window
        self._ywindow = int(window / x2y_ratio)
        self._xstride = stride
        self._ystride = int(stride / x2y_ratio)
        self._num_samples = int((X.shape[0] - self._xwindow) / self._xstride) + 1
        assert self._num_samples <= int((y.shape[0] - self._ywindow) / self._ystride) + 1,\
            'number of examples X (%d) does not match number of labels y(%d)' % (self._num_samples, int((y.shape[0] - self._ywindow) / self._ystride) + 1)
        self._X = X
        self._y = y

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        X = self._X[idx * self._xstride: idx * self._xstride + self._xwindow].squeeze().astype(np.float32)
        y = self._y[idx * self._ystride: idx * self._ystride + self._ywindow].squeeze().astype(np.float32)
        if y.ndim < 2:
            y = np.expand_dims(y, axis=1)
        return X, y


class ConcatDataset(Dataset):
    def __init__(self, datasets, batch_first=False, time_last=False):
        self._len = min([len(dataset) for dataset in datasets])
        self.datasets = list(datasets)
        self.batch_first = batch_first
        self.time_last = time_last

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        X = []
        y = []
        if self.batch_first:
            for dataset in self.datasets:
                dataset_X, dataset_y = dataset.__getitem__(idx)
                if self.time_last:
                   dataset_X = dataset_X.transpose()

                X.append(np.expand_dims(dataset_X, axis=0))
                y.append(np.expand_dims(dataset_y, axis=0))
            return np.concatenate(X, axis=0).astype(np.float32), np.concatenate(y, axis=0).astype(np.float32)
        else:
            for dataset in self.datasets:
                dataset_X, dataset_y = dataset.__getitem__(idx)
                if self.time_last:
                    dataset_X = dataset_X.transpose()
                X.append(np.expand_dims(dataset_X, axis=1))
                y.append(np.expand_dims(dataset_y, axis=0))
            return np.concatenate(X, axis=1).astype(np.float32), np.concatenate(y, axis=0).astype(np.float32)

