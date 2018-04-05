import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
from braindecode.datautil.signalproc import exponential_running_standardize
from sklearn.preprocessing import MinMaxScaler
from torch.nn.modules.loss import _WeightedLoss


class WeightedMSE(_WeightedLoss):
    # input should be NXSXNumClasses
    def forward(self, input, target):
        if torch.is_tensor(self.weight):
            weight = Variable(self.weight)
        else:
            weight = self.weight
        return torch.mean(weight * ((input - target) ** 2).mean(dim=0))


class CorrCoeff:
    def __init__(self, weights=None):
        self.weights = weights

    @staticmethod
    def corrcoeff(targets, predictions):
        return np.corrcoef(targets.reshape(1, -1).squeeze(),
                           predictions.reshape(1, -1).squeeze())[0, 1]

    def weighted_corrcoef(self, targets, predictions):
        assert self.weights is not None, 'weighted corr coeff. needs weights'
        # input should be NXS
        return np.corrcoef((targets * self.weights).reshape(1, -1).squeeze(),
                           (predictions * self.weights).reshape(1, -1).squeeze())[0, 1]


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


def crops_from_trial(X, y, crop_len, stride=0, time_last=True, normalize=True):
    x_list, y_list = list(), list()
    if stride > 0:
        num_valid_crops = int((X.shape[0] - crop_len) / stride) + 1
    else:
        num_valid_crops = int(X.shape[0] // crop_len)

    for crop in range(num_valid_crops):
        if stride > 0:
            crop_idx = crop * stride
        else:
            crop_idx = crop * crop_len
            
        x_crop = X[crop_idx:crop_idx + crop_len, ]
        y_crop = y[crop_idx:crop_idx + crop_len, ]
        if normalize:
                y_crop = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y_crop.reshape(-1, 1)).squeeze()
                x_crop = exponential_running_standardize(x_crop, init_block_size=250, factor_new=0.001, eps=1e-4)
        
        x_list.append(x_crop.T.astype(np.float32) if time_last else x_crop.astype(np.float32))
        y_list.append(y_crop.astype(np.float32))

    return x_list, y_list


class ConcatCrops(Dataset):
    def __init__(self, x_crops, y_crops, dummy_idx=0):
        assert len(x_crops) == len(y_crops)
        self._x_crops = x_crops
        self._y_crops = y_crops
        self.dummy_idx = dummy_idx
        super(ConcatCrops, self).__init__()

    def __len__(self):
        return len(self._x_crops)

    def __getitem__(self, idx):
        return np.expand_dims(self._x_crops[idx], axis=self.dummy_idx), self._y_crops[idx]


class ConcatDataset(Dataset):
    def __init__(self, datasets, batch_first=False, time_last=False):
        self._len = min([len(dataset) for dataset in datasets])
        self.datasets = list(datasets)
        self.batch_first = batch_first
        self.time_last = time_last
        super(ConcatCrops, self).__init__()

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

