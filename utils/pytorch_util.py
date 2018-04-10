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
    def __init__(self, weights=None, bin_size=0):
        self.weights = weights
        self.bin_size = bin_size

    @staticmethod
    def corrcoeff(targets, predictions):
        return np.corrcoef(targets.reshape(1, -1).squeeze(),
                           predictions.reshape(1, -1).squeeze())[0, 1]

    def weighted_corrcoef(self, targets, predictions):
        assert self.weights is not None, 'weighted corr coeff. needs weights'
        # input should be NXS
        return np.corrcoef((targets * self.weights).reshape(1, -1).squeeze(),
                           (predictions * self.weights).reshape(1, -1).squeeze())[0, 1]

    def hist_corrcoeff(self, targets, predictions):
        # input should be NxSxNumClasses
        assert self.bin_size > 0
        num_bins = int(np.ceil(targets.shape[1]/self.bin_size))
        corr = np.zeros((num_bins, 1), dtype=np.float32)
        for bin in range(num_bins-1):
            corr[bin] = np.corrcoef(targets[:, bin*self.bin_size:(bin+1)*self.bin_size, ].reshape(1, -1).squeeze(),
                                    predictions[:, bin*self.bin_size:(bin+1)*self.bin_size, ].reshape(1, -1).squeeze())[0, 1]

        corr[num_bins-1] = np.corrcoef(targets[:, (num_bins-1)*self.bin_size:, ].reshape(1, -1).squeeze().squeeze(),
                                       predictions[:, (num_bins-1)*self.bin_size:, ].reshape(1, -1).squeeze().squeeze())[0, 1]
        return corr

    def weighted_hist_corrcoeff(self, targets, predictions):
        assert self.weights is not None, 'weighted corr coeff. needs weights'
        assert self.bin_size > 0
        targets = targets * self.weights
        predictions = predictions * self.weights
        num_bins = int(np.ceil(targets.shape[1]/self.bin_size))
        corr = np.zeros((num_bins, 1), dtype=np.float32)
        for bin in range(num_bins-1):
            corr[bin] = np.corrcoef(targets[:, bin*self.bin_size:(bin+1)*self.bin_size, ].reshape(1, -1).squeeze(),
                                    predictions[:, bin*self.bin_size:(bin+1)*self.bin_size, ].reshape(1, -1).squeeze())[0, 1]

        corr[num_bins-1] = np.corrcoef(targets[:, (num_bins-1)*self.bin_size:, ].reshape(1, -1).squeeze().squeeze(),
                                       predictions[:, (num_bins-1)*self.bin_size:, ].reshape(1, -1).squeeze().squeeze())[0, 1]
        return corr


def crops_from_trial(X, y, crop_len, stride=0, time_last=True, dummy_idx=0, normalize=True):
    crop_len = int(crop_len)
    x_list, y_list = list(), list()
    if stride > 0:
        num_valid_crops = int((X.shape[0] - crop_len) / stride) + 1
    else:
        num_valid_crops = int(X.shape[0] // crop_len)

    for crop in range(num_valid_crops):
        if stride > 0:
            crop_idx = int(crop * stride)
        else:
            crop_idx = int(crop * crop_len)
            
        x_crop = X[crop_idx:crop_idx + crop_len, ]
        y_crop = y[crop_idx:crop_idx + crop_len, ]
        if normalize:
                y_crop = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y_crop.reshape(-1, 1)).squeeze()
                x_crop = exponential_running_standardize(x_crop, init_block_size=250, factor_new=0.001, eps=1e-4)
        
        x_list.append(
            np.expand_dims(x_crop.T if time_last else x_crop, axis=dummy_idx).astype(np.float32)
        )
        y_list.append(y_crop.astype(np.float32))
    return x_list, y_list


class ConcatCrops(Dataset):
    def __init__(self, x_crops, y_crops):
        assert len(x_crops) == len(y_crops)
        self._x_crops = x_crops
        self._y_crops = y_crops
        super(ConcatCrops, self).__init__()

    def __len__(self):
        return len(self._x_crops)

    def __getitem__(self, idx):
        return self._x_crops[idx], self._y_crops[idx]


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
    def __init__(self, X, y, window=1, stride=1, x2y_ratio=1, time_last=True, dummy_idx=0):
        '''
        param: X is an nd array with first dim is time axis
        y is
        '''
        self._xwindow = window
        self._ywindow = int(window / x2y_ratio)
        self._xstride = stride
        self._ystride = int(stride / x2y_ratio)
        self._num_samples = int((X.shape[0] - self._xwindow) / self._xstride) + 1
        assert self._num_samples <= int((y.shape[0] - self._ywindow) / self._ystride) + 1, \
            'number of examples X (%d) does not match number of labels y(%d)' % (self._num_samples, int((y.shape[0] - self._ywindow) / self._ystride) + 1)
        self._X = X
        self._y = y
        self.time_last = time_last
        self.dummy_idx = dummy_idx

    def __len__(self):
        return self._num_samples

    def __getitem__(self, idx):
        X = self._X[idx * self._xstride: idx * self._xstride + self._xwindow].squeeze().astype(np.float32)
        X = np.expand_dims(X.T if self.time_last else X, axis=self.dummy_idx).astype(np.float32)
        y = self._y[idx * self._ystride: idx * self._ystride + self._ywindow].squeeze().astype(np.float32)
        return X, y

