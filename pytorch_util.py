import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable


class ECoGDatast(Dataset):
    def __init__(self, dataset_file, trial, window=1, stride=1, flatten=True, full_load=True):
        # this loads all the trials into memory
        self._trial = dataset_file[trial]
        if full_load:
            self._X = self._trial['X'][:]
            self._y = self._trial['y'][:]
        else:
            self._X = self._trial['X']
            self._y = self._trial['y']
        assert self._X.shape[-1] == self._y.shape[0]
        self._num_samples = self._X.shape[-1]
        self._game_type = self._trial.attrs['gameType']
        self._srate = self._trial.attrs['srate']
        self._window = window
        self._stride = stride
        self._flatten = flatten

    def __len__(self):
        return int((self._num_samples - self._window) / self._stride) + 1

    def __getitem__(self, idx):
        X = self._X[:, :, idx * self._stride:idx * self._stride + self._window].squeeze().astype(np.float32)
        y = self._y[idx * self._stride:idx * self._stride + self._window].squeeze().astype(np.float32)
        if self._flatten:
            X = X.reshape((-1, self._window))
        return X, y

    def srate(self):
        return self._srate


class ConcatDataset(Dataset):
    def __init__(self, datasets):
        self._len = min([len(dataset) for dataset in datasets])
        self.datasets = list(datasets)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        X = []
        y = []
        for dataset in self.datasets:
            dataset_X, dataset_y = dataset.__getitem__(idx)
            X.append(np.expand_dims(dataset_X, axis=0))
            y.append(np.expand_dims(dataset_y, axis=0))
        return np.concatenate(X, axis=0).astype(np.float32), np.concatenate(y, axis=0).astype(np.float32)


def train(model, data_loader, optimizer, loss_fun, cuda=False):
    model.train()
    for data, target in data_loader:
        if cuda:
            data, target = Variable(data.squeeze().cuda()), Variable(target.squeeze().cuda())
            # might be helpfull to clear cache
            torch.cuda.empty_cache()
        else:
            data, target = Variable(data.squeeze()), Variable(target.squeeze())
        optimizer.zero_grad()
        output = model(data).squeeze()
        loss = loss_fun(output, target[:, -output.size()[-1]:])
        loss.backward()
        optimizer.step()


def metrics(model, data_loader, loss_fun, writer, epoch, cuda):
    model.eval()
    # loop over the training dataset
    avg_corr = []
    avg_loss = 0
    for data, target_cpu in data_loader:
        if cuda:
            data, = Variable(data.squeeze().cuda(), volatile=True)
            target = Variable(target_cpu.squeeze().cuda(), volatile=True)
        else:
            data, target = Variable(data.squeeze()), Variable(target_cpu.squeeze())
        output = model(data).squeeze()
        avg_corr += np.corrcoef(target_cpu[:, -output.size()[-1]:].data.numpy().squeeze(),
                                output.data.cpu().numpy().squeeze())[0, 1]
        avg_loss += loss_fun(output, target[:, -output.size()[-1]:])

    writer.add_scalar('corr', avg_corr / len(data_loader.dataset), epoch)
    writer.add_scalar('loss', avg_loss / len(data_loader.dataset), epoch)
    return avg_loss, avg_corr
