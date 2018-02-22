import scipy.io as sio
import h5py
import numpy as np
from mat73_to_pickle import recursive_dict
import os
import scipy.signal
import sys
from torch.utils.data import Dataset
from torch.autograd import Variable


def read_day(day_path):
    # reads a matlab header and data files and return the following
    # ch_val is a num_recordings * num_ecog_channel list of list. each value is dict with
    # 'ch_hbox' and 'val' if the channel is valid, otherwise, an empty dict.
    # ch_pos is num_ecog_channels list where each element is the MNI xyz coordinates.
    # target is list of targets for every recording session
    # srate is a list of the sampling rates for each recording session
    relative_path = os.path.split(day_path)[-1]
    data_path = os.path.join(day_path, 'alignedData', relative_path + '_data.mat')
    header_path = os.path.join(day_path, 'alignedData', relative_path + '_header.mat')

    ch_hbox, dims, ecog_channels_idx, sorting_idx, valid = read_header(header_path)
    ch_val, srate, game_type, target = read_raw_data(data_path, ecog_channels_idx, sorting_idx, valid, ch_hbox, dims)
    assert np.prod(ch_val[0].shape[:-1]) == len(valid) == len(ch_hbox)
    return ch_val, target, srate, game_type


def read_raw_data(data_path, ecog_channels_idx, sorting_idx, valid, hbox, dims):
    try:
        data = sio.loadmat(data_path)['D']
    except NotImplementedError:
        f = h5py.File(data_path, mode='r')
        data = recursive_dict(f)['D']
    ch_val = []
    targets = []
    srates= []
    gameType = []
    # for each recording
    for recording in data:
        # raw ecog data
        ecog_channels = recording['ampData'][ecog_channels_idx].astype(np.float32)
        # sort the channels
        sorted_ch = ecog_channels[sorting_idx]
        # fill non-valid channels with zeros
        sorted_ch[np.bitwise_not(valid)] = 0
        # apply CAR
        car(sorted_ch, valid, hbox)
        # apply high pass filtering
        srate = recording['srate'].tolist()
        filtered_ch = highpass_filtering(sorted_ch, 1.5, srate)
        # reshape to create one image per time step and append
        ch_val.append(np.reshape(filtered_ch, (*dims, -1)))
        # extract raw target values
        targets.append(recording['tracker'].astype(np.float32))
        assert ch_val[-1].shape[-1] == targets[-1].shape[0]
        # append sample rate
        srates.append(srate)
        # append game type
        gameType.append(recording['gameType'])

    if len(ch_val) > 1:
        assert all([ch.shape[:-1] == ch_val[0].shape[:-1] for ch in ch_val[1:]])
    return ch_val, srates, gameType, targets


def read_header(header_path):
    try:
        header = sio.loadmat(header_path)['H']['channels']
    except NotImplementedError:
        f = h5py.File(header_path, mode='r')
        header = recursive_dict(f['H/channels'])

    signalType = header['signalType']
    ecog_channels_idx = signalType == 'ECoG-Grid'
    if np.all(ecog_channels_idx == False):
        raise KeyError('No ECoG-Grid electrods found')
    if 'seizureOnset' in header:
        soz = header['seizureOnset'][ecog_channels_idx]
    else:
        soz = np.zeros((np.sum(ecog_channels_idx), 1))
    valid = soz == 0
    if 'rejected' in header:
        rejected = header['rejected'][ecog_channels_idx]
        not_rejected = np.array([np.all(rejected[idx] == 0) for idx in range(len(rejected))])
        valid = np.bitwise_and(valid, not_rejected)
    if 'headboxNumber' in header:
        ch_hbox = header['headboxNumber'][ecog_channels_idx]
    else:
        ch_hbox = np.zeros_like(soz)
    # sort the channels' attributes
    names = header['name'][ecog_channels_idx]
    rows = [name[-2] for name in names]
    cols = [name[-1] for name in names]
    dims = (len(np.unique(rows)), len(np.unique(cols)))
    sorting_idx = np.lexsort((cols, rows))
    ch_hbox = ch_hbox[sorting_idx]
    valid = valid[sorting_idx]
    return ch_hbox, dims, ecog_channels_idx, sorting_idx, valid


def car(ecog, valid, hbox):
    # computes the mean and std per headbox and then standardize the channels per hbox
    assert len(ecog) == len(hbox)
    if not isinstance(ecog, np.ndarray):
        ecog = np.array(ecog, dtype=np.float32)
    if not isinstance(hbox, np.ndarray):
        hbox = np.array(hbox)

    unique_hbox = np.unique(hbox).tolist()
    for hb in unique_hbox:
        hb_idx = np.bitwise_and(hbox == hb, valid)
        mean = np.mean(ecog[hb_idx, :], dtype=np.float32)
        std = np.std(ecog[hb_idx, :], dtype=np.float32)
        assert std != 0
        ecog[hb_idx, :] -= mean
        ecog[hb_idx, :] /= std

    return ecog


def highpass_filtering(data, cut_feq, fs):
    for filter_order in reversed(range(10)):
        b, a = scipy.signal.butter(filter_order, cut_feq / (fs / 2.0), btype='highpass')
        if np.all(np.abs(np.roots(a)) < 1):
            break
    assert filter_order > 0, 'Could not find a proper filter order'
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a)))
    # from http://stackoverflow.com/a/8812737/1469195

    return scipy.signal.lfilter(b, a, data, axis=1)


def read_dataset_dir(dataset_path):
    dirs = os.listdir(dataset_path)
    subject_pathes = dict()
    for dir in dirs:
        if len(dir.split('_')) > 1:
            subject_id = dir.split('_')[1]
            if subject_id in subject_pathes:
                subject_pathes[subject_id].append(os.path.join(dataset_path, dir))
            else:
                subject_pathes[subject_id] = [os.path.join(dataset_path, dir)]

    return subject_pathes


def create_dataset(input_dirs, output_file):
    with h5py.File(output_file, 'x') as hdf:
        trial = 0
        for day_path in input_dirs:
            for ch_val, target, srate, game_type in zip(*read_day(day_path)):
                trial += 1
                # create a group
                grp = hdf.create_group('trial%d' % trial)
                # create the dataset
                grp.create_dataset('X', data=ch_val)
                grp.create_dataset('y', data=target)
                # add the sampling rate to the attributes
                grp.attrs['srate'] = srate
                # add game type
                grp.attrs['gameType'] = game_type


class ECoGDatast(Dataset):
    def __init__(self, dataset_file, trial, window=1, stride=1, flatten=True):
        # this loads all the trials into memory
        self._trial = dataset_file[trial]
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
        return np.concatenate(X, axis=0), np.concatenate(y, axis=0)


def train(model, data_loader, optimizer, loss_fun, cuda=False):
    model.train()
    for data, target in data_loader:
        if cuda:
            data, target = Variable(data.squeeze().cuda()), Variable(target.squeeze().cuda())
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
            data, target = Variable(data.squeeze().cuda()), Variable(target_cpu.squeeze().cuda())
        else:
            data, target = Variable(data.squeeze()), Variable(target_cpu.squeeze())
        output = model(data).squeeze()
        avg_corr += np.corrcoef(target_cpu[:, -output.size()[-1]:].data.numpy().squeeze(),
                                output.data.cpu().numpy().squeeze())[0, 1]
        avg_loss += loss_fun(output, target[:, -output.size()[-1]:])

    writer.add_scalar('corr', avg_corr / len(data_loader.dataset), epoch)
    writer.add_scalar('loss', avg_loss / len(data_loader.dataset), epoch)
    return avg_loss, avg_corr


if __name__ == '__main__':
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        all_subject_pathes = read_dataset_dir(dataset_path)
        for subject_id, subject_pathes in all_subject_pathes.items():
            subject_dataset_path = os.path.join(dataset_path, subject_id + '.h5')
            try:
                print('Creating dataset for subject', subject_id)
                create_dataset(subject_pathes, subject_dataset_path)
            except KeyError:
                os.remove(subject_dataset_path)
                continue
    else:
        print('You should pass the path to the dataset')

