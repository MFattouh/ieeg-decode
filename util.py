import scipy.io as sio
import h5py
import numpy as np
from mat73_to_pickle import recursive_dict
import os
import scipy.signal
import sys
from pandas import HDFStore, DataFrame, Series


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
    ch_val, srate, target = read_raw_data(data_path, ecog_channels_idx, sorting_idx, valid, ch_hbox, dims)
    assert np.prod(ch_val[0].shape[:-1]) == len(valid) == len(ch_hbox)
    return ch_val, target, srate


def read_raw_data(data_path, ecog_channels_idx, sorting_idx, valid, hbox, dims):
    try:
        data = sio.loadmat(data_path)['D']
    except NotImplementedError:
        f = h5py.File(data_path, mode='r')
        data = recursive_dict(f)['D']
    ch_val = []
    targets = []
    srates= []

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
        assert ch_val[-1].shape[-1] == targets[-1].shape[-1]
        # append sample rate
        srates.append(srate)

    if len(ch_val) > 1:
        assert all([ch.shape[:-1] == ch_val[0].shape[:-1] for ch in ch_val[1:]])
    return ch_val, srates, targets


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
    with HDFStore(output_file, 'w') as hdf:
        trial = 0
        for day_path in input_dirs:
            for ch_val, target, srate in zip(*read_day(day_path)):
                trial += 1
                # create a data frame
                df = DataFrame({'X': [ch_val[:, :, idx].squeeze() for idx in range(ch_val.shape[-1])], 'y': target})
                # put the dataframe into the store
                hdf.put('trial%d' % trial, df, 'f')
                # add the sampling rate as a metadata
                hdf.get_storer('trial%d' % trial).attrs.metadata = {'srate': srate}


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




