import scipy.io as sio
import h5py
import numpy as np
from mat73_to_pickle import recursive_dict
import os
import scipy.signal
import sys
from glob import glob


def read_dataset_dir(dataset_path):
    # takes the path to the dataset and returns a dict. of lists, where for each subject a list contains the file names
    # (without extention) of all recordings is added to the dictionary.
    files = glob(dataset_path + '/*_header.mat')

    subject_pathes = dict()
    for file in files:
        if len(file.split('_')) > 1:
            subject_id = file.split('_')[-3]
            if subject_id in subject_pathes:
                subject_pathes[subject_id].append((os.path.join(dataset_path, '_'.join(file.split('_')[:-1]))))
            else:
                subject_pathes[subject_id] = [os.path.join(dataset_path, '_'.join(file.split('_')[:-1]))]

    return subject_pathes


def extract_names_from_header(header_path):
    # takes a header file path, read it and return the names of valid ecog-grid and seeg signals
    try:
        header = sio.loadmat(header_path)['H']['channels']
    except NotImplementedError:
        f = h5py.File(header_path, mode='r')
        header = recursive_dict(f['H/channels'])

    signalType = header['signalType']
    ecog_channels_idx = np.chararray.find(np.chararray.lower(signalType), 'ecog-grid') != -1
    seeg_channels_idx = np.chararray.find(np.chararray.lower(signalType), 'seeg') != -1
    soi_idx = np.bitwise_or(ecog_channels_idx, seeg_channels_idx)
    if np.all(soi_idx == False):
        raise KeyError('No ECoG-Grid or SEEG electrods found')
    if 'seizureOnset' in header:
        soz = header['seizureOnset']
    else:
        soz = np.zeros((soi_idx.shape[-1], 1))
    valid = soz == 0
    if 'rejected' in header:
        rejected = header['rejected']
        not_rejected = np.array([np.all(rejected[idx] == 0) for idx in range(len(rejected))])
        valid = np.bitwise_and(valid, not_rejected)

    soi_idx = np.bitwise_and(soi_idx, valid)
    return header['name'][soi_idx]


def extract_common_names(headers):
    # takes headers for the same subject and return the name of channels (ecog-grid + seeg) that are valid and available
    # in all recordings
    # we start by first header and
    signal_names = extract_names_from_header(headers[0] + '_header.mat')
    common_signals = set(signal_names)
    for header in headers[1:]:
        signal_names = extract_names_from_header(header + '_header.mat')
        common_signals = common_signals & set(signal_names)

    return list(common_signals)


def read_header_given_names(header_path, common_signals):
    # reads a header file and extract the data with valid signal names
    try:
        header = sio.loadmat(header_path)['H']['channels']
    except NotImplementedError:
        f = h5py.File(header_path, mode='r')
        header = recursive_dict(f['H/channels'])

    names = np.chararray.lower(header['name'])
    soi_idx = np.array([False]*names.shape[-1])
    for signal_name in common_signals:
        soi_idx = np.bitwise_or(soi_idx, np.chararray.find(names, signal_name.lower()) != -1)

    if 'headboxNumber' in header:
        ch_hbox = header['headboxNumber']
    else:
        ch_hbox = np.zeros_like(soi_idx)

    return soi_idx, ch_hbox


def read_day(day_path, common_signals):
    # reads a matlab header and data files and return the following
    # ch_val is a num_recordings * num_ecog_channel list of list. each value is dict with
    # 'ch_hbox' and 'val' if the channel is valid, otherwise, an empty dict.
    # ch_pos is num_ecog_channels list where each element is the MNI xyz coordinates.
    # target is list of targets for every recording session
    # srate is a list of the sampling rates for each recording session
    data_path = day_path + '_data.mat'
    header_path = day_path + '_header.mat'

    channels_idx, ch_hbox = read_header_given_names(header_path, common_signals)
    ch_val, srate, game_type, target = read_raw_data(data_path, channels_idx, ch_hbox)
    return ch_val, target, srate, game_type


def read_raw_data(data_path, ecog_channels_idx, hbox):
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
        # extract raw ecog-grid data
        ecog_channels = recording['ampData'][ecog_channels_idx].astype(np.float32)
        # apply CAR
        ecog_channels = car(ecog_channels, hbox[ecog_channels_idx])
        # apply high pass filtering with cut-off freq. 0.15 Hz
        srate = recording['srate'].tolist()
        ecog_channels = highpass_filtering(ecog_channels, 0.15, srate)
        # add to the list
        ch_val.append(ecog_channels)
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
    ecog_channels_idx = np.chararray.find(np.chararray.lower(signalType), 'ecog-grid') != -1
    if np.all(ecog_channels_idx == False):
        raise KeyError('No ECoG-Grid electrods found')
    if 'seizureOnset' in header:
        soz = header['seizureOnset']
    else:
        soz = np.zeros((np.sum(ecog_channels_idx), 1))
    valid = soz == 0
    if 'rejected' in header:
        rejected = header['rejected']
        not_rejected = np.array([np.all(rejected[idx] == 0) for idx in range(len(rejected))])
        valid = np.bitwise_and(valid, not_rejected)
    if 'headboxNumber' in header:
        ch_hbox = header['headboxNumber']
    else:
        ch_hbox = np.zeros_like(soz)

    ecog_channels_idx = np.bitwise_and(ecog_channels_idx, valid)
    return ecog_channels_idx, ch_hbox


def car(channels, hbox):
    # computes the mean and std per headbox and then standardize the channels per hbox
    assert channels.shape[0] == len(hbox)
    unique_hbox = np.unique(hbox).tolist()

    for hb in unique_hbox:
        # use only valid signals to compute mean and SD
        hb_idx = hbox == hb
        mean = np.mean(channels[hb_idx, ], dtype=np.float32)
        channels -= mean

    return channels


def highpass_filtering(data, cut_feq, fs):
    # data is CxT
    for filter_order in reversed(range(10)):
        b, a = scipy.signal.butter(filter_order, cut_feq / (fs / 2.0), btype='highpass')
        if np.all(np.abs(np.roots(a)) < 1):
            break
    assert filter_order > 0, 'Could not find a proper filter order'
    assert a[0] == 1.0, (
        "a[0] should normally be zero, did you accidentally supply b?\n"
        "a: {:s}".format(str(a)))
    # from http://stackoverflow.com/a/8812737/1469195

    return scipy.signal.lfilter(b, a, data, axis=-1)


# def read_dataset_dir(dataset_path):
#     dirs = os.listdir(dataset_path)
#     subject_pathes = dict()
#     for dir in dirs:
#         if len(dir.split('_')) > 1:
#             subject_id = dir.split('_')[1]
#             if subject_id in subject_pathes:
#                 subject_pathes[subject_id].append(os.path.join(dataset_path, dir))
#             else:
#                 subject_pathes[subject_id] = [os.path.join(dataset_path, dir)]
#
#     return subject_pathes

def create_new_dataset(recording_name, output_file, file_format):
    common_signals = extract_common_names(recording_name)
    trial = 0
    if file_format == 'hdf':
        with h5py.File(output_file, 'x') as hdf:
            for day_path in recording_name:
                for ch_val, target, srate, game_type in zip(*read_day(day_path, common_signals)):
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

    else:
        dataset_dict = dict()
        for day_path in recording_name:
            for ch_val, target, srate, game_type in zip(*read_day(day_path, common_signals)):
                trial += 1
                dataset_dict['trial%d' % trial] = {'X': ch_val,
                                                   'y': target,
                                                   'srate': srate,
                                                   'gameType': game_type}
        sio.savemat(output_file, dataset_dict)


def create_dataset(input_dirs, output_file, file_format):
    if file_format == 'hdf':
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

    else:
        trial = 0
        dataset_dict = dict()
        for day_path in input_dirs:
            for ch_val, target, srate, game_type in zip(*read_day(day_path)):
                trial += 1
                dataset_dict['trial%d' % trial] = {'X': ch_val,
                                                   'y': target,
                                                   'srate': srate,
                                                   'gameType': game_type}
        sio.savemat(output_file, dataset_dict)


def generate_toy_dataset(root, name, size):
    import random
    import shutil
    path = os.path.join(root, name)
    if not os.path.exists(path):
        os.mkdir(path)

    # generate data file
    data_path = os.path.join(path, 'data.txt')
    with open(data_path, 'w') as fout:
        for _ in range(size):
            length = random.randint(1, 50)
            seq = []
            for _ in range(length):
                seq.append(str(random.randint(0, 9)))
            fout.write("\t".join([" ".join(seq), " ".join(reversed(seq))]))
            fout.write('\n')

    # generate vocabulary
    src_vocab = os.path.join(path, 'vocab.source')
    with open(src_vocab, 'w') as fout:
        fout.write("\n".join([str(i) for i in range(10)]))
    tgt_vocab = os.path.join(path, 'vocab.target')
    shutil.copy(src_vocab, tgt_vocab)


def main(dataset_path, extension):
    all_subject_pathes = read_dataset_dir(dataset_path)
    for subject_id, subject_pathes in all_subject_pathes.items():
        subject_dataset_path = os.path.join(dataset_path, subject_id + extension)
        try:
            print('Creating dataset for subject', subject_id)
            create_dataset(subject_pathes, subject_dataset_path, sys.argv[2])
        except KeyError:
            if extension == '.h5':
                os.remove(subject_dataset_path)
            continue


def new_main(dataset_path, extension):
    all_subject_pathes = read_dataset_dir(dataset_path)
    for subject_id, subject_pathes in all_subject_pathes.items():
        subject_dataset_path = os.path.join(dataset_path, 'new_' + subject_id + extension)
        try:
            print('Creating dataset for subject', subject_id)
            create_new_dataset(subject_pathes, subject_dataset_path, sys.argv[2])
        except KeyError as e:
            if extension == '.h5':
                os.remove(subject_dataset_path)
            print(e)
            continue


if __name__ == '__main__':
    assert len(sys.argv) > 2, 'You should pass the path to the dataset and output file format'
    assert sys.argv[2] == 'hdf' or sys.argv[2] == 'mat', sys.argv[2] + 'is not supported'
    dataset_path = sys.argv[1]
    extension = '.h5' if sys.argv[2] == 'hdf' else '.mat'
    new_main(dataset_path, extension)


