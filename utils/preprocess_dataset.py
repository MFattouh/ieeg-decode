import click
import scipy.io as sio
import h5py
import numpy as np
from mat73_to_pickle import recursive_dict
import os
import scipy.signal
from glob import glob
from resampy import resample


def read_dataset_dir(dataset_path):
    # takes the path to the dataset and returns a dict. of lists, where for each subject a list contains the file names
    # (without extention) of all recordings is added to the dictionary.
    files = glob(dataset_path + '/*_header.mat')

    subject_pathes = dict()
    for file_path in files:
        # check if a data file also exists
        assert os.path.isfile(file_path.rsplit('_', 1)[0] + '_data.mat'), \
            'no matching data file  found for header file ' + file_path
        file_name = os.path.basename(file_path)
        if len(file_name.split('_')) > 1:
            subject_id = file_name.split('_')[-3]
            if subject_id in subject_pathes:
                subject_pathes[subject_id].append('_'.join(file_name.split('_')[:-1]))
            else:
                subject_pathes[subject_id] = ['_'.join(file_name.split('_')[:-1])]

    return subject_pathes


def read_header(header_path):
    try:
        header = sio.loadmat(header_path)['H']['channels']
    except NotImplementedError:
        f = h5py.File(header_path, mode='r')
        header = recursive_dict(f['H/channels'])
    header_keys = dict([(key.lower().replace('_', '').replace('-', ''), key) for key in header.keys()])
    signal_type = [stype.replace('_', '').replace('-', '').lower() for stype in header[header_keys['signaltype']].tolist()]
    ecog_grid_channels_idx = [stype.find('ecoggrid') != -1 for stype in signal_type]
    ecog_strip_channels_idx = [stype.find('ecogstrip') != -1 for stype in signal_type]
    seeg_channels_idx = [stype.find('seeg') != -1 for stype in signal_type]
    ieeg_idx = np.bitwise_or(np.bitwise_or(ecog_grid_channels_idx, ecog_strip_channels_idx), seeg_channels_idx)
    if np.all(ieeg_idx == False):
        raise KeyError('No ECoG-Grid, ECoG-Strip or SEEG were electrods found!')
    if 'seizureonset' in header_keys:
        soz = header[header_keys['seizureonset']]
        if not np.all(soz == 0):
            print('found soz channels')
            print(header[header_keys['name']][soz == 1])
    else:
        soz = np.zeros((ieeg_idx.shape[-1], 1)).squeeze()
    valid = soz == 0
    if 'rejected' in header_keys:
        rejected = header[header_keys['rejected']]
        not_rejected = np.array([np.all(rejected[idx] == 0) for idx in range(len(rejected))])
        if not np.all(not_rejected == True):
            print('found rejected channels')
            print(header[header_keys['name']][np.bitwise_not(not_rejected)])
        valid = np.bitwise_and(valid, not_rejected)
    if 'headboxnumber' in header_keys:
        ch_hbox = header[header_keys['headboxnumber']]
    else:
        ch_hbox = np.zeros_like(soz)

    ieeg_idx = np.bitwise_and(ieeg_idx, valid)

    return ieeg_idx, ch_hbox[ieeg_idx]


def extract_names_from_header(header_path):
    # takes a header file path, read it and return the names of valid ecog-grid and seeg signals
    try:
        header = sio.loadmat(header_path)['H']['channels']
    except NotImplementedError:
        f = h5py.File(header_path, mode='r')
        header = recursive_dict(f['H/channels'])

    header_keys = dict([(key.lower().replace('_', '').replace('-', ''), key) for key in header.keys()])
    signal_type = [stype.replace('_', '').replace('-', '').lower() for stype in header[header_keys['signaltype']].tolist()]
    ecog_grid_channels_idx = [stype.find('ecoggrid') != -1 for stype in signal_type]
    ecog_strip_channels_idx = [stype.find('ecogstrip') != -1 for stype in signal_type]
    seeg_channels_idx = [stype.find('seeg') != -1 for stype in signal_type]
    ieeg_idx = np.bitwise_or(np.bitwise_or(ecog_grid_channels_idx, ecog_strip_channels_idx), seeg_channels_idx)

    if np.all(ieeg_idx == False):
        raise KeyError('No ECoG-Grid, ECoG-Strip or SEEG electrods found')
    if 'seizureonset' in header_keys:
        soz = header[header_keys['seizureonset']]
        if not np.all(soz == 0):
            print('found soz channels')
            print(header[header_keys['name']][soz == 1])
    else:
        soz = np.zeros((ieeg_idx.shape[-1], 1)).squeeze()
    valid = soz == 0
    if 'rejected' in header_keys:
        rejected = header[header_keys['rejected']]
        not_rejected = np.array([np.all(rejected[idx] == 0) for idx in range(len(rejected))])
        if not np.all(not_rejected == True):
            print('found rejected channels')
            print(header[header_keys['name']][np.bitwise_not(not_rejected)])
        valid = np.bitwise_and(valid, not_rejected)

    ieeg_idx = np.bitwise_and(ieeg_idx, valid)

    return header[header_keys['name']][ieeg_idx]


def extract_common_names(headers):
    # takes headers for the same subject and return the name of channels (ecog-grid + seeg) that are valid and available
    # in all recordings
    # we start by first header and
    signal_names, _ = extract_names_from_header(headers[0] + '_header.mat')
    common_signals, _ = set(signal_names)
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
        # with h5py.File(data_path, 'r') as h5file:
        #     data = [h5file[obj_ref] for obj_ref in h5file['D'][0]]

    header_keys = dict([(key.lower().replace('_', '').replace('-', ''), key) for key in header.keys()])
    names = np.chararray.lower(header[header_keys['name']])
    soi_idx = np.array([False]*names.shape[-1])
    for signal_name in common_signals:
        soi_idx = np.bitwise_or(soi_idx, names == signal_name.lower())

    if 'headboxnumber' in header_keys:
        ch_hbox = header[header_keys['headboxnumber']]
    else:
        ch_hbox = np.zeros_like(soi_idx)

    return soi_idx, ch_hbox[soi_idx]


def preprocess_day(data_path, header_path, common_signals, crop_len, normalize_inputs, normalize_targets,
                   new_srate, smooth_targets):
    if common_signals is not None:
        channels_idx, ch_hbox = read_header_given_names(header_path, common_signals)
    else:
        channels_idx, ch_hbox = read_header(header_path)

    return preprocess_raw_data(data_path, channels_idx, ch_hbox, crop_len, normalize_inputs,
                               normalize_targets, smooth_targets, new_srate)


def preprocess_raw_data(data_path, ieeg_channels_idx, hbox, crop_len, normalize_inputs, normalize_targets, smooth_targets,
                        new_srate):
    try:
        data = sio.loadmat(data_path)['D']
    except NotImplementedError:
        f = h5py.File(data_path, mode='r')
        data = recursive_dict(f)['D']
        # # instad this can be handeled like this
        # with h5py.File(data_path, 'r') as h5file:
        #     data = [h5file[obj_ref] for obj_ref in h5file['D'][0]]

    ch_val = []
    targets = []
    srates= []
    # for each recording
    for recording in data:
        gameType = recording['gameType']
        if gameType.lower().find('pause') != -1 or gameType.lower().find('discrete') != -1:
            print('found game type', gameType)
            print('will be ignored!')
            continue

        srate = recording['srate'].tolist()
        # extract raw ecog-grid data
        ieeg_channels = recording['ampData'][ieeg_channels_idx].astype(np.float32)
        trial_targets = recording['tracker'].astype(np.float32)

        ieeg_channels, trial_targets = preprocessing(ieeg_channels, trial_targets, srate, hbox, normalize_inputs,
                                                     normalize_targets, smooth_targets)
        if new_srate > 0:
            ieeg_channels = resample(ieeg_channels, srate, new_srate)
            trial_targets = resample(trial_targets, srate, new_srate)
            srate = new_srate

        # remove first and last 2 seconds
        seconds_to_remove = 2
        ieeg_channels = ieeg_channels[:, seconds_to_remove * srate:-seconds_to_remove * srate]
        trial_targets = trial_targets[seconds_to_remove * srate:-seconds_to_remove * srate]
        if crop_len > 0:
            num_samples_per_crop = int(crop_len * srate)
            num_valid_crops = int((ieeg_channels.shape[-1] // num_samples_per_crop))
            for crop_idx in range(num_valid_crops-1):
                ieeg_crop = ieeg_channels[:, crop_idx*num_samples_per_crop:(crop_idx+1)*num_samples_per_crop]
                targets_crop = trial_targets[crop_idx*num_samples_per_crop:(crop_idx+1)*num_samples_per_crop]
                ch_val.append(ieeg_crop)
                targets.append(targets_crop)
                srates.append(srate)
            # last crop
            ieeg_crop = ieeg_channels[:, (crop_idx+1)*num_samples_per_crop:]
            targets_crop = trial_targets[(crop_idx+1)*num_samples_per_crop:]

            ch_val.append(ieeg_crop)
            targets.append(targets_crop)
            srates.append(srate)

        else:

            ch_val.append(ieeg_channels)
            targets.append(trial_targets)
            srates.append(srate)

    if len(ch_val) > 1:
        assert all([ch.shape[:-1] == ch_val[0].shape[:-1] for ch in ch_val[1:]])
    return ch_val, targets, srates


def preprocessing(ieeg_channels, trial_targets, srate, hbox, normalize_inputs, normalize_targets, smooth_targets):
    # apply CAR
    ieeg_channels = common_average_referencing(ieeg_channels, hbox)
    # apply high pass filtering with cut-off freq. 0.1 Hz
    ieeg_channels = highpass_filtering(ieeg_channels, 6, 0.1, srate)
    if normalize_inputs:
        std = ieeg_channels.std(axis=1, keepdims=True)
        assert np.all(std > 0), 'zero SD'
        ieeg_channels -= ieeg_channels.mean(axis=1, keepdims=True)
        ieeg_channels /= std
    # add to the list
    if normalize_targets:
        std = trial_targets.std()
        trial_targets -= trial_targets.mean()
        trial_targets /= std
    if smooth_targets:
        trial_targets = lowpass_filtering(trial_targets, 8, 10, srate)
    assert ieeg_channels.shape[-1] == trial_targets.shape[0], 'ieeg and targets have different number of samples!'
    return ieeg_channels, trial_targets


def common_average_referencing(channels, hbox):
    # computes the mean and std per headbox and then standardize the channels per hbox
    assert channels.shape[0] == len(hbox)
    unique_hbox = np.unique(hbox).tolist()

    for hb in unique_hbox:
        # use only valid signals to compute mean and SD
        hb_idx = hbox == hb
        mean = np.mean(channels[hb_idx, ], dtype=np.float32)
        channels -= mean

    return channels


def highpass_filtering(data, order, cut_feq, fs):
    # data is CxT
    z, p, k = scipy.signal.butter(order, cut_feq / (fs / 2.0), btype='highpass', output='zpk')
    assert np.all(np.abs(p) < 1), 'unstable filter'
    sos = scipy.signal.zpk2sos(z, p, k)

    return scipy.signal.sosfiltfilt(sos, data, axis=-1)


def lowpass_filtering(data, order, cut_feq, fs):
    # data is CxT
    z, p, k = scipy.signal.butter(order, cut_feq / (fs / 2.0), btype='lowpass', output='zpk')
    assert np.all(np.abs(p) < 1), 'unstable filter'
    sos = scipy.signal.zpk2sos(z, p, k)

    return scipy.signal.sosfiltfilt(sos, data, axis=-1)


def create_grouped_dataset(dataset_path, subject_paths, output_path, file_format, crop_len, normalize_inputs,
                           normalize_targets, smooth_targets, new_srate):
    if file_format == 'hdf':
        with h5py.File(output_path, 'x') as hdf:
            trial = 0
            for day in subject_paths:
                data_path = os.path.join(dataset_path, day + '_data.mat')
                header_path = os.path.join(dataset_path, day + '_header.mat')
                try:
                    trial += create_hdf_dataset(hdf, data_path, header_path, crop_len, normalize_targets, smooth_targets,
                                                new_srate, start_from_trial=trial)
                except KeyError:
                    continue

    else:
        trial = 0
        dataset_dict = dict()
        for day in subject_paths:
            data_path = os.path.join(dataset_path, day + '_data.mat')
            header_path = os.path.join(dataset_path, day + '_header.mat')
            dataset_dict, num_trials = create_mat_dataset(data_path, header_path, crop_len, normalize_inputs,
                                                          normalize_targets, smooth_targets, new_srate, dataset_dict,
                                                          start_from_trial=trial)
            trial += num_trials
        print('Writing dataset to', output_path)
        scipy.io.savemat(output_path, dataset_dict)
        print('done!')


def create_individual_datasets(dataset_path, subject_paths, output_dir, file_format, crop_len, normalize_inputs,
                               normalize_targets, smooth_targets, new_srate):
    for day in subject_paths:
        data_path = os.path.join(dataset_path, day + '_data.mat')
        header_path = os.path.join(dataset_path, day + '_header.mat')
        if file_format == 'hdf':
            ext = '.h5'
            output_path = os.path.join(output_dir, day + ext)
            with h5py.File(output_path, 'x') as hdf:
                try:
                    create_hdf_dataset(hdf, data_path, header_path, crop_len, normalize_inputs, normalize_targets,
                                       smooth_targets, new_srate, start_from_trial=0)
                except KeyError:
                    os.remove(output_path)
                    continue

        else:
            ext = '.mat'
            output_path = os.path.join(output_dir, day + ext)
            dataset_dict, trial_count = create_mat_dataset(data_path, header_path, crop_len, normalize_inputs, normalize_targets,
                                                           smooth_targets, new_srate, dataset_dict={}, start_from_trial=0)

            if trial_count > 0:
                print('Writing dataset to', output_path)
                scipy.io.savemat(output_path, dataset_dict)
                print('done!')


def create_mat_dataset(data_path, header_path, crop_len, normalize_inputs, normalize_targets, smooth_targets,
                       new_srate, dataset_dict={}, start_from_trial=0):
    print('Working on:')
    print(data_path)
    print(header_path)
    trial_count = 0
    for trial_count, (ch_val, target, srate) in \
            enumerate(zip(*preprocess_day(data_path, header_path, common_signals=None, crop_len=crop_len,
                                          normalize_inputs=normalize_inputs, normalize_targets=normalize_targets,
                                          smooth_targets=smooth_targets, new_srate=new_srate)),
                      1):
        dataset_dict['trial%d' % (start_from_trial + trial_count)] = \
            {'X': ch_val,
             'y': target,
             'srate': srate
             }

    print('%d trials were found!' % trial_count)
    if trial_count > 0:
        print('Trials\' ieeg signals shape')
        print([trial['X'].shape for trial in dataset_dict.values()])
    return dataset_dict, trial_count


def create_hdf_dataset(hdf, data_path, header_path, crop_len, normalize_inputs, normalize_targets, smooth_targets,
                       new_srate, start_from_trial=0):
    print('Working on:')
    print(data_path)
    print(header_path)
    shapes_list = []
    trial_count = 0
    for trial_count, (ch_val, target, srate) in\
            enumerate(zip(*preprocess_day(data_path, header_path, common_signals=None, crop_len=crop_len,
                                          normalize_inputs=normalize_inputs, normalize_targets=normalize_targets,
                                          smooth_targets=smooth_targets, new_srate=new_srate)),
                      1):
        # create a group
        grp = hdf.create_group('trial%d' % (start_from_trial + trial_count))
        # create the dataset
        grp.create_dataset('X', data=ch_val)
        grp.create_dataset('y', data=target)
        # add the sampling rate to the attributes
        grp.attrs['srate'] = srate
        shapes_list.append(ch_val.shape)

    print('%d trials were found!' % trial_count)
    if trial_count > 0:
        print('Trials\' ieeg signals shape')
        print(shapes_list)

    return trial_count


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True)) # , help='path to the directory of the raw recording files')
@click.argument('output_dir', type=click.Path(exists=True))   # , help='path to the directory of the raw recording files')
@click.option('--group', '-g', is_flag=True, help='recordings of the same subject will be grouped and only signals '
                                                  'that appears in all recordings will be selected')
@click.option('--smooth_targets', is_flag=True, help='smooth targets with 8\'th order Butterworth filter with 10-Hz '
                                                     'cut-off frequency and zero-phase shift')
@click.option('--normalize_inputs', is_flag=True, help='inputs will normalized to their SD')
@click.option('--normalize_targets', is_flag=True, help='targets will normalized to their SD')
@click.option('--output_format', default='mat', help='in what format should the output files be stored. MAT and HDF '
                                                     'formats are supported')
@click.option('--subject', '-s', default='all', help='pre-process recordings of subject. if not provided, all subjects'
                                                     'appear in the dataset path will be pre-processed')
@click.option('--crop_len', '-c', default=0, help='crop length. if not provided trials will not be cropped.')
@click.option('--new_srate', default=0, help='if not 0 (defalut) the signals will be downsampled to the passed number.')
def main(dataset_path, subject, output_dir, output_format, group, crop_len, normalize_inputs, normalize_targets,
         smooth_targets, new_srate):
    output_format = output_format.lower()
    assert output_format == 'mat' or 'hdf', 'only MAT and HDF file formats are support'
    all_subject_pathes = read_dataset_dir(dataset_path)
    if subject.lower() != 'all':
        assert subject.lower() in [sub_id.lower() for sub_id in all_subject_pathes.keys()], 'subject not found!'
    for subject_id, subject_pathes in all_subject_pathes.items():
        if subject.lower() != 'all' and subject.lower() != subject_id.lower():
            continue
        print_msg = 'Creating dataset for subject %s:' % subject_id
        print(print_msg)
        print('='*len(print_msg))
        try:
            if group:
                if output_format == 'hdf':
                    ext = '.h5'
                else:
                    ext = '.mat'
                output_path = os.path.join(output_dir, subject_id + ext)
                create_grouped_dataset(dataset_path, subject_pathes, output_path, output_format, crop_len,
                                       normalize_inputs, normalize_targets, smooth_targets, new_srate)
            else:
                create_individual_datasets(dataset_path, subject_pathes, output_dir, output_format, crop_len,
                                           normalize_inputs, normalize_targets, smooth_targets, new_srate)
        except KeyError:
            continue


if __name__ == '__main__':
    main()
