import click
import pandas as pd
import os
from glob import glob
import h5py
from mat73_to_pickle import recursive_dict


def read_dataset_dir(dataset_path):
    # takes the path to the dataset and returns a dict. of lists, where for each subject a list contains the file names
    # (without extention) of all recordings is added to the dictionary.
    files = glob(dataset_path + '/*_xpos.mat')

    subject_pathes = dict()
    for file_path in files:
        # check if a data file also exists
        file_name = os.path.basename(file_path)
        if len(file_name.split('_')) > 1:
            subject_id = file_name.split('_')[-3]
            if subject_id in subject_pathes:
                subject_pathes[subject_id].append(file_name)
            else:
                subject_pathes[subject_id] = [file_name]

    return subject_pathes


def condence_jiri(dataset_dir):
    df_list = []
    mat_pathes = glob(os.path.join(dataset_dir, '*/*.mat'))
    for mat_path in mat_pathes:
        with h5py.File(mat_path) as fp:
            data = fp['CC_folds'][:]
        df = pd.DataFrame(data, columns=['corr'])
        df['rec'] = os.path.basename(os.path.dirname(mat_path))
        df['model'] = 'DEEP4'
        df['fold'] = ['fold%d' % id for id in range(data.shape[0])]
        df_list.append(df)
    results = pd.concat(df_list)
    results.reset_index(drop=True, inplace=True)
    return results
    # results.to_csv(os.path.join(dataset_dir, 'results.csv'), index=False)


def condence_results(dataset_dir):
    df_list = []
    csv_pathes = glob(os.path.join(dataset_dir, '*/*/*.csv'))
    experiments = [os.path.basename(os.path.dirname(path)) for path in csv_pathes]
    for experiment, csv_path in zip(experiments, csv_pathes):
        df = pd.read_csv(csv_path)
        if 'day' not in df.columns.values:
            df.rename(columns={'Unnamed: 0': 'day'}, inplace=True)
        subject = os.path.basename(os.path.dirname(os.path.dirname(csv_path)))
        df['rec'] = df['day'].apply(lambda day: '_'.join([day.split('_')[0], subject,  day.split('_')[1]]))
        df['model'] = os.path.basename(os.path.dirname(csv_path))
        df_list.append(df)
    results = pd.concat(df_list)
    results.reset_index(drop=True, inplace=True)
    results.drop('day', axis=1, inplace=True)
    results.drop('mse', axis=1, inplace=True)
    # results.to_csv(os.path.join(dataset_dir, 'results.csv'), index=False)
    return results
    # print(results.head())


@click.command()
@click.argument('dataset_path', type=click.Path(exists=True))
def subject_stats(dataset_path):
    subjects = read_dataset_dir(dataset_path)
    df = pd.DataFrame(columns=['recording', 'input_channels', 'num_training_trials', 'training_len[min]', 'num_valid_trials', 'valid_len[min]',
                               'srate'])

    for recordings in subjects.values():
        for recording in recordings:
            with h5py.File(os.path.join(dataset_path, recording)) as hf:
                num_training_trials = len(hf['D'][0])
                channels = recursive_dict(hf['H/channels'])
                esm = channels['esm']
                num_hand_arm_channels = len((list(filter(lambda e: ('arm' in e or 'hand' in e) and 'motor' in e, esm))))
                srate = hf[hf['D'][0][0]]['srate'][:][0][0]
                in_channels = hf[hf['D'][0][0]]['ieeg'].shape[0]
                training_len = 0
                for training_trial in range(num_training_trials):
                    training_len += hf[hf['D'][0][training_trial]]['ieeg'].shape[-1] / srate / 60

                num_valid_trials = len(hf['F'][0])
                valid_len = 0
                for valid_trial in range(num_valid_trials):
                    valid_len += hf[hf['F'][0][valid_trial]]['ieeg'].shape[-1] / srate / 60

                df = df.append({'recording': '_'.join(recording.split('_')[1:-1]),
                                'input_channels': in_channels,
                                'num_training_trials': num_training_trials,
                                'training_len[min]': training_len,
                                'num_valid_trials': num_valid_trials,
                                'valid_len[min]': valid_len,
                                'num_arm_hand': num_hand_arm_channels,
                                'srate': srate},
                               ignore_index=True)

    df.to_csv('subject_stats.csv', index=False, )


if __name__ == '__main__':
    subject_stats()
