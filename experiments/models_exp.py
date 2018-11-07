import torch
import os
import yaml
import json
import click
import logging
os.sys.path.insert(0, '..')
from utils.experiment_util import *
from tensorboardX import SummaryWriter
import h5py
from glob import glob
from sklearn.model_selection import KFold, LeaveOneOut
import pandas as pd
from utils.config import cfg, merge_configs

CUDA = True
EXPERIMENT_NAME = 'models'
np.random.seed(cfg.TRAINING.RANDOM_SEED)
torch.manual_seed(cfg.TRAINING.RANDOM_SEED)
TASK_NAMES = ['XPOS', 'XVEL', 'ABSPOS', 'ABSVEL']

logger = logging.getLogger(__name__)


@click.command(name='models-experiments')
@click.argument('configs', type=click.Path(), default=os.path.curdir)
@click.argument('mode', type=click.Choice(['train', 'eval', 'cv']), default='train')
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('subject', type=str)
@click.option('--log_dir', '-l', type=click.Path(), default=os.path.curdir)
@click.option('--n_splits', default=5, help='Number of cross-validation splits')
@click.option('--task', '-t', type=click.Choice(['xpos', 'xvel', 'abspos', 'absvel', 'multi']), default='xpos',
              help='Task to decode. acceptable are:\n'
                   '* xpos for position decoding.\n'
                   '* xvel for velocity decoding.\n'
                   '* abspos for absolute position decoding.\n'
                   '* absvel for absolute velocity decoding.\n'
                   '* multi for multi-task decoding.\n'
                   'default is pos')
def main(mode, configs, dataset_dir, subject, log_dir, n_splits, task):
    with open(configs, 'r') as f:
        merge_configs(yaml.load(f))

    if mode == 'eval':
        train_path = os.path.join(log_dir, task.upper(), 'TRAIN', subject, cfg.TRAINING.MODEL.upper())
        assert os.path.exists(train_path), f"Can't detect training folder: {train_path}"

    log_dir = os.path.join(log_dir, task.upper(), mode.upper(), subject, cfg.TRAINING.MODEL.upper())
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(log_dir, 'log.txt'), filemode='w+',
                        format='%(levelname)s %(filename)s:%(lineno)4d: %(message)s')
    logger.info('Called with configs:')
    logger.info(json.dumps(cfg, indent=2))
    datasets = []
    if task == 'xpos':
        datasets = glob(dataset_dir + '*' + subject+'_*_xpos.mat')
    elif task == 'xvel':
        datasets = glob(dataset_dir + '*' + subject+'_*_xvel.mat')
    elif task == 'abspos':
        datasets = glob(dataset_dir + '*' + subject+'_*_absPos.mat')
    elif task == 'absvel':
        datasets = glob(dataset_dir + '*' + subject+'_*_absVel.mat')
    elif task == 'multi':
        pos_datasets = glob(dataset_dir + '*' + subject+'_*_xpos.mat')
        for pos_dataset in pos_datasets:
            recording_day = pos_dataset.rstrip('_xpos.mat')
            vel_dataset = recording_day + '_xvel.mat'
            assert os.path.exists(vel_dataset), 'could not find enough files for multi-task decoding'
            datasets.append((pos_dataset, vel_dataset))
    else:
        raise KeyError
    assert len(datasets) > 0, 'no datasets for subject %s found!' % subject
    rec_names = []
    if task == 'multi':
        for dataset_path in [dataset[0] for dataset in datasets]:
            rec_day_name = os.path.basename(dataset_path).split('.')[0].split('_')
            rec_names.append('_'.join([rec_day_name[1], rec_day_name[3]]))
    else:
        for dataset_path in datasets:
            rec_day_name = os.path.basename(dataset_path).split('.')[0].split('_')
            rec_names.append('_'.join([rec_day_name[1], rec_day_name[3]]))

    if mode == 'cv':
        index = pd.MultiIndex.from_product([rec_names, ['fold%d' % fold for fold in range(1, n_splits+1)]],
                                           names=['day', 'fold'])
    else:
        index = pd.Index(rec_names, names='day')

    if task == 'multi':
        columns = TASK_NAMES + ['mse']
    else:
        columns = ['corr', 'mse']

    df = pd.DataFrame(index=index, columns=columns)
    for dataset_path, rec_name in zip(datasets, rec_names):
        msg = str('Working on dataset %s:' % rec_name if task == 'multi' else dataset_path)
        logger.info(msg + '\n' + '=' * len(msg))
        if mode == 'cv' or mode == 'train':
            dataset_name = 'D'
        else:
            dataset_name = 'F'
        if task == 'multi':
            trials, in_channels = read_multi_datasets(dataset_path, dataset_name)
            num_classes = len(dataset_path)
        else:
            trials, in_channels = read_dataset(dataset_path, dataset_name)
            num_classes = 1
        logger.info(f'{len(trials)} trials found')
        logger.info(f'Number of input input channels: {in_channels}')

        if mode == 'cv':
            crop_idx = list(range((len(trials))))
            if n_splits > 0:
                kfold = KFold(n_splits=n_splits, shuffle=False, random_state=cfg.TRAINING.RANDOM_SEED)
            elif n_splits == -1:
                kfold = LeaveOneOut()
            else:
                raise ValueError(f'Invalid number of splits: {n_splits}')

            for fold_idx, (train_split, valid_split) in enumerate(kfold.split(crop_idx), 1):

                model, optimizer, scheduler, loss_fun, metric = create_model(in_channels, num_classes, CUDA)

                msg = str(f'FOLD{fold_idx}:')
                logger.info(msg)
                logger.info('='*len(msg))
                logger.info('Training trials:')
                logger.info(train_split)
                logger.info('Validation trials:')
                logger.info(valid_split)

                training_writer = SummaryWriter(os.path.join(log_dir, rec_name, 'fold' + str(fold_idx), 'train'))
                valid_writer = SummaryWriter(os.path.join(log_dir, rec_name, 'fold' + str(fold_idx), 'valid'))
                weights_path = os.path.join(log_dir, rec_name, 'fold' + str(fold_idx), 'weights.pt')

                training_trials = [trials[idx] for idx in train_split]
                valid_trials = [trials[idx] for idx in valid_split]
                corr, mse = training_loop(model, optimizer, scheduler, loss_fun, metric, training_trials,
                                          training_writer, valid_trials, valid_writer, weights_path, cuda=CUDA)
                if task == 'multi':
                    for task_idx in range(len(corr)):
                        df.loc[(rec_name, 'fold' + str(fold_idx)), TASK_NAMES[task_idx]] = \
                            corr['Class%d' % task_idx]
                    df.loc[(rec_name, 'fold' + str(fold_idx)), 'mse'] = mse
                else:
                    df.loc[(rec_name, 'fold' + str(fold_idx)), :] = [corr, mse]
                # writes every time just in case it couldn't run the complete script
                df.to_csv(os.path.join(log_dir, 'cv_acc.csv'), index=True)

        elif mode == 'train':
            model, optimizer, scheduler, loss_fun, metric = create_model(in_channels, num_classes, CUDA)

            weights_path = os.path.join(log_dir, rec_name, 'weights.pt')

            training_writer = SummaryWriter(os.path.join(log_dir, rec_name, 'train'))

            corr, mse = training_loop(model, optimizer, scheduler, loss_fun, metric, trials, training_writer,
                                      weights_path=weights_path, cuda=CUDA)

            if task == 'multi':
                for task_idx in range(len(corr)):
                    df.loc[rec_name, TASK_NAMES[task_idx]] = \
                        corr['Class%d' % task_idx]
                df.loc[rec_name, 'mse'] = mse
            else:
                df.loc[rec_name, :] = [corr, mse]

            df.to_csv(os.path.join(log_dir, 'cv_acc.csv'), index=True)

        # eval
        else:
            model, optimizer, scheduler, loss_fun, metric = create_model(in_channels, num_classes, CUDA)

            weights_path = os.path.join(train_path, rec_name, 'weights.pt')
            assert os.path.exists(weights_path), 'No weights are detected for this recording!'
            corr, mse = eval_model(model, loss_fun, metric, trials, weights_path, cuda=CUDA)

            if task == 'multi':
                for task_idx in range(len(corr)):
                    df.loc[rec_name, TASK_NAMES[task_idx]] = \
                        corr['Class%d' % task_idx]
                df.loc[rec_name, 'mse'] = mse
            else:
                df.loc[rec_name, :] = [corr, mse]
            # writes every time just in case it couldn't run the complete script
            df.to_csv(os.path.join(log_dir, 'cv_acc.csv'), index=True)


if __name__ == '__main__':
    torch.multiprocessing.set_sharing_strategy('file_system')
    main()


