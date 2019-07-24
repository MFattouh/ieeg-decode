import torch.multiprocessing as mp
import os
os.sys.path.insert(0, '..')
from utils.experiment_util import *
from torch import optim
import h5py
import yaml
from glob import glob
from sklearn.model_selection import KFold, LeaveOneOut, train_test_split
import click
import pandas as pd
from utils.config import cfg, merge_configs
import random
import json
from packaging import version

if version.parse(torch.__version__) < version.parse("1.1.0"):
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

CUDA = True
TASK_NAMES = ['XPOS', 'XVEL', 'ABSPOS', 'ABSVEL']

logger = logging.getLogger(__name__)

@click.command()
@click.argument('configs', type=click.Path())
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('subject', type=str)
@click.argument('log_dir', type=click.Path())
@click.argument('num_layers', type=int)
@click.option('--n_splits', default=5, help='Number of cross-validation splits')
@click.option('--task', '-t', type=click.Choice(['xpos', 'xvel', 'abspos', 'absvel', 'xacc', 'absacc']), default='xpos',
              help='Task to decode. acceptable are:\n'
                   '* xpos for position decoding.\n'
                   '* xvel for velocity decoding.\n'
                   '* abspos for absolute position decoding.\n'
                   '* absvel for absolute velocity decoding.\n'
                   '* xacc for acceleration decoding.\n'
                   '* absacc for absolute acceleration decoding.\n'
                   'default is pos')
def main(configs, dataset_dir, subject, log_dir, num_layers, n_splits, task):
    with open(configs, 'r') as f:
        merge_configs(yaml.load(f))
        assert cfg.TRAINING.MODEL == 'RNN'
        assert cfg.HYBRID.RNN.NUM_LAYERS == num_layers

    # set the random state
    np.random.seed(cfg.TRAINING.RANDOM_SEED)
    torch.manual_seed(cfg.TRAINING.RANDOM_SEED)
    random.seed(cfg.TRAINING.RANDOM_SEED)

    if num_layers == 1:
        log_dir = os.path.join(log_dir, task.upper(), subject, '1Layer')
    else:
        log_dir = os.path.join(log_dir, task.upper(), subject, str(num_layers) + 'Layers')
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
    elif task == 'xacc':
        datasets = glob(dataset_dir + 'ALL_*' + subject+'_*_xacc.mat')
    elif task == 'absacc':
        datasets = glob(dataset_dir + '*' + subject+'_*_absAcc.mat')
    else:
        raise KeyError
    assert len(datasets) > 0, 'no datasets for subject %s found!' % subject

    # data frame to hold cv cross. corr.
    rec_names = []
    for dataset_path in datasets:
        rec_day_name = os.path.basename(dataset_path).split('.')[0].split('_')
        rec_names.append('_'.join([rec_day_name[1], rec_day_name[3]]))

    index = pd.MultiIndex.from_product([rec_names, ['fold%d' % fold for fold in range(1,n_splits+1)]],
                                       names=['day', 'fold'])
    df = pd.DataFrame(index=index, columns=['corr', 'mse'])
    for dataset_path, rec_name in zip(datasets, rec_names):
        msg = str('Working on dataset %s:' % rec_name if task == 'multi' else dataset_path)
        logger.info(msg + '\n' + '=' * len(msg))
        dataset_name = 'D'
        if task == 'multi':
            trials, in_channels = read_multi_datasets(dataset_path, dataset_name, mha_only=cfg.TRAINING.MHA_CHANNELS_ONLY)
            num_classes = len(dataset_path)
        else:
            trials, in_channels = read_dataset(dataset_path, dataset_name, mha_only=cfg.TRAINING.MHA_CHANNELS_ONLY)
            num_classes = 1
        logger.info(f'{len(trials)} trials found')
        logger.info(f'Number of input input channels: {in_channels}')
        if in_channels < 1:
            logger.warning(f'Zero valid channels found!!!!!!')
            print(f'Zero valid channels found!!!!!!')
            return

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

    print("Done!")


if __name__ == '__main__':
    # mp = mp.get_context('spawn')
    main()


