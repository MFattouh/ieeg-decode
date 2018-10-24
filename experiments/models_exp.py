# import torch.multiprocessing as mp
import os

from scipy.io.arff.tests.test_arffread import expected_types

os.sys.path.insert(0, '..')
from utils.experiment_util import *
from tensorboardX import SummaryWriter
from torch import optim
import yaml
import logging
import json
import h5py
from glob import glob
from sklearn.model_selection import KFold
import click
import pandas as pd
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.modules import Expression
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet as Shallow
from torch.nn.functional import mse_loss
from utils.config import cfg, merge_configs
import yaml

MAX_EPOCHS = 1000
EVAL_TRAIN_EVERY = 50
EVAL_VALID_EVERY = 50
CUDA = True
EXPERIMENT_NAME = 'models'
np.random.seed(cfg.TRAINING.RANDOM_SEED)
torch.manual_seed(cfg.TRAINING.RANDOM_SEED)
TASK_NAMES = ['POS', 'VEL']

logger = logging.getLogger(__name__)


@click.command(name='models-experiments')
@click.argument('exp_type', type=click.Choice(['train', 'eval', 'cv']), default='train')
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('subject', type=str)
@click.option('--log_dir', '-l', type=click.Path(), default=os.path.curdir)
@click.option('--configs', '-c', type=click.Path(), default=os.path.curdir)
@click.option('--model_type', '-m', type=click.Choice(['rnn', 'deep4', 'shallow', 'hybrid']), default='rnn')
@click.option('--n_splits', default=5, help='Number of cross-validation splits')
@click.option('--task', '-t', type=click.Choice(['xpos', 'xvel', 'abspos', 'absvel', 'multi']), default='xpos',
              help='Task to decode. acceptable are:\n'
                   '* xpos for position decoding.\n'
                   '* xvel for velocity decoding.\n'
                   '* abspos for absolute position decoding.\n'
                   '* absvel for absolute velocity decoding.\n'
                   '* multi for multi-task decoding.\n'
                   'default is pos')
def main(exp_type, dataset_dir, subject, model_type, log_dir, n_splits, task, configs):
    if configs is not None:
        with open(configs, 'r') as f:
            merge_configs(yaml.load(f))

    if exp_type == 'eval':
        train_path = os.path.join(log_dir, task.upper(), 'TRAIN', subject, model_type.upper())
        assert os.path.exists(train_path), f"Can't detect training folder: {train_path}"

    log_dir = os.path.join(log_dir, task.upper(), exp_type.upper(), subject, model_type.upper())
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    logging.basicConfig(level=logging.DEBUG, filename=os.path.join(log_dir, 'log.o'), filemode='w+',
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
    new_srate_x = cfg.TRAINING.INPUT_SAMPLING_RATE
    new_srate_y = cfg.TRAINING.OUTPUT_SAMPLING_RATE
    x2y_ratio = new_srate_x / new_srate_y
    batch_size = cfg.TRAINING.BATCH_SIZE

    # window size
    crop_len = cfg.TRAINING.CROP_LEN  # [sec]
    num_relaxed_samples = 681  # int(relax_window * new_srate_x)

    # stride = crop_len - num_relaxed_samples
    stride = cfg.TRAINING.INPUT_STRIDE

    # define some constants related to model type
    if model_type == 'rnn':
        learning_rate = cfg.OPTIMIZATION.BASE_LR
        wd_const = cfg.OPTIMIZATION.WEIGHT_DECAY
        dummy_idx = 'f'
        weights = make_weights(crop_len, num_relaxed_samples, type='step')
        weights_tensor = torch.from_numpy(weights)
        if CUDA:
            weights_tensor = weights_tensor.cuda()
    elif model_type == 'deep4':
        learning_rate = 1e-4    # value from robin's script
        wd_const = 0
        dummy_idx = 'l'
    elif model_type == 'shallow':
        wd_const = 0
        dummy_idx = 'l'
        learning_rate = 1e-4    # value from robin's script
        num_dropped_samples = 113
        weights = make_weights(crop_len - num_dropped_samples, num_relaxed_samples - num_dropped_samples, type='step')
        weights_tensor = torch.from_numpy(weights)
        if CUDA:
            weights_tensor = weights_tensor.cuda()
    elif model_type == 'hybrid':
        learning_rate = 5e-3    # value from paper
        wd_const = 5e-6         # value from paper
        dummy_idx = 'f'
        num_dropped_samples = 121
        weights = make_weights(crop_len - num_dropped_samples, num_relaxed_samples - num_dropped_samples, type='step')
        weights_tensor = torch.from_numpy(weights)
        if CUDA:
            weights_tensor = weights_tensor.cuda()

    else:
        raise NotImplementedError

    rec_names = []
    if task == 'multi':
        for dataset_path in [dataset[0] for dataset in datasets]:
            rec_day_name = os.path.basename(dataset_path).split('.')[0].split('_')
            rec_names.append('_'.join([rec_day_name[1], rec_day_name[3]]))
    else:
        for dataset_path in datasets:
            rec_day_name = os.path.basename(dataset_path).split('.')[0].split('_')
            rec_names.append('_'.join([rec_day_name[1], rec_day_name[3]]))

    if exp_type == 'cv':
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
        logger.info(msg + '\n' + '=' * len(msg) + '\n' + '=' * len(msg))
        if exp_type == 'cv' or exp_type == 'train':
            dataset_name = 'D'
        else:
            dataset_name = 'F'
        if task == 'multi':
            crops, in_channels = read_multi_datasets(dataset_path, dataset_name, crop_len, stride, x2y_ratio, dummy_idx)
            num_classes = len(dataset_path)
        else:
            crops, in_channels = read_dataset(dataset_path, dataset_name, crop_len, stride, x2y_ratio, dummy_idx)
            num_classes = 1
        logger.info(f'{len(crops)} trials found!')
        # create the model
        if model_type == 'rnn':
            model = HybridModel(in_channels=in_channels, output_stride=int(cfg.HYBRID.OUTPUT_STRIDE))

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd_const)
            loss_fun = WeightedMSE(weights_tensor)
            metric = CorrCoeff(weights).weighted_corrcoef

        elif model_type == 'deep4':
            model = Deep4Net(in_chans=in_channels, n_classes=num_classes, input_time_length=crop_len,
                             final_conv_length=2, stride_before_pool=True).create_network()

            # remove softmax
            new_model = nn.Sequential()
            for name, module in model.named_children():
                if name == 'softmax':
                    # continue
                    break
                new_model.add_module(name, module)

            # lets remove empty final dimension
            def squeeze_out(x):
                assert x.size()[1] == num_classes and x.size()[3] == 1
                return x.squeeze()
                # assert x.size()[1] == 1 and x.size()[3] == 1
                # return x[:, 0, :, 0]

            new_model.add_module('squeeze', Expression(squeeze_out))
            if num_classes > 1:
                def transpose_class_time(x):
                    return x.transpose(2, 1)
                new_model.add_module('trans', Expression(transpose_class_time))

            model = new_model

            to_dense_prediction_model(model)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd_const)
            loss_fun = mse_loss
            metric = CorrCoeff().corrcoeff

        elif model_type == 'shallow':
            model = Shallow(in_chans=in_channels, n_classes=num_classes, input_time_length=crop_len,
                            final_conv_length=2).create_network()

            # remove softmax
            new_model = nn.Sequential()
            for name, module in model.named_children():
                if name == 'softmax':
                    break
                new_model.add_module(name, module)

            # lets remove empty final dimension
            def squeeze_out(x):
                assert x.size()[1] == num_classes and x.size()[3] == 1
                return x.squeeze()
                # return x[:, 0, :, 0]

            new_model.add_module('squeeze', Expression(squeeze_out))
            model = new_model

            to_dense_prediction_model(model)

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd_const)
            loss_fun = WeightedMSE(weights_tensor)
            metric = CorrCoeff(weights).weighted_corrcoef
        elif model_type == 'hybrid':
            cfg.HYBRID.SPATIAL_CONVS['num_filters'] = [in_channels]
            model = HybridModel(in_channels=in_channels, output_stride=int(cfg.HYBRID.OUTPUT_STRIDE))

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd_const)
            loss_fun = WeightedMSE(weights_tensor)
            metric = CorrCoeff(weights).weighted_corrcoef

        elif model_type == 'tcn':
            raise NotImplementedError

        if CUDA:
            model.cuda()

        if exp_type == 'cv':
            crop_idx = np.arange(len(crops)).squeeze().tolist()
            kfold = KFold(n_splits=n_splits, shuffle=False, random_state=cfg.TRAINING.RANDOM_SEED)

            for fold_idx, (train_split, valid_split) in enumerate(kfold.split(crop_idx), 1):
                training_loader, valid_loader = create_loaders(crops, train_split, valid_split, batch_size)
                msg = str(f'FOLD{fold_idx}:')
                logger.info(msg)
                logger.info('='*len(msg))
                logger.info('Training trials:')
                logger.info(train_split)
                logger.info('Validation trials:')
                logger.info(valid_split)

                # scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
                training_writer = SummaryWriter(os.path.join(log_dir, rec_name, 'fold' + str(fold_idx), 'train'))
                valid_writer = SummaryWriter(os.path.join(log_dir, rec_name, 'fold' + str(fold_idx), 'valid'))
                # training_writer.add_text('Model parameters', str(HybridModel.get_meta(model)))
                training_writer.add_text('Description', model_type.upper())
                training_writer.add_text('Learning Rate', str(learning_rate))
                training_writer.add_text('Weight Decay', str(wd_const))
                training_writer.add_text('Crop Length[sec]', str(crop_len))
                training_writer.add_text('Input srate[Hz]', str(new_srate_x))
                training_writer.add_text('Output srate[Hz]', str(new_srate_y))
                training_writer.add_text('relaxed samples', str(num_relaxed_samples))
                training_writer.add_text('Input channels', str(in_channels))

                weights_path = os.path.join(log_dir, rec_name, 'fold' + str(fold_idx), 'weights.pt')

                corr, mse = run_experiment(model, optimizer, loss_fun, metric, training_loader, training_writer,
                                           valid_loader, valid_writer, weights_path, max_epochs=MAX_EPOCHS,
                                           eval_train_every=EVAL_TRAIN_EVERY, eval_valid_every=EVAL_VALID_EVERY,
                                           cuda=CUDA)
                if task == 'multi':
                    for task_idx in range(len(corr)):
                        df.loc[(rec_name, 'fold' + str(fold_idx)), TASK_NAMES[task_idx]] = \
                            corr['Class%d' % task_idx]
                    df.loc[(rec_name, 'fold' + str(fold_idx)), 'mse'] = mse
                else:
                    df.loc[(rec_name, 'fold' + str(fold_idx)), :] = [corr, mse]
                # writes every time just in case it couldn't run the complete script
                df.to_csv(os.path.join(log_dir, 'cv_acc.csv'), index=True)

        elif exp_type == 'train':
            num_crops = len(crops)
            train_split = list(np.arange(0, num_crops - 2))
            valid_split = list(np.arange(num_crops - 2, num_crops))

            training_loader, valid_loader = create_loaders(crops, train_split, valid_split, batch_size)

            # print(num_classes)
            # print(in_channels)
            # print(len(crops))
            # print(len(training_loader))
            # print(len(valid_loader))

            weights_path = os.path.join(log_dir, rec_name, 'weights.pt')

            training_dataset = ConcatDataset(crops)
            training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
            # scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
            training_writer = SummaryWriter(os.path.join(log_dir, rec_name, 'train'))
            # training_writer.add_text('Model parameters', str(HybridModel.get_meta(model)))
            training_writer.add_text('Description', model_type.upper())
            training_writer.add_text('Learning Rate', str(learning_rate))
            training_writer.add_text('Weight Decay', str(wd_const))
            training_writer.add_text('Crop Length', str(crop_len))
            training_writer.add_text('Input srate[Hz]', str(new_srate_x))
            training_writer.add_text('Output srate[Hz]', str(new_srate_y))
            training_writer.add_text('relaxed samples', str(num_relaxed_samples))
            training_writer.add_text('Input channels', str(in_channels))

            valid_writer = SummaryWriter(os.path.join(log_dir, rec_name, 'valid'))
            valid_writer.add_text('Description', model_type.upper())
            valid_writer.add_text('Learning Rate', str(learning_rate))
            valid_writer.add_text('Weight Decay', str(wd_const))
            valid_writer.add_text('Crop Length', str(crop_len))
            valid_writer.add_text('Input srate[Hz]', str(new_srate_x))
            valid_writer.add_text('Output srate[Hz]', str(new_srate_y))
            valid_writer.add_text('relaxed samples', str(num_relaxed_samples))
            valid_writer.add_text('Input channels', str(in_channels))

            corr, mse = run_experiment(model, optimizer, loss_fun, metric, training_loader, training_writer,
                                       valid_loader, valid_writer, weights_path, max_epochs=MAX_EPOCHS,
                                       eval_train_every=EVAL_TRAIN_EVERY, eval_valid_every=EVAL_VALID_EVERY,
                                       cuda=CUDA)

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
            weights_path = os.path.join(train_path, rec_name, 'weights.pt')
            assert os.path.exists(weights_path), 'No weights are detected for this recording!'
            valid_dataset = ConcatDataset(crops)
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

            corr, mse = run_eval(model, loss_fun, metric, valid_loader, weights_path, cuda=CUDA)

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
    # mp = mp.get_context('spawn')
    main()


