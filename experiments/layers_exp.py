import torch.multiprocessing as mp
import os
os.sys.path.insert(0, '..')
from utils.experiment_util import *
from tensorboardX import SummaryWriter
from torch import optim
import h5py
from glob import glob
from sklearn.model_selection import KFold
import click
import pandas as pd

MAX_EPOCHS = 1000
EVAL_EVERY = 50
CUDA = True
EXPERIMENT_NAME = 'layers'
RANDOM_SEED = 10418
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


@click.command()
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('subject', type=str)
@click.argument('log_dir', type=click.Path())
@click.argument('num_layers', type=int)
@click.option('--n_splits', default=5, help='Number of cross-validation splits')
def main(dataset_dir, subject, log_dir, num_layers, n_splits):
    log_dir = os.path.join(log_dir, EXPERIMENT_NAME, str(num_layers)+'Layers', subject)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    datasets = glob(dataset_dir + '*'+subject+'_*_xpos.mat')
    assert len(datasets) > 0, 'no datasets for subject %s found!' % subject
    new_srate_x = 250
    new_srate_y = 250
    x2y_ratio = new_srate_x / new_srate_y
    batch_size = 32
    hidden_size = 64
    channel_config = None
    time_config = None
    weights_dropout = []
    if num_layers > 1:
        weights_dropout.extend(['weight_ih_l%d' % layer for layer in range(num_layers)])
        weights_dropout.extend(['weight_hh_l%d' % layer for layer in range(num_layers)])
    rnn_config = {'rnn_type': 'gru',
                  'hidden_size': hidden_size,
                  'num_layers': num_layers,
                  'dropout': 0.3,
                  'weights_dropout': weights_dropout,
                  'batch_norm': False}
    fc_config = {'num_classes': 1,
                 'fc_size': [32, 10],
                 'batch_norm': [False, False],
                 'dropout': [0.5, .3],
                 'activations': [nn.Hardtanh(-1, 1, inplace=True)] * 2}

    # window size
    crop_len = 16
    num_relaxed_samples = 681  # int(relax_window * new_srate_x)
    learning_rate = 5e-3
    wd_const = 5e-6
    stride = crop_len * new_srate_x - num_relaxed_samples
    weights = make_weights(crop_len * new_srate_x, num_relaxed_samples, type='step')
    weights_tensor = torch.from_numpy(weights)
    if CUDA:
        weights_tensor = weights_tensor.cuda()

    # data frame to hold cv cross. corr.
    rec_names = []
    for dataset_path in datasets:
        rec_day_name = os.path.basename(dataset_path).split('.')[0].split('_')
        rec_names.append('_'.join([rec_day_name[1], rec_day_name[3]]))

    index = pd.MultiIndex.from_product([rec_names, ['fold%d' % fold for fold in range(1,n_splits+1)]],
                                       names=['day', 'fold'])
    df = pd.DataFrame(index=index, columns=['corr', 'mse'])
    for dataset_path, rec_name in zip(datasets, rec_names):
        msg = str(datetime.now()) + ': Start working on dataset %s:' % dataset_path
        print(msg)
        print('='*len(msg))
        print('='*len(msg))
        crops, in_channels = read_dataset(dataset_path, crop_len*new_srate_x, stride)
        print(len(crops), 'trials found!')
        crop_idx = np.arange(len(crops)).squeeze().tolist()
        kfold = KFold(n_splits=n_splits, shuffle=False, random_state=RANDOM_SEED)
        for cross_valid_idx, (train_split, valid_split) in enumerate(kfold.split(crop_idx), 1):
            training_loader, valid_loader = create_loader(crops, train_split, valid_split, batch_size)
            msg = str(datetime.now()) + ': FOLD%d:' % cross_valid_idx
            print(msg)
            print('='*len(msg))
            print('Training trials:')
            print(train_split)
            print('Validation trials:')
            print(valid_split)
            model = HybridModel(in_channels=in_channels, channel_conv_config=channel_config,
                                time_conv_config=time_config, rnn_config=rnn_config,
                                fc_config=fc_config, output_stride=int(x2y_ratio))

            if CUDA:
                model.cuda()

            optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd_const)
            loss_fun = WeightedMSE(weights_tensor)
            metric = CorrCoeff(weights).weighted_corrcoef

            # scheduler = StepLR(optimizer, step_size=100, gamma=0.9)
            training_writer = SummaryWriter(os.path.join(log_dir, rec_name, 'fold' + str(cross_valid_idx), 'train'))
            valid_writer = SummaryWriter(os.path.join(log_dir, rec_name, 'fold' + str(cross_valid_idx), 'valid'))
            # training_writer.add_text('Model parameters', str(HybridModel.get_meta(model)))
            training_writer.add_text('Description', 'RNN with %s layer(s)' % num_layers)
            training_writer.add_text('Learning Rate', str(learning_rate))
            training_writer.add_text('Weight Decay', str(wd_const))
            training_writer.add_text('Crop Length[sec]', str(crop_len))
            training_writer.add_text('Input srate[Hz]', str(new_srate_x))
            training_writer.add_text('Output srate[Hz]', str(new_srate_y))
            training_writer.add_text('relaxed samples', str(num_relaxed_samples))
            training_writer.add_text('Input channels', str(in_channels))

            weights_path = os.path.join(log_dir, rec_name, 'fold' + str(cross_valid_idx), 'weights.pt')

            fold_corr, fold_mse = run_experiment(model, optimizer, loss_fun, metric, training_loader, training_writer,
                                                 valid_loader, valid_writer, weights_path, max_epochs=MAX_EPOCHS,
                                                 eval_train_every=EVAL_EVERY, eval_valid_every=EVAL_EVERY, cuda=CUDA)

            df.loc[(rec_name, 'fold' + str(cross_valid_idx)), :] = [fold_corr, fold_mse]
            # writes every time just in case it couldn't run the complete script
            df.to_csv(os.path.join(log_dir, 'cv_acc.csv'), index=True)


if __name__ == '__main__':
    # mp = mp.get_context('spawn')
    main()


