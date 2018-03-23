import torch
import os
import h5py
from torch.utils.data import DataLoader
import torch.multiprocessing
from resampy import resample
import numpy as np
from scipy.io import loadmat
from braindecode.datautil.signalproc import exponential_running_standardize
from sklearn.preprocessing import MinMaxScaler
from pytorch_util import *
from hybrid import *
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import torch as th
from torch import optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

os.sys.path.insert(0, '.')
os.sys.path.insert(0, '/data/schirrmr/fattouhm/datasets/')

subject = 'FR3'
FIRST_IDX = 40
NUM_EXPERIMENTS = 10
MAX_EPOCHS = 200
EVAL_EVERY = 20
cuda = True


def create_dataset(subject, new_srate_x, new_srate_y, window, stride):
    dataset_path = '/data/schirrmr/fattouhm/datasets/new_%s.mat' % subject
    x2y_ratio = new_srate_x / new_srate_y
    mat_dataset = loadmat(dataset_path)
    datasets_list = []
    num_trials = (np.char.find(list(mat_dataset.keys()), 'trial') != -1).sum()
    for idx in range(1, num_trials + 1):
        # read the trial from the storage
        # ignore the first 10 sec. to avoid artifacts (might be due highpass )
        srate = int(mat_dataset['trial%d' % idx][0, 0]['srate'].squeeze())
        X = mat_dataset['trial%d' % idx][0, 0]['X'][:, 10 * srate:].squeeze()
        y = mat_dataset['trial%d' % idx][0, 0]['y'][:, 10 * srate:].squeeze()

        game_type = mat_dataset['trial%d' % idx][0, 0]['gameType'].squeeze()

        X = X.T
        # check the distribution of targets before and after normalization
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        # now lets prepare the data
        X_resampled = resample(X, srate, new_srate_x, filter='kaiser_fast', axis=0)
        y_resampled = resample(y, srate, new_srate_y, filter='kaiser_fast', axis=0)
        standardized_Xs = exponential_running_standardize(X_resampled, init_block_size=window, factor_new=0.001,
                                                          eps=1e-4)
        dataset = ECoGDatast(standardized_Xs, y_resampled, window=window, stride=stride, x2y_ratio=x2y_ratio)
        datasets_list.append(dataset)
    return datasets_list, X.shape[-1]


def main():
    new_srate_x = 250
    new_srate_y = 250
    x2y_ratio = new_srate_x / new_srate_y
    n_sec = 1
    # window size
    window = n_sec * new_srate_x
    # we want no overlapping data
    stride = window
    datasets_list, in_channels = create_dataset(subject, new_srate_x, new_srate_y, window, stride)
    training_dataset = ConcatDataset(datasets_list[:-2], batch_first=True, time_last=True)
    training_loader = DataLoader(training_dataset, batch_size=1, shuffle=False, num_workers=2)
    valid_dataset = ConcatDataset([datasets_list[-2]], batch_first=True, time_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=2)
    hidden_size = 64
    n_layers = 3
    initializer = torch.nn.init.xavier_normal
    experiment_name = 'lr_wd'
    for run in range(NUM_EXPERIMENTS):
        run = run + FIRST_IDX
        model = HybridModel(rnn_type='gru', num_classes=1, rnn_hidden_size=hidden_size, rnn_layers=n_layers,
                            in_channels=in_channels,
                            channel_filters=[], time_filters=[], time_kernels=[], fc_size=[10],
                            output_stride=int(x2y_ratio), batch_norm=False, max_length=window, dropout=0.3,
                            initializer=initializer)

        # this exponent will results in a value between [-5, -3] -> learning rates = [10e-5, 10e-1]
        r = -2*np.random.rand(1) - 3
        learning_rate = 10 ** r[0]
        # range of r is [-3, -6] -> wd = 1 - 10^[-3, .., -6]
        r = -3*np.random.rand(1) - 3
        # wd_const = 1 - 10 ** r[0]
        wd_const = 0
        cuda = True
        if cuda:
            model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd_const)
        loss_fun = F.mse_loss
        # lr_decay_const = 1
        # scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

        log_dir = 'pos/%s/%s/run%d/' % (experiment_name, subject, run)
        training_writer = SummaryWriter(log_dir + 'train')
        valid_writer = SummaryWriter(log_dir + 'valid')
        training_writer.add_text('Model parameters', str(HybridModel.get_meta(model)))
        training_writer.add_text('Initializer', str(initializer))
        training_writer.add_text('Description', ' Subject %s. keep state between examples.'
                                                'Adam opt. normalized targets. Xavir init.' % subject)
        training_writer.add_text('Learning Rate', str(learning_rate))
        training_writer.add_text('Weight Decay', str(wd_const))
        training_writer.add_text('Input window [sec]', str(n_sec))
        training_writer.add_text('Input srate [Hz]', str(new_srate_x))
        training_writer.add_text('Output srate [Hz]', str(new_srate_y))
        # training_writer.add_text('learning rate decay const', str(lr_decay_const))

        weights_path = log_dir + 'weights.pt'

        min_loss = float('inf')
        for epoch in range(MAX_EPOCHS):
            # scheduler.step()
            train(model, training_loader, optimizer, loss_fun, keep_state=True, clip=10, cuda=cuda)
            if epoch % EVAL_EVERY == 0:
                _, train_corr = evaluate(model, training_loader, loss_fun, keep_state=True, writer=training_writer,
                                         epoch=epoch, cuda=cuda)
                valid_loss, valid_corr = evaluate(model, valid_loader, loss_fun, keep_state=True,
                                                  writer=valid_writer, epoch=epoch, cuda=cuda)

                if valid_loss < min_loss:
                    print('found new valid_loss value', valid_loss, 'at epoch', epoch)
                    # save model parameters
                    torch.save(model.state_dict(), weights_path)
                    min_loss = valid_loss

                if np.isnan(train_corr) or np.isnan(valid_corr):
                    break

        test_dataset = ConcatDataset([datasets_list[-1]], batch_first=True, time_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

        model.load_state_dict(torch.load(weights_path))

        if cuda:
            model = model.cuda()

        test_loss, test_corr = evaluate(model, test_loader, loss_fun, writer=None, cuda=cuda)
        del test_loader
        training_writer.add_text('test corr', str(test_corr))


if __name__ == '__main__':
    main()


