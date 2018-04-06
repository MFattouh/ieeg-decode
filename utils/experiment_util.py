from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing
from resampy import resample
from scipy.io import loadmat
from utils.pytorch_util import *
from models.hybrid import *


def create_dataset(subject, new_srate_x, new_srate_y, window, stride, crop_len=0, batch_size=16):
    dataset_path = '/data/schirrmr/fattouhm/datasets/%s.mat' % subject
    x2y_ratio = new_srate_x / new_srate_y
    mat_dataset = loadmat(dataset_path)
    datasets_list = []
    x_list, y_list = [], []
    num_trials = (np.char.find(list(mat_dataset.keys()), 'trial') != -1).sum()
    for idx in range(1, num_trials + 1):
        # read the trial from the storage
        # ignore the first 10 sec. to avoid artifacts (might be due highpass )
        srate = int(mat_dataset['trial%d' % idx][0, 0]['srate'].squeeze())
        X = mat_dataset['trial%d' % idx][0, 0]['X'][:, 10 * srate:].squeeze()
        y = mat_dataset['trial%d' % idx][0, 0]['y'][:, 10 * srate:].squeeze()

        game_type = str(mat_dataset['trial%d' % idx][0, 0]['gameType'].squeeze())
        if game_type == 'pause' or game_type.find('discrete') != -1:
            continue

        X = X.T
        X_resampled = resample(X, srate, new_srate_x, filter='kaiser_fast', axis=0)
        y_resampled = resample(y, srate, new_srate_y, filter='kaiser_fast', axis=0)
        if crop_len > 0:
            x_crops, y_crops = crops_from_trial(X_resampled, y_resampled, crop_len*new_srate_x, stride=stride,
                                                time_last=True, normalize=True, dummy_idx=0)
            x_list.extend(x_crops)
            y_list.extend(y_crops)
            train_split = int(np.ceil(0.7 * len(x_list)/batch_size)*batch_size)
            valid_split = train_split + int(np.ceil(0.3 * len(x_list)/batch_size)*batch_size)
            training_dataset = ConcatCrops(x_list[:train_split], y_list[:train_split])
            training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
            valid_dataset = ConcatCrops(x_list[train_split:valid_split], y_list[train_split:valid_split])
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
        else:
            y_normalized = MinMaxScaler(feature_range=(-1, 1)).fit_transform(y_resampled.reshape(-1, 1)).squeeze()
            standardized_Xs = exponential_running_standardize(X_resampled, init_block_size=250, factor_new=0.001,
                                                              eps=1e-4)
            dataset = ECoGDatast(standardized_Xs, y_normalized, window=window, stride=stride, x2y_ratio=x2y_ratio)
            datasets_list.append(dataset)

            training_dataset = ConcatDataset(datasets_list[:-2])
            training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            valid_dataset = ConcatDataset([datasets_list[-2]])
            valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return training_loader, valid_loader, X.shape[-1]


def make_weights(crop_len, num_relaxed_samples, type='qubic'):
    if type == 'qubic':
        weights = np.ones((crop_len, 1), dtype=np.float32).squeeze()
        if num_relaxed_samples > 0:
            x = np.arange(num_relaxed_samples, dtype=np.float32)
            weights[:num_relaxed_samples] = x ** 3 / (num_relaxed_samples ** 3)
        return weights
    if type == 'step':
        weights = np.ones((crop_len, 1), dtype=np.float32).squeeze()
        if num_relaxed_samples > 0:
            weights[:num_relaxed_samples] = 0
        return weights


def run_experiment(model, optimizer, loss_fun, metric, training_loader, training_writer, valid_loader, valid_writer,
                   weights_path, max_epochs, eval_every, cuda):
    min_loss = float('inf')
    for epoch in range(max_epochs):
        # scheduler.step()
        train(model, training_loader, optimizer, loss_fun, keep_state=False, clip=10, cuda=cuda)
        if epoch % eval_every == 0:
            _, train_corr = evaluate(model, training_loader, loss_fun, metric, keep_state=False, writer=training_writer,
                                     epoch=epoch, cuda=cuda)
            valid_loss, valid_corr = evaluate(model, valid_loader, loss_fun, metric, keep_state=False,
                                              writer=valid_writer, epoch=epoch, cuda=cuda)

            if valid_loss < min_loss:
                print('found new valid_loss value', valid_loss, 'at epoch', epoch)
                # save model parameters
                torch.save(model.state_dict(), weights_path)
                min_loss = valid_loss

            if np.isnan(train_corr) or np.isnan(valid_corr):
                break



