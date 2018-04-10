from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing
from resampy import resample
from utils.pytorch_util import *
from models.hybrid import *
import h5py


def create_dataset(dataset_path, new_srate_x, new_srate_y, window, stride=0, batch_size=16):

    x2y_ratio = new_srate_x/new_srate_y

    datasets_list = []
    with h5py.File(dataset_path, 'r') as hf:
        trials = list(hf.keys())
        for idx, trial in enumerate(trials):
            # read the trial from the storage
            X, y, srate, game_type = load_trial(hf, trial, full_load=False)
            if game_type == 'pause' or game_type.find('discrete') != -1:
                print('will be ignored')
                continue

            X = X.T
            X_resampled = resample(X, srate, new_srate_x, filter='kaiser_fast', axis=0)
            y_resampled = resample(y, srate, new_srate_y, filter='kaiser_fast', axis=0)

            datasets_list.append(ECoGDatast(X_resampled, y_resampled, window * new_srate_x, stride, x2y_ratio=x2y_ratio))

    training_dataset = ConcatDataset(datasets_list[:-1])
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    valid_dataset = ConcatDataset([datasets_list[-1]])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

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



