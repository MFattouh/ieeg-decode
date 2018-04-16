from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing
from utils.pytorch_util import *
from models.hybrid import *
import h5py
import logging
from datetime import datetime

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('__name__')


def read_dataset(dataset_path, window, stride=1):
    datasets_list = []
    with h5py.File(dataset_path, 'r') as hf:
        trials = [hf[obj_ref] for obj_ref in hf['D'][0]]
        for idx, trial in enumerate(trials, 1):
            try:
                # read data
                X = trial['ieeg'][:]
                y = trial['traj'][:][:].squeeze()
                datasets_list.append(ECoGDatast(X, y, window, stride, input_shape='ct'))
                in_channels = X.shape[0]
            except ValueError as e:
                logger.warning('exception found while creating dataset object from trial %s \n%s' % (idx, e))
                continue

    return datasets_list, in_channels


def create_loader(datasets, train_split, valid_split, batch_size):
    training_dataset = ConcatDataset([datasets[idx] for idx in train_split])
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)
    valid_dataset= ConcatDataset([datasets[idx] for idx in valid_split])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=2)

    return training_loader, valid_loader


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
    max_acc = -float('inf')
    for epoch in range(max_epochs+1):
        # scheduler.step()
        train(model, training_loader, optimizer, loss_fun, keep_state=False, clip=10, cuda=cuda)
        if epoch % eval_every == 0:
            train_loss, train_corr = evaluate(model, training_loader, loss_fun, metric, keep_state=False,
                                              writer=training_writer, epoch=epoch, cuda=cuda)
            print(str(datetime.now()),
                  ': training loss value', train_loss, 'training corr', train_corr,
                  'at epoch', epoch)
            valid_loss, valid_corr = evaluate(model, valid_loader, loss_fun, metric, keep_state=False,
                                              writer=valid_writer, epoch=epoch, cuda=cuda)
            print(str(datetime.now()),
                  ': valid loss value', valid_loss, 'valid corr', valid_corr,
                  'at epoch', epoch)

            if valid_loss < min_loss:
                print(str(datetime.now()),
                      ': found new valid loss value', valid_loss, 'at epoch', epoch)
                # save model parameters
                torch.save(model.state_dict(), weights_path)
                min_loss = valid_loss

            if valid_corr > max_acc:
                max_acc = valid_corr

            # if training stalls
            if np.isnan(train_corr) or np.isnan(valid_corr):
                break

    # report last acc
    print(str(datetime.now()),
          ':final valid loss:', valid_loss, 'final valid corr', valid_corr)

    return max_acc
