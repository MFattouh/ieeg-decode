from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing
from utils.pytorch_util import *
from models.hybrid import *
import h5py
import logging
import os

logger = logging.getLogger('__name__')


def read_dataset(dataset_path, dataset_name):
    datasets_list = []
    with h5py.File(dataset_path, 'r') as hf:
        trials = [hf[obj_ref] for obj_ref in hf[dataset_name][0]]
        for idx, trial in enumerate(trials, 1):
            # read data
            X = trial['ieeg'][:]
            y = trial['traj'][:][:].squeeze()
            if X.ndim < 2:
                logger.warning('irregular trial shape encountered. Trial%d will be ignored' % idx)
                continue
            in_channels = X.shape[0]
            if idx == 1:
                in_channels = X.shape[0]
            else:
                if in_channels != X.shape[0]:
                    logger.exception('different channels in different trials %d != %d' % (in_channels, X.shape[0]))
            datasets_list.append((X, y))

    return datasets_list, in_channels


def read_multi_datasets(input_datasets_path, dataset_name, window, stride, x2y_ratio, dummy_idx):
    datasets_list = []
    with h5py.File(input_datasets_path[0], 'r') as hf:
        trials = [hf[obj_ref] for obj_ref in hf[dataset_name][0]]
        for idx, trial in enumerate(trials, 1):
                # read data
                X = trial['ieeg'][:]
                y = trial['traj'][:][:].squeeze()
                if X.ndim < 2:
                    logger.warning('irregular trial shape encountered. Trial%d will be ignored' %idx)
                    continue
                in_channels = X.shape[0]
                if idx == 1:
                    in_channels = X.shape[0]
                else:
                    if in_channels != X.shape[0]:
                        logger.exception('different channels in different trials %d != %d' % (in_channels, X.shape[0]))

                datasets_list.append((X, y))

    for dataset_path in input_datasets_path[1:]:
        with h5py.File(dataset_path, 'r') as hf:
            trials = [hf[obj_ref] for obj_ref in hf['D'][0]]
            for idx, trial in enumerate(trials):
                    # read data
                    X = trial['ieeg'][:]
                    if X.ndim < 2:
                        logger.warning('irregular trial shape encountered. Trial%d will be ignored' % (idx+1))
                        continue
                    np.testing.assert_equal(X, datasets_list[idx][0], 'iEEG channels did not match')
                    datasets_list[idx][1] = np.c_[datasets_list[idx][1],
                                                  trial['traj'][:][:].squeeze()]

    return datasets_list, in_channels


def create_loaders(trials, train_split, valid_split, batch_size, dummy_idx):
    training_trials = [trials[idx] for idx in train_split]
    training_dataset = ConcatDataset(
        [ECoGDatast(X, y, window=cfg.TRAINING.CROP_LEN, stride=cfg.TRAINING.INPUT_STRIDE,
                    x2y_ratio=cfg.TRAINING.INPUT_SAMPLING_RATE / cfg.TRAINING.OUTPUT_SAMPLING_RATE,
                    input_shape='ct', dummy_idx=dummy_idx)
         for X, y in training_trials])
    training_loader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

    valid_trials = [trials[idx] for idx in valid_split]
    valid_dataset = ConcatDataset(
        [ECoGDatast(X, y, window=cfg.TRAINING.CROP_LEN, stride=cfg.EVAL.INPUT_STRIDE,
                    x2y_ratio=cfg.TRAINING.INPUT_SAMPLING_RATE / cfg.TRAINING.OUTPUT_SAMPLING_RATE,
                    input_shape='ct', dummy_idx=dummy_idx)
         for X, y in valid_trials])
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)

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


def run_eval(model, loss_fun, metric, valid_loader, weights_path, cuda):
    assert os.path.exists(weights_path), 'weights_path does not exists'
    model.load_state_dict(torch.load(weights_path))
    valid_loss, valid_corr = evaluate(model, valid_loader, loss_fun, metric, keep_state=False, cuda=cuda)
    return valid_corr, valid_loss


def run_experiment(model, optimizer, loss_fun, metric, training_loader, training_writer, valid_loader, valid_writer,
                   weights_path, max_epochs, eval_train_every, eval_valid_every, cuda):
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        path, ext = os.path.splitext(weights_path)
        weights_path = path + '_best' + ext
    min_loss = float('inf')
    last_best = 0
    for epoch in range(max_epochs+1):
        # scheduler.step()
        train(model, training_loader, optimizer, loss_fun, keep_state=False, clip=10, cuda=cuda)
        if epoch % eval_train_every == 0:
            train_loss, train_corr = evaluate(model, training_loader, loss_fun, metric, keep_state=False,
                                              writer=training_writer, epoch=epoch, cuda=cuda)
            logger.info(f"===========epoch: {epoch}=============")
            logger.info(f'training loss: {train_loss}')
            logger.info(f'training corr: {train_corr}')
        if type(train_corr) == dict:
            max_acc = dict(zip(list(train_corr.keys()), [-float('inf')] * len(train_corr)))
        else:
            max_acc = -float('inf')
        if epoch % eval_valid_every == 0:
            valid_loss, valid_corr = evaluate(model, valid_loader, loss_fun, metric, keep_state=False,
                                              writer=valid_writer, epoch=epoch, cuda=cuda)

            if epoch % eval_train_every != 0:
                logger.info(f"===========epoch: {epoch}=============")

            logger.info(f'valid loss: {valid_loss}')
            logger.info(f'valid corr: {valid_corr}')

            if valid_loss < min_loss:
                logger.info(f'found new valid loss: {valid_loss}')
                # save model parameters
                torch.save(model.state_dict(), weights_path)
                min_loss = valid_loss
                last_best = epoch

            if epoch - last_best > 200:
                logger.info("valid loss have not decreased for 200 epochs!")
                logger.info("stop training to avoid overfitting!")
                return max_acc, min_loss

            if type(valid_corr) == dict:
                for task, corr in valid_corr.items():
                    if corr > max_acc[task]:
                        max_acc[task] = corr

            elif valid_corr > max_acc:
                max_acc = valid_corr

        # if training stalls
        if type(valid_corr) == dict:
            if np.any(np.isnan(list(train_corr.values()))) or np.any(np.isnan(list(valid_corr.values()))):
                logger.error('Training stalled')
                break
        else:
            if np.isnan(train_corr) or np.isnan(valid_corr):
                logger.error('Training stalled')
                break

    # report last acc
    logger.info(f'final valid loss: {valid_loss}')
    logger.info(f'final valid corr: {valid_corr}')

    return max_acc, min_loss
