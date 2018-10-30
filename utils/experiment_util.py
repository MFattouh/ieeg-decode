from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing
from utils.pytorch_util import *
import h5py
import logging
import os
import datetime
from utils.config import cfg
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.modules import Expression
from braindecode.models.deep4 import Deep4Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet as Shallow
from torch.nn.functional import mse_loss
from models.hybrid import HybridModel
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn as nn


logger = logging.getLogger('__name__')


def read_dataset(dataset_path, dataset_name):
    datasets_list = []
    with h5py.File(dataset_path, 'r') as hf:
        trials = [hf[obj_ref] for obj_ref in hf[dataset_name][0]]
        if len(trials) > 1:
            trials = trials[:-1]
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


def read_multi_datasets(input_datasets_path, dataset_name):
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


def concatenate_trials(trials):
    return ConcatDataset(
        [ECoGDatast(X, y, window=cfg.TRAINING.CROP_LEN, stride=cfg.TRAINING.INPUT_STRIDE,
                    x2y_ratio=cfg.TRAINING.INPUT_SAMPLING_RATE / cfg.TRAINING.OUTPUT_SAMPLING_RATE,
                    input_shape='ct', dummy_idx=cfg.TRAINING.DUMMY_IDX) for (X, y) in trials])


def create_loaders(trials, train_split, valid_split=None):
    training_trials = [trials[idx] for idx in train_split]
    training_dataset = concatenate_trials(training_trials)
    training_loader = DataLoader(training_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, drop_last=False,
                                 pin_memory=True, num_workers=4)

    if valid_split is not None:
        valid_trials = [trials[idx] for idx in valid_split]
        valid_dataset = concatenate_trials(valid_trials)
        valid_loader = DataLoader(valid_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, drop_last=False,
                                  pin_memory=True, num_workers=4)
    else:
        valid_loader = None

    return training_loader, valid_loader


def make_weights(crop_len, num_relaxed_samples, dtype='qubic'):
    if dtype == 'qubic':
        weights = np.ones((crop_len, 1), dtype=np.float32).squeeze()
        if num_relaxed_samples > 0:
            x = np.arange(num_relaxed_samples, dtype=np.float32)
            weights[:num_relaxed_samples] = x ** 3 / (num_relaxed_samples ** 3)
        return weights
    if dtype == 'step':
        weights = np.ones((crop_len, 1), dtype=np.float32).squeeze()
        if num_relaxed_samples > 0:
            weights[:num_relaxed_samples] = 0
        return weights


def create_model(model_type, in_channels, num_classes, cuda=True):
    num_relaxed_samples = 681
    if model_type == 'rnn':
        model = HybridModel(in_channels=in_channels, output_stride=int(cfg.HYBRID.OUTPUT_STRIDE))

        if cfg.HYBRID.OUTPUT_STRIDE > 1:
            loss_fun = mse_loss
            metric = CorrCoeff().corrcoeff
        else:
            weights = make_weights(cfg.TRAINING.CROP_LEN, num_relaxed_samples, dtype='step')
            weights_tensor = torch.from_numpy(weights)
            if cuda:
                weights_tensor = weights_tensor.cuda()
            loss_fun = WeightedMSE(weights_tensor)
            metric = CorrCoeff(weights).weighted_corrcoef

    elif model_type == 'deep4':
        model = Deep4Net(in_chans=in_channels, n_classes=num_classes, input_time_length=cfg.TRAINING.CROP_LEN,
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

        loss_fun = mse_loss
        metric = CorrCoeff().corrcoeff

    elif model_type == 'shallow':
        model = Shallow(in_chans=in_channels, n_classes=num_classes, input_time_length=cfg.TRAINING.CROP_LEN,
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

        num_dropped_samples = 113
        weights = make_weights(cfg.TRAINING.CROP_LEN - num_dropped_samples, num_relaxed_samples - num_dropped_samples,
                               dtype='step')
        weights_tensor = torch.from_numpy(weights)
        if cuda:
            weights_tensor = weights_tensor.cuda()
        loss_fun = WeightedMSE(weights_tensor)
        metric = CorrCoeff(weights).weighted_corrcoef
    elif model_type == 'hybrid':
        cfg.HYBRID.SPATIAL_CONVS['num_filters'] = [in_channels]
        model = HybridModel(in_channels=in_channels, output_stride=int(cfg.HYBRID.OUTPUT_STRIDE))
        num_dropped_samples = 121
        weights = make_weights(cfg.TRAINING.CROP_LEN - num_dropped_samples, num_relaxed_samples - num_dropped_samples,
                               dtype='step')
        weights_tensor = torch.from_numpy(weights)
        if cuda:
            weights_tensor = weights_tensor.cuda()
        loss_fun = WeightedMSE(weights_tensor)
        metric = CorrCoeff(weights).weighted_corrcoef

    elif model_type == 'tcn':
        raise NotImplementedError
    optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZATION.BASE_LR, weight_decay=cfg.OPTIMIZATION.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.TRAINING.MAX_EPOCHS)
    if cuda:
        model.cuda()
    return model, optimizer, scheduler, loss_fun, metric


def evaluate(model, data_loader, loss_fun, metric, keep_state=False, writer=None, epoch=0, cuda=False):
    model.eval()
    # loop over the dataset
    avg_loss = 0
    targets = []
    preds = []
    with torch.no_grad():
        for itr, (data, target_cpu) in enumerate(data_loader):
            data, target = Variable(data), Variable(target_cpu)
            if cuda:
                data, target = data.cuda(), target.cuda()

            if keep_state:
                if itr == 0:
                    hidden = None
                else:
                    if model.rnn_type == 'lstm':
                        for h in hidden:
                            h.detach_()
                    else:
                        hidden.detach_()
                output, hidden = model(data, hidden)  # NxTxnum_classes
            else:
                output = model(data)

            output_size = list(output.size())
            seq_len = output_size[1] if len(output_size) > 1 else output_size[0]
            batch_size = output_size[0] if len(output_size) > 1 else 1
            num_classes = output_size[2] if len(output_size) > 2 else 1
            avg_loss += loss_fun(output.squeeze(), target[:, -seq_len:].squeeze()).detach().cpu().numpy().squeeze()
            # compute the correlation coff. for each seq. in batch
            target = target_cpu[:, -seq_len:].numpy().squeeze()
            output = output.detach().cpu().numpy().squeeze()
            if seq_len > 1:
                if itr == 0:
                    cum_corr = np.zeros((num_classes, 1))
                    valid_corr = np.zeros((num_classes, 1))
                if num_classes == 1:
                    corr = np.arctanh(metric(target, output))
                    if not np.isnan(corr):
                        cum_corr[0] += corr
                        valid_corr[0] += 1
                else:
                    for class_idx in range(num_classes):
                        # compute correlation, apply fisher's transform
                        corr = np.arctanh(metric(target[:, :, class_idx], output[:, :, class_idx]))
                        if not np.isnan(corr):
                            cum_corr[class_idx] += corr
                            valid_corr[class_idx] += 1

            # one sample per-sequence -> store results and compute corr. at once
            else:
                targets.append(target)
                preds.append(output)

    # divide by the number of mini-batches
    avg_loss /= len(data_loader)
    if seq_len == 1:
        targets = np.concatenate(targets, axis=0).squeeze()
        preds = np.concatenate(preds, axis=0).squeeze()
        num_classes = targets.shape[-1] if len(targets.shape) > 1 else 1
        if num_classes > 1:
            avg_corr = dict()
            for class_idx in range(num_classes):
                # compute correlation, apply fisher's transform
                avg_corr[f"Class{class_idx}"] = metric(targets[:, class_idx], preds[:, class_idx])
        else:
            avg_corr = metric(targets, preds)
    else:
        # average the correlations across over iterations apply inverse fisher's transform find mean over batch
        if num_classes == 1:
            avg_corr = np.tanh(cum_corr.squeeze() / valid_corr.squeeze()).mean()
        else:
            avg_corr = dict()
            for i in range(num_classes):
                avg_corr['Class%d' % i] = np.tanh(cum_corr[i] / valid_corr[i]).mean()

    if writer is not None:
        writer.add_scalar('loss', avg_loss, epoch)
        if num_classes == 1:
            writer.add_scalar('corr', avg_corr, epoch)
        else:
            writer.add_scalars('corr', avg_corr, epoch)
    return avg_loss, avg_corr


def train(model, data_loader, optimizer, loss_fun, keep_state=False, clip=0, cuda=False):
    model.train()
    for itr, (data, target_cpu) in enumerate(data_loader):
        # data, target = Variable(data.transpose(1, 0)), Variable(target_cpu.squeeze(0))
        data, target = Variable(data), Variable(target_cpu)
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # detach to stop back-propagation to older state
        if keep_state:
            if model.rnn_type == 'lstm':
                for h in hidden:
                    h.detach_()
            else:
                hidden.detach_()

            output, hidden = model(data, hidden)  # NxTxnum_classes

        else:
            output = model(data)

        output_size = list(output.size())
        seq_len = output_size[1] if len(output_size) > 1 else output_size[0]
        loss = loss_fun(output.squeeze(), target[:, -seq_len:].squeeze())
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()


def run_eval(model, loss_fun, metric, valid_loader, weights_path, cuda):
    assert os.path.exists(weights_path), 'weights_path does not exists'
    model.load_state_dict(torch.load(weights_path))
    valid_loss, valid_corr = evaluate(model, valid_loader, loss_fun, metric, keep_state=False, cuda=cuda)
    return valid_corr, valid_loss


def run_training(model, optimizer, scheduler, loss_fun, metric, training_loader, training_writer, weights_path, cuda):
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        path, ext = os.path.splitext(weights_path)
        weights_path = path + '_best' + ext
    min_loss = float('inf')
    last_best = 0
    for epoch in range(cfg.TRAINING.MAX_EPOCHS+1):
        if epoch % cfg.TRAINING.EVAL_TRAIN_EVERY == 0 or epoch % cfg.TRAINING.EVAL_VALID_EVERY == 0:
            logger.info(f"===========epoch: {epoch}=============")
            logger.info(f"Started at: {datetime.datetime.now():%d %b: %H:%M}")

        # report init. error before training
        if epoch == 0:
            train_loss, train_corr = evaluate(model, training_loader, loss_fun, metric, keep_state=False,
                                              writer=training_writer, epoch=epoch, cuda=cuda)
            logger.info(f'init. training loss value: {train_loss}')
            logger.info(f'init. training corr: {train_corr}')

            if type(train_corr) == dict:
                max_acc = dict(zip(list(train_corr.keys()), [-float('inf')] * len(train_corr)))
            else:
                max_acc = -float('inf')

            continue

        if scheduler is not None:
            scheduler.step(epoch-1)

        train(model, training_loader, optimizer, loss_fun, keep_state=False, clip=0, cuda=cuda)

        if epoch % cfg.TRAINING.EVAL_TRAIN_EVERY == 0:
            train_loss, train_corr = evaluate(model, training_loader, loss_fun, metric, keep_state=False,
                                              writer=training_writer, epoch=epoch, cuda=cuda)
            logger.info(f'training loss: {train_loss}')
            logger.info(f'training corr: {train_corr}')

            if train_loss < min_loss:
                logger.info(f'found new valid loss: {train_loss}')
                min_loss = train_loss
                last_best = epoch

            if type(train_corr) == dict:
                for task, corr in train_corr.items():
                    if corr > max_acc[task]:
                        max_acc[task] = corr

            elif train_corr > max_acc:
                max_acc = train_corr

        # if training stalls
        if epoch - last_best > 200:
            logger.info("valid loss have not decreased for 200 epochs!")
            logger.info("stop training to avoid overfitting!")
            break

        if type(train_corr) == dict:
            if np.any(np.isnan(list(train_corr.values()))) or np.any(np.isnan(list(train_corr.values()))):
                logger.error('Training stalled')
                break
        else:
            if np.isnan(train_corr) or np.isnan(train_corr):
                logger.error('Training stalled')
                break

    # report best values
    logger.info(f'Best valid loss: {min_loss}')
    logger.info(f'Best valid corr: {max_acc}')

    # save model parameters
    torch.save(model.state_dict(), weights_path)
    return max_acc, min_loss


def run_cv_experiment(model, optimizer, scheduler, loss_fun, metric, training_loader, training_writer, valid_loader,
                      valid_writer, weights_path, cuda):
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        path, ext = os.path.splitext(weights_path)
        weights_path = path + '_best' + ext
    min_loss = float('inf')
    last_best = 0
    for epoch in range(cfg.TRAINING.MAX_EPOCHS+1):
        if epoch % cfg.TRAINING.EVAL_TRAIN_EVERY == 0 or epoch % cfg.TRAINING.EVAL_VALID_EVERY == 0:
            logger.info(f"===========epoch: {epoch}=============")
            logger.info(f"Started at: {datetime.datetime.now():%d %b: %H:%M}")

        # report init. error before training
        if epoch == 0:
            train_loss, train_corr = evaluate(model, training_loader, loss_fun, metric, keep_state=False,
                                              writer=training_writer, epoch=epoch, cuda=cuda)
            logger.info(f'init. training loss value: {train_loss}')
            logger.info(f'init. training corr: {train_corr}')

            valid_loss, valid_corr = evaluate(model, valid_loader, loss_fun, metric, keep_state=False,
                                              writer=valid_writer, epoch=epoch, cuda=cuda)
            logger.info(f'init. valid loss: {valid_loss}')
            logger.info(f'init. valid corr: {valid_corr}')

            if type(train_corr) == dict:
                max_acc = dict(zip(list(train_corr.keys()), [-float('inf')] * len(train_corr)))
            else:
                max_acc = -float('inf')

            continue

        if scheduler is not None:
            scheduler.step(epoch-1)

        train(model, training_loader, optimizer, loss_fun, keep_state=False, clip=0, cuda=cuda)

        if epoch % cfg.TRAINING.EVAL_TRAIN_EVERY == 0:
            train_loss, train_corr = evaluate(model, training_loader, loss_fun, metric, keep_state=False,
                                              writer=training_writer, epoch=epoch, cuda=cuda)
            logger.info(f'training loss: {train_loss}')
            logger.info(f'training corr: {train_corr}')

        if epoch % cfg.TRAINING.EVAL_VALID_EVERY == 0:
            valid_loss, valid_corr = evaluate(model, valid_loader, loss_fun, metric, keep_state=False,
                                              writer=valid_writer, epoch=epoch, cuda=cuda)
            logger.info(f'valid loss: {valid_loss}')
            logger.info(f'valid corr: {valid_corr}')

            if valid_loss < min_loss:
                logger.info(f'found new valid loss: {valid_loss}')
                # save model parameters
                torch.save(model.state_dict(), weights_path)
                min_loss = valid_loss
                last_best = epoch

            if type(valid_corr) == dict:
                for task, corr in valid_corr.items():
                    if corr > max_acc[task]:
                        max_acc[task] = corr

            elif valid_corr > max_acc:
                max_acc = valid_corr

        # if training stalls
        if epoch - last_best > 200:
            logger.info("valid loss have not decreased for 200 epochs!")
            logger.info("stop training to avoid overfitting!")
            break

        if type(valid_corr) == dict:
            if np.any(np.isnan(list(train_corr.values()))) or np.any(np.isnan(list(valid_corr.values()))):
                logger.error('Training stalled')
                break
        else:
            if np.isnan(train_corr) or np.isnan(valid_corr):
                logger.error('Training stalled')
                break

    # report best values
    logger.info(f'Best valid loss: {min_loss}')
    logger.info(f'Best valid corr: {max_acc}')

    return max_acc, min_loss
