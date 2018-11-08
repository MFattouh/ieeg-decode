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
from resampy import resample

logger = logging.getLogger('__name__')


def read_dataset(dataset_path, dataset_name):
    datasets_list = []
    with h5py.File(dataset_path, 'r') as hf:
        trials = [hf[obj_ref] for obj_ref in hf[dataset_name][0]]
        for idx, trial in enumerate(trials, 1):
            if trial['ieeg'].ndim != 2:
                logger.warning('irregular trial shape {} encountered in trial {}. '
                               'Expected 2D array! This trial will be discarded'.format(X.shape, idx))
                continue

            # read data
            X = trial['ieeg'][:]
            y = trial['traj'][:][:].squeeze()

            # resample if necessary
            srate = int(trial['srate'][:][:])
            # TODO: HANDCODED axis
            if srate > cfg.TRAINING.INPUT_SAMPLING_RATE:
                X = resample(X, srate, cfg.TRAINING.INPUT_SAMPLING_RATE, axis=1)
            else:
                assert srate == cfg.TRAINING.INPUT_SAMPLING_RATE, \
                    'The desired sampling rate "{}" is larger than the original sampling rate "{}"!'.format(
                        cfg.TRAINING.INPUT_SAMPLING_RATE, srate)

            if srate > cfg.TRAINING.OUTPUT_SAMPLING_RATE:
                y = resample(y, srate, cfg.TRAINING.OUTPUT_SAMPLING_RATE, axis=0)
            else:
                assert srate == cfg.TRAINING.OUTPUT_SAMPLING_RATE, \
                    'The desired sampling rate "{}" is larger than the original sampling rate "{}"!'.format(
                        cfg.TRAINING.OUTPUT_SAMPLING_RATE, srate)

            # ignore trials if there isn't enough input for one crop
            if X.shape[1] < cfg.TRAINING.CROP_LEN:
                logger.warning('Trial {} is too short. Only {} samples found!'.foramt(X.shape, idx))
                continue

            # TODO: HANDCODED axis
            in_channels = X.shape[0]
            if idx == 1:
                in_channels = X.shape[0]
            else:
                assert in_channels == X.shape[0],\
                    'different channels in different trials {} != {}'.format(in_channels, X.shape[0])

            datasets_list.append((X, y))

    return datasets_list, in_channels


def read_multi_datasets(input_datasets_path, dataset_name):
    datasets_list, in_channels = read_dataset(input_datasets_path[0], dataset_name)

    # TODO: if read_dataset ignores some trials we have no clue which!
    for dataset_path in input_datasets_path[1:]:
        with h5py.File(dataset_path, 'r') as hf:
            trials = [hf[obj_ref] for obj_ref in hf[dataset_name][0]]
            for idx, trial in enumerate(trials):
                # ignore trials if there isn't enough input for one training sample
                # read data
                y = trial['traj'][:][:].squeeze()
                srate = int(trial['srate'][:][:])
                if srate > cfg.TRAINING.OUTPUT_SAMPLING_RATE:
                    # we need to downsample the targets
                    y = resample(y, srate, cfg.TRAINING.OUTPUT_SAMPLING_RATE, axis=0)

                else:
                    assert srate == cfg.TRAINING.OUTPUT_SAMPLING_RATE, \
                        'The desired sampling rate "{}" is larger than the original sampling rate "{}"!'.format(
                            cfg.TRAINING.OUTPUT_SAMPLING_RATE, srate)

                assert y.shape == datasets_list[1].shape, "shape miss-match between targets in different tasks!"
                datasets_list[idx][1] = np.c_[datasets_list[idx][1], y]

    return datasets_list, in_channels


def lr_finder(model, loss_fun, optimizer, training_trials, output_path, min_lr=-6, max_lr=-2, steps=100, cuda=True):
    lr_values = np.logspace(min_lr, max_lr, steps).tolist()
    losses = []
    if cuda:
        model.cuda()
    model.train()
    training_loader = create_training_loader(training_trials)
    training_iterator = iter(training_loader)
    with torch.enable_grad():
        for lr_value in lr_values:
            optimizer.zero_grad()
            # init. the optimizer with the lr value
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_value

            try:
                data, target = next(training_iterator)
            except StopIteration:
                training_iterator = iter(training_loader)
                data, target = next(training_iterator)

            if cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            output_size = list(output.size())
            seq_len = output_size[1] if len(output_size) > 1 else output_size[0]
            loss = loss_fun(output.squeeze(), target[:, -seq_len:].squeeze())
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

    np_values = np.vstack((lr_values, losses)).T
    np.savetxt(os.path.join(output_path, 'lr_finder.csv'), np_values, delimiter=',')


def create_eval_loader(trials):
    valid_dataset = ConcatDataset(
        [ECoGDatast(X, y, window=cfg.TRAINING.CROP_LEN, stride=cfg.EVAL.INPUT_STRIDE,
                    x2y_ratio=cfg.TRAINING.INPUT_SAMPLING_RATE / cfg.TRAINING.OUTPUT_SAMPLING_RATE,
                    input_shape='ct', dummy_idx=cfg.TRAINING.DUMMY_IDX) for (X, y) in trials])

    valid_loader = DataLoader(valid_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, drop_last=False,
                              pin_memory=True, num_workers=4)
    return valid_loader


def create_training_loader(trials):
    training_dataset = ConcatDataset(
        [ECoGDatast(X, y, window=cfg.TRAINING.CROP_LEN, stride=cfg.TRAINING.INPUT_STRIDE,
                    x2y_ratio=cfg.TRAINING.INPUT_SAMPLING_RATE / cfg.TRAINING.OUTPUT_SAMPLING_RATE,
                    input_shape='ct', dummy_idx=cfg.TRAINING.DUMMY_IDX) for (X, y) in trials])

    batch_sampler = BalancedBatchSampler(training_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True)

    training_loader = DataLoader(training_dataset, batch_sampler=batch_sampler, pin_memory=True, num_workers=4)
    return training_loader


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


def create_model(in_channels, num_classes, cuda=True):
    num_relaxed_samples = 681
    if cfg.TRAINING.MODEL.lower() == 'rnn':
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

    elif cfg.TRAINING.MODEL.lower() == 'deep4':
        model = Deep4Net(in_chans=in_channels, n_classes=num_classes, input_time_length=cfg.TRAINING.CROP_LEN,
                         final_conv_length=2, stride_before_pool=True).create_network()

        # remove softmax
        new_model = nn.Sequential()
        for name, module in model.named_children():
            if name == 'softmax':
                # continue
                break
            new_model.add_module(name, module)

        # remove empty final dimension
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

    elif cfg.TRAINING.MODEL.lower() == 'shallow':
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
    elif cfg.TRAINING.MODEL.lower() == 'hybrid':
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

    elif cfg.TRAINING.MODEL.lower() == 'tcn':
        raise NotImplementedError
    else:
        assert False, f"Unknown Model {cfg.TRAINING.MODEL}"
    optimizer = optim.Adam(model.parameters(), lr=cfg.OPTIMIZATION.BASE_LR, weight_decay=cfg.OPTIMIZATION.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.TRAINING.MAX_EPOCHS)
    if cuda:
        model.cuda()

    metric = lambda targets, predictions: np.corrcoef(targets, predictions)[0, 1]
    return model, optimizer, scheduler, loss_fun, metric


def evaluate_one_epoch(model, data_loader, loss_fun, metric, keep_state=False, writer=None, epoch=0, cuda=False):
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

            output = output.squeeze()
            output_size = list(output.size())
            seq_len = output_size[1] if len(output_size) > 1 else output_size[0]
            batch_size = output_size[0] if len(output_size) > 1 else 1
            num_classes = output_size[2] if len(output_size) > 2 else 1
            if batch_size == 1:
                output = output.unsqueeze(0)
            # loss value
            avg_loss += loss_fun(output, target[:, -seq_len:]).cpu().numpy().squeeze()
            # compute the correlation coff. for each seq. in batch
            target = target_cpu[:, -cfg.TRAINING.OUTPUT_SEQ_LEN:].numpy().squeeze()
            output = output[:, -cfg.TRAINING.OUTPUT_SEQ_LEN:].cpu().numpy().squeeze()
            if num_classes > 1:
                target = target.transpose(0, 2)
                output = output.transpose(0, 2)

            target = target.reshape(num_classes, -1)
            output = output.reshape(num_classes, -1)

            preds.append(output)
            targets.append(target)

    # divide by the number of mini-batches
    avg_loss /= len(data_loader)
    targets = np.hstack(targets).squeeze()
    preds = np.hstack(preds).squeeze()
    num_classes = targets.shape[0] if len(targets.shape) > 1 else 1
    if num_classes > 1:
        avg_corr = dict()
        for class_idx in range(num_classes):
            avg_corr[f"Class{class_idx}"] = metric(targets[class_idx, ], preds[class_idx, ])
    else:
        avg_corr = metric(targets, preds)

    if writer is not None:
        writer.add_scalar('loss', avg_loss, epoch)
        if num_classes == 1:
            writer.add_scalar('corr', avg_corr, epoch)
        else:
            writer.add_scalars('corr', avg_corr, epoch)
    return avg_loss, avg_corr, preds


def train_one_epoch(model, data_loader, optimizer, loss_fun, keep_state=False, clip=0, cuda=False):
    model.train()
    for itr, (data, target_cpu) in enumerate(data_loader):
        # data, target = Variable(data.transpose(1, 0)), Variable(target_cpu.squeeze(0))
        data, target = Variable(data), Variable(target_cpu)
        if cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        # detach to stop back-propagation to older state
        if cfg.TRAINING.MODEL.lower() in ('hybrid', 'rnn') and keep_state:
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


def eval_model(model, loss_fun, metric, valid_trials, weights_path, cuda):
    valid_loader = create_eval_loader(valid_trials)
    assert os.path.exists(weights_path), 'weights_path does not exists'
    model.load_state_dict(torch.load(weights_path))
    valid_loss, valid_corr, preds = evaluate_one_epoch(model, valid_loader, loss_fun, metric, keep_state=False, cuda=cuda)

    if cfg.EVAL.SAVE_PREDICTIONS:
        preds_dir = os.path.join(os.path.dirname(weights_path), 'predictions')
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        np.savetxt(os.path.join(preds_dir, 'predictions.csv'), preds, delimiter=',')
    return valid_corr, valid_loss


def training_loop(model, optimizer, scheduler, loss_fun, metric, training_trials, training_writer,
                  valid_trials=[], valid_writer=None, weights_path=None, cuda=True):
    training_loader = create_training_loader(training_trials)
    training_eval_loader = create_training_loader(training_trials)
    if valid_trials:
        valid_loader = create_eval_loader(valid_trials)

    if cfg.EVAL.SAVE_PREDICTIONS:
        assert weights_path, 'weights_path is required with SAVE PREDICTIONS'
        preds_dir = os.path.join(os.path.dirname(weights_path), 'predictions')
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)

    weights_dir, ext = os.path.splitext(weights_path)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path))
        weights_path = weights_dir + '_best' + ext

    min_loss = float('inf')
    for epoch in range(cfg.TRAINING.MAX_EPOCHS+1):
        if epoch % cfg.TRAINING.EVAL_TRAIN_EVERY == 0 or epoch % cfg.TRAINING.EVAL_VALID_EVERY == 0:
            logger.info(f"===========epoch: {epoch}=============")
            logger.info(f"Started at: {datetime.datetime.now():%d %b: %H:%M}")

        # report init. error before training
        if epoch == 0:
            train_loss, train_corr, _ = evaluate_one_epoch(model, training_eval_loader, loss_fun, metric, keep_state=False,
                                                           writer=training_writer, epoch=epoch, cuda=cuda)
            logger.info(f'init. training loss value: {train_loss}')
            logger.info(f'init. training corr: {train_corr}')

            if valid_trials:
                valid_loss, valid_corr, preds = evaluate_one_epoch(model, valid_loader, loss_fun, metric, keep_state=False,
                                                                   writer=valid_writer, epoch=epoch, cuda=cuda)
                if cfg.EVAL.SAVE_PREDICTIONS:
                    np.savetxt(os.path.join(preds_dir, f'{epoch}.csv'), preds, delimiter=',')
                logger.info(f'init. valid loss: {valid_loss}')
                logger.info(f'init. valid corr: {valid_corr}')

            if type(train_corr) == dict:
                max_acc = dict(zip(list(train_corr.keys()), [-float('inf')] * len(train_corr)))
            else:
                max_acc = -float('inf')

            if not valid_trials:
                # training without validation set. dummy valid corr coeff. value
                valid_corr = max_acc

            continue

        if scheduler is not None:
            scheduler.step(epoch-1)

        train_one_epoch(model, training_loader, optimizer, loss_fun, keep_state=False, clip=0, cuda=cuda)

        if epoch % cfg.TRAINING.EVAL_TRAIN_EVERY == 0:
            if training_writer is not None and cfg.TRAINING.WEIGHT_STATS:
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    training_writer.add_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                    training_writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)

            train_loss, train_corr, preds = evaluate_one_epoch(model, training_eval_loader, loss_fun, metric, keep_state=False,
                                                               writer=training_writer, epoch=epoch, cuda=cuda)
            if cfg.EVAL.SAVE_PREDICTIONS:
                np.savetxt(os.path.join(preds_dir, f'training_preds{epoch}.csv'), preds, delimiter=',')
            logger.info(f'training loss: {train_loss}')
            logger.info(f'training corr: {train_corr}')

            #  training without validation set. report training loss and corr. coeff.
            if not valid_trials:
                if train_loss < min_loss:
                    logger.info(f'found new training loss: {train_loss}')
                    min_loss = train_loss
                    last_best = epoch

                if type(train_corr) == dict:
                    for task, corr in train_corr.items():
                        if corr > max_acc[task]:
                            max_acc[task] = corr

                elif train_corr > max_acc:
                    max_acc = train_corr

        if valid_trials and epoch % cfg.TRAINING.EVAL_VALID_EVERY == 0:
            valid_loss, valid_corr, preds = evaluate_one_epoch(model, valid_loader, loss_fun, metric, keep_state=False,
                                                               writer=valid_writer, epoch=epoch, cuda=cuda)
            if cfg.EVAL.SAVE_PREDICTIONS:
                np.savetxt(os.path.join(preds_dir, f'valid_preds{epoch}.csv'), preds, delimiter=',')

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
        if type(valid_corr) == dict:
            if np.any(np.isnan(list(train_corr.values()))) or np.any(np.isnan(list(valid_corr.values()))):
                logger.error('Training stalled')
                break
        else:
            if np.isnan(train_corr) or np.isnan(valid_corr):
                logger.error('Training stalled')
                break

    torch.save(model.state_dict(), weights_dir + '_final' + ext)

    # report best values
    logger.info(f'Best loss value: {min_loss}')
    logger.info(f'Best corr value: {max_acc}')

    return max_acc, min_loss
