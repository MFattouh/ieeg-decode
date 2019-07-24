import logging
import os
import datetime
from torch.utils.data import DataLoader, ConcatDataset
import torch.multiprocessing
import h5py
import torch.nn as nn
from braindecode.models.util import to_dense_prediction_model
from braindecode.torch_ext.modules import Expression
from models.hybrid import HybridModel
from braindecode.models.deep4 import Deep4Net
from models.deep5net import Deep5Net
from braindecode.models.shallow_fbcsp import ShallowFBCSPNet as Shallow
from torch.nn.functional import mse_loss
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.mat73_to_pickle import recursive_dict
from resampy import resample
from utils.pytorch_util import *
from utils.config import cfg


logger = logging.getLogger('__name__')


def read_dataset(dataset_path, dataset_name, mha_only):
    datasets_list = []

    with h5py.File(dataset_path, 'r') as hf:
        if mha_only:
            # # first remove the non-valid channels
            header = recursive_dict(hf['H/channels'])
            valid_channels_idx = extract_valid_channels(header)
            num_valid_channels = len(valid_channels_idx)

            header_keys = dict([(key.lower().replace('_', '').replace('-', ''), key) for key in header.keys()])
            esm = header[header_keys['esm']]
            mha_channels = list(map(lambda x: 'arm motor' in x or 'hand motor' in x,
                                    [esm[i] for i in valid_channels_idx]))
            mha_channels_idx = [i for i, x in enumerate(mha_channels) if x]

        trials = [hf[obj_ref] for obj_ref in hf[dataset_name][0]]
        for idx, trial in enumerate(trials, 1):
            if trial['ieeg'].ndim != 2:
                logger.warning('Irregular trial shape {} encountered in trial {}. '
                               'Expected 2D array! This trial will be discarded'.format(trial['ieeg'].shape, idx))
                continue

            # read data
            X = trial['ieeg'][:]
            y = trial['traj'][:][:].T

            if mha_only:
                assert X.shape[0] == num_valid_channels
                X = X[mha_channels_idx, :]

            # resample if necessary
            srate = int(trial['srate'][:][:])
            # TODO: HARDCODED axis
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

            # TODO: HARDCODED axis
            datasets_list.append((X, y))

    return datasets_list, X.shape[0]

def extract_valid_channels(header):
    header_keys = dict([(key.lower().replace('_', '').replace('-', ''), key) for key in header.keys()])
    signal_type = [stype.replace('_', '').replace('-', '').lower() for stype in header[header_keys['signaltype']].tolist()]
    ecog_grid_channels_idx = [stype.find('ecoggrid') != -1 for stype in signal_type]
    ecog_strip_channels_idx = [stype.find('ecogstrip') != -1 for stype in signal_type]
    seeg_channels_idx = [stype.find('seeg') != -1 for stype in signal_type]
    ieeg_idx = np.bitwise_or(np.bitwise_or(ecog_grid_channels_idx, ecog_strip_channels_idx), seeg_channels_idx)
    if np.all(ieeg_idx==False):
        raise KeyError('No ECoG-Grid, ECoG-Strip or SEEG were electrods found!')
    if 'seizureonset' in header_keys:
        soz = header[header_keys['seizureonset']]
    else:
        soz = np.zeros((ieeg_idx.shape[-1], 1)).squeeze()
    valid_channels = soz == 0
    if 'rejected' in header_keys:
        rejected = header[header_keys['rejected']]
        not_rejected = np.array([np.all(rejected[idx] == 0) for idx in range(len(rejected))])
        valid_channels = np.bitwise_and(valid_channels, not_rejected)

    if 'interictaloften' in header_keys:
        rejected = header[header_keys['interictaloften']]
        not_rejected = np.array([np.all(rejected[idx] == 0) for idx in range(len(rejected))])
        valid_channels = np.bitwise_and(valid_channels, not_rejected)

    valid_channels = np.bitwise_and(ieeg_idx, valid_channels)
    valid_channels_idx = [i for i, x in enumerate(valid_channels) if x]
    return valid_channels_idx 

def read_multi_datasets(input_datasets_path, dataset_name, mha_only):
    datasets_list, in_channels = read_dataset(input_datasets_path[0], dataset_name, mha_only)

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
                del training_iterator
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
        [ECoGDataset(X, y, window=cfg.TRAINING.CROP_LEN, stride=cfg.EVAL.INPUT_STRIDE,
                    x2y_ratio=cfg.TRAINING.INPUT_SAMPLING_RATE / cfg.TRAINING.OUTPUT_SAMPLING_RATE,
                    input_shape='ct', dummy_idx=cfg.TRAINING.DUMMY_IDX) for (X, y) in trials])

    valid_loader = DataLoader(valid_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, drop_last=False,
                              pin_memory=True, num_workers=4)
    return valid_loader


def create_training_loader(trials):
    training_dataset = ConcatDataset(
        [ECoGDataset(X, y, window=cfg.TRAINING.CROP_LEN, stride=cfg.TRAINING.INPUT_STRIDE,
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
    def squeeze_out(x):
        assert x.size()[1] == num_classes and x.size()[3] == 1
        return x.squeeze(3).transpose(1, 2)

    num_relaxed_samples = 681
    if cfg.TRAINING.MODEL.lower() == 'rnn':
        model = HybridModel(in_channels=in_channels, output_stride=int(cfg.HYBRID.OUTPUT_STRIDE))

        # if cfg.HYBRID.OUTPUT_STRIDE > 1:
        #     loss_fun = mse_loss
        #     metric = CorrCoeff().corrcoeff
        # else:
        #     weights = make_weights(cfg.TRAINING.CROP_LEN, num_relaxed_samples, dtype='step')
        #     weights_tensor = torch.from_numpy(weights)
        #     if cuda:
        #         weights_tensor = weights_tensor.cuda()
        #     loss_fun = WeightedMSE(weights_tensor)
        #     metric = CorrCoeff(weights).weighted_corrcoef

    
    elif 'deep4' in cfg.TRAINING.MODEL.lower():
        if 'wide' in cfg.TRAINING.MODEL.lower():
            pool_length=4
            pool_stride=4
        elif 'narrow' in cfg.TRAINING.MODEL.lower():
            pool_length=2
            pool_stride=2
        else:
            pool_length=3
            pool_stride=3
            
        model = Deep4Net(in_chans=in_channels, n_classes=num_classes, input_time_length=cfg.TRAINING.CROP_LEN,
                         pool_time_length=pool_length, pool_time_stride=pool_stride,
                         final_conv_length=2, stride_before_pool=True).create_network()

        # remove softmax
        new_model = nn.Sequential()
        for name, module in model.named_children():
            if name == 'softmax':
                # continue
                break
            new_model.add_module(name, module)

        # remove empty final dimension and permute output shape
        new_model.add_module('squeeze', Expression(squeeze_out))
        # if num_classes > 1:
        #     def transpose_class_time(x):
        #         return x.transpose(2, 1)
        #
        #     new_model.add_module('trans', Expression(transpose_class_time))

        model = new_model

        to_dense_prediction_model(model)

        loss_fun = mse_loss
        metric = CorrCoeff().corrcoeff

    elif cfg.TRAINING.MODEL.lower() == 'deep5':
        #  pool_time_length=3
        #  pool_time_stride=3
        model = Deep5Net(in_chans=in_channels, n_classes=num_classes, input_time_length=cfg.TRAINING.CROP_LEN,
                         final_conv_length=2, stride_before_pool=True).create_network()

        # remove softmax
        new_model = nn.Sequential()
        for name, module in model.named_children():
            if name == 'softmax':
                # continue
                break
            new_model.add_module(name, module)

        # remove empty final dimension and permute output shape
        new_model.add_module('squeeze', Expression(squeeze_out))
        # if num_classes > 1:
        #     def transpose_class_time(x):
        #         return x.transpose(2, 1)
        #
        #     new_model.add_module('trans', Expression(transpose_class_time))

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

        # remove empty final dimension and permute output shape
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

    model.eval()
    
    metric = lambda targets, predictions: np.corrcoef(targets, predictions)[0, 1]
    loss_fun = mse_loss
    logger.info(model)
    return model, optimizer, scheduler, loss_fun, metric


def evaluate_one_epoch(model, data_loader, loss_fun, metric, keep_state=False, writer=None, epoch=0, cuda=False):
    model.eval()
    avg_loss = 0
    targets = []
    preds = []
    with torch.no_grad():
        # loop over the dataset
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
            batch_size, num_classes = output_size[0], output_size[-1]
            # loss value
            avg_loss += loss_fun(output[:, -cfg.TRAINING.OUTPUT_SEQ_LEN:], target[:, -cfg.TRAINING.OUTPUT_SEQ_LEN:]).item()
            # compute the correlation coff. for each seq. in batch
            target = target_cpu[:, -cfg.TRAINING.OUTPUT_SEQ_LEN:, ].numpy()
            output = output[:, -cfg.TRAINING.OUTPUT_SEQ_LEN:, ].cpu().numpy()

            output = output.transpose((2, 1, 0)).reshape(num_classes, -1)
            target = target.transpose((2, 1, 0)).reshape(num_classes, -1)

            preds.append(output)
            targets.append(target)

    # divide by the number of mini-batches
    avg_loss /= len(data_loader)
    targets = np.hstack(targets)
    preds = np.hstack(preds)
    num_classes = targets.shape[0]
    if num_classes > 1:
        avg_corr = dict()
        for class_idx in range(num_classes):
            avg_corr[f"Class{class_idx}"] = metric(targets[class_idx, ].squeeze(), preds[class_idx, ].squeeze())
    else:
        avg_corr = metric(targets.squeeze(), preds.squeeze())

    if writer is not None:
        writer.add_scalar('loss', avg_loss, epoch)
        if num_classes == 1:
            writer.add_scalar('corr', avg_corr, epoch)
        else:
            writer.add_scalars('corr', avg_corr, epoch)
    return avg_loss, avg_corr, targets, preds


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

        loss = loss_fun(output[:, -cfg.TRAINING.OUTPUT_SEQ_LEN:], target[:, -cfg.TRAINING.OUTPUT_SEQ_LEN:])
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_value_(model.parameters(), clip)

        optimizer.step()


def eval_model(model, loss_fun, metric, valid_trials, weights_path, cuda):
    valid_loader = create_eval_loader(valid_trials)
    assert os.path.exists(weights_path), 'weights_path does not exists'
    model.load_state_dict(torch.load(weights_path))
    valid_loss, valid_corr, targets, preds = evaluate_one_epoch(model, valid_loader, loss_fun, metric, keep_state=False, cuda=cuda)

    if cfg.EVAL.SAVE_PREDICTIONS:
        preds_dir = os.path.join(os.path.dirname(weights_path.replace('TRAIN', 'EVAL')), 'predictions')
        if not os.path.exists(preds_dir):
            os.makedirs(preds_dir)
        np.savetxt(os.path.join(preds_dir, 'predictions.csv'), preds, delimiter=',')
        np.savetxt(os.path.join(preds_dir, 'targets.csv'), targets, delimiter=',')
    return valid_corr, valid_loss


def training_loop(model, optimizer, scheduler, loss_fun, metric, training_trials, training_writer,
                  valid_trials=[], valid_writer=None, weights_path=None, cuda=True):
    training_loader = create_training_loader(training_trials)
    logger.info(f'Number of training mini-batches: {len(training_loader)}')
    training_eval_loader = create_training_loader(training_trials)

    if valid_trials:
        valid_loader = create_eval_loader(valid_trials)
        logger.info(f'Number of validation mini-batches: {len(valid_loader)}')

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
            train_loss, train_corr, _, _ = evaluate_one_epoch(model, training_eval_loader, loss_fun, metric, keep_state=False,
                                                           writer=training_writer, epoch=epoch, cuda=cuda)
            logger.info(f'init. training loss value: {train_loss}')
            logger.info(f'init. training corr: {train_corr}')

            if valid_trials:
                valid_loss, valid_corr, targets, preds = evaluate_one_epoch(model, valid_loader, loss_fun, metric, keep_state=False,
                                                                   writer=valid_writer, epoch=epoch, cuda=cuda)
                if cfg.EVAL.SAVE_PREDICTIONS:
                    np.savetxt(os.path.join(preds_dir, f'preds_{epoch}.csv'), preds, delimiter=',')
                    np.savetxt(os.path.join(preds_dir, f'targets_{epoch}.csv'), targets, delimiter=',')
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

        train_one_epoch(model, training_loader, optimizer, loss_fun, keep_state=cfg.TRAINING.KEEP_STATE,
                        clip=cfg.TRAINING.GRAD_CLIP, cuda=cuda)

        if epoch % cfg.TRAINING.EVAL_TRAIN_EVERY == 0:
            if training_writer is not None and cfg.TRAINING.WEIGHT_STATS:
                for tag, param in model.named_parameters():
                    if not param.requires_grad:
                        continue
                    tag = tag.replace('.', '/')
                    weights = param.data.cpu()
                    training_writer.add_histogram(tag, weights.numpy(), epoch + 1)
                    training_writer.add_scalar('weights/' + tag + '/min', weights.min().item(), epoch + 1)
                    training_writer.add_scalar('weights/' + tag + '/max', weights.max().item(), epoch + 1)
                    training_writer.add_scalar('weights/' + tag + '/norm', weights.norm().item(), epoch + 1)
                    training_writer.add_scalar('weights/' + tag + '/std', weights.std().item(), epoch + 1)
                    grads = param.grad.data.cpu()
                    training_writer.add_histogram('grads/' + tag, grads.numpy(), epoch + 1)
                    training_writer.add_scalar('grads/' + tag + '/min', grads.min().item(), epoch + 1)
                    training_writer.add_scalar('grads/' + tag + '/max', grads.max().item(), epoch + 1)
                    training_writer.add_scalar('grads/' + tag + '/norm', grads.norm().item(), epoch + 1)
                    training_writer.add_scalar('grads/' + tag + '/std', grads.std().item(), epoch + 1)

            train_loss, train_corr, targets, preds = evaluate_one_epoch(model, training_eval_loader, loss_fun, metric, keep_state=False,
                                                               writer=training_writer, epoch=epoch, cuda=cuda)
            if cfg.EVAL.SAVE_PREDICTIONS:
                np.savetxt(os.path.join(preds_dir, f'training_preds{epoch}.csv'), preds, delimiter=',')
                np.savetxt(os.path.join(preds_dir, f'training_targets{epoch}.csv'), targets, delimiter=',')
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
            valid_loss, valid_corr, targets, preds = evaluate_one_epoch(model, valid_loader, loss_fun, metric, keep_state=False,
                                                                        writer=valid_writer, epoch=epoch, cuda=cuda)
            if cfg.EVAL.SAVE_PREDICTIONS:
                np.savetxt(os.path.join(preds_dir, f'valid_preds{epoch}.csv'), preds, delimiter=',')
                np.savetxt(os.path.join(preds_dir, f'valid_targets{epoch}.csv'), targets, delimiter=',')

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

    if valid_trials is not None:
        return valid_corr, valid_loss
    else:
        return train_corr, train_loss
