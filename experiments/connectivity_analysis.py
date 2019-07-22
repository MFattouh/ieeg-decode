import os
import yaml
import click
os.sys.path.insert(0, '..')
from utils.experiment_util import read_dataset, create_model, create_eval_loader, evaluate_one_epoch
from glob import glob
from utils.config import cfg, merge_configs
import random
from braindecode.visualization.perturbation import amp_perturbation_additive
from braindecode.util import corr
import numpy as np
import torch
import torch as th

CUDA = True

STAT = 'mean'
# STAT = 'median'

stat_fn = np.mean if 'mean' in STAT else np.median

def eval_dropouts(mod):
        module_name =  mod.__class__.__name__
        if 'Dropout' in module_name or 'BatchNorm' in module_name: mod.training = False
        for module in mod.children(): eval_dropouts(module)


@click.command(name='connectivity-analysis')
@click.argument('command', type=click.Choice(['io_conn', 'freq_conn', 'pert_conn', 'spectrogram']))
@click.argument('configs', type=click.Path(), default=os.path.curdir)
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('subject', type=str)
@click.option('--log_dir', '-l', type=click.Path(), default=os.path.curdir)
@click.option('--task', '-t', type=click.Choice(['xpos', 'xvel', 'abspos', 'absvel', 'xacc', 'absacc']), default='xpos',
              help='Task to decode. acceptable are:\n'
                   '* xpos for position decoding.\n'
                   '* xvel for velocity decoding.\n'
                   '* abspos for absolute position decoding.\n'
                   '* absvel for absolute velocity decoding.\n'
                   '* xacc for acceleration decoding.\n'
                   '* absacc for absolute acceleration decoding.\n'
                   'default is pos')
def main(command, configs, dataset_dir, subject, log_dir, task):
    with open(configs, 'r') as f:
        merge_configs(yaml.load(f))

    # FIXME: PERTURBATION based connectivity analysis is buggy at the moment 
    # set the random state
    np.random.seed(cfg.TRAINING.RANDOM_SEED)
    torch.manual_seed(cfg.TRAINING.RANDOM_SEED)
    random.seed(cfg.TRAINING.RANDOM_SEED)

    train_path = os.path.join(log_dir, task.upper(), 'TRAIN', subject, cfg.TRAINING.MODEL.upper())
    assert os.path.exists(train_path), f"Can't detect training folder: {train_path}"

    log_dir = os.path.join(log_dir, task.upper(), 'TRAIN', subject, cfg.TRAINING.MODEL.upper())

    if task == 'xpos':
        datasets = glob(dataset_dir + '*' + subject+'_*_xpos.mat')
    elif task == 'xvel':
        datasets = glob(dataset_dir + '*' + subject+'_*_xvel.mat')
    elif task == 'abspos':
        datasets = glob(dataset_dir + '*' + subject+'_*_absPos.mat')
    elif task == 'absvel':
        datasets = glob(dataset_dir + '*' + subject+'_*_absVel.mat')
    elif task == 'xacc':
        datasets = glob(dataset_dir + 'ALL_*' + subject+'_*_xacc.mat')
    elif task == 'absacc':
        datasets = glob(dataset_dir + '*' + subject+'_*_absAcc.mat')
    else:
        raise KeyError

    assert len(datasets) > 0, 'no datasets for subject %s found!' % subject

    for dataset_path in datasets:
        rec_day_name = os.path.basename(dataset_path).split('.')[0].split('_')
        rec_name = '_'.join([rec_day_name[1], rec_day_name[3]])
        dataset_name = cfg.EVAL.DATASET

        trials, in_channels = read_dataset(dataset_path, dataset_name, mha_only=cfg.TRAINING.MHA_CHANNELS_ONLY)
        data_loader = create_eval_loader(trials)
        num_classes = 1

        weights_path = os.path.join(train_path, rec_name, 'weights_final.pt')
        assert os.path.exists(weights_path), 'No weights are detected for this recording!'
        model, _, _, _, _ = create_model(in_channels, num_classes, CUDA)
        if CUDA:
            model.cuda()

        model.load_state_dict(torch.load(weights_path))
        # TODO: check if the weights are loaded properly. check the corr of validation set for example.

        # cudnn RNN backward can only be called in training mode 
        model.train()
        eval_dropouts(model)

        # mean_autocorr_x = np.zeros(cfg.TRAINING.CROP_LEN, dtype=np.float32)
        # mean_autocorr_y = np.zeros(cfg.TRAINING.CROP_LEN, dtype=np.float32)
        grads = []
        if 'io' in command:
            output_name = 'input'
            #  grads = np.zeros(cfg.TRAINING.CROP_LEN, dtype=np.float32)

        elif 'freq' in command:
            output_name = 'amps'
            #  grads = np.zeros(cfg.TRAINING.CROP_LEN // 2 + 1, dtype=np.float32)

        elif 'pert' in command:
            output_name = 'pert'
            rng = np.random.RandomState(cfg.TRAINING.RANDOM_SEED)
            perturbation_list = []
            output_diff_list = []

        elif 'spectrogram' in command:
            output_name = 'spectrogram'
            window_size = 250
#             overlap = 125
            overlap = 245
            unique = window_size - overlap
            han = torch.tensor(np.hanning(window_size), requires_grad=False, dtype=th.float)

            # we have now 3 dimensions
            # 1. batch (average over samples)
            # 2. fft amps
            # 3. time (4000 / 250) (concatanate grads w.r.t. amps along the x axis)
            num_freq_bins = (window_size / 2) + 1 if window_size % 2 == 0 else (window_size+1)/2
            time_bins = list(range(0, cfg.TRAINING.CROP_LEN - window_size, unique))

#             grads = np.zeros((len(time_bins), int(num_freq_bins)), dtype=np.float32)
            grads = []

        else:
            raise RuntimeError('command not understood!')

        for X, Y in data_loader:
            # autocorr_x = np.zeros(cfg.TRAINING.CROP_LEN, dtype=np.float32)
            # for c in range(X.shape[2]):
            #     autocorr_x += np.correlate(X[0, 0, c, :], X[0, 0, c, :], 'full')[cfg.TRAINING.CROP_LEN-1:]
            # mean_autocorr_x += autocorr_x / X.shape[2]
            # mean_autocorr_y += np.correlate(Y.squeeze(), Y.squeeze(), 'full')[cfg.TRAINING.CROP_LEN-1:]

            if 'freq' in command:
                # grads w.r.t. frequency amp
                amps_th, iffted = fb_fft(X, cfg.TRAINING.CROP_LEN)
                model.zero_grad()
                output = model(iffted)
                output[0, -1, 0].backward()
#                 grads += torch.mean(np.abs(amps_th.grad.squeeze()), dim=0).cpu().numpy()
                grads.append(torch.mean(torch.abs(amps_th.grad.squeeze()), dim=0).cpu().numpy())

            elif 'spectrogram' in command:
                # time-resolved grads w.r.t frequency amp
                window_grads = []
                for i in time_bins:
                    window = X[:, :, :, i:i + window_size] * han
                    amps_th, iffted = fb_fft(window, window_size)
                    rest_after = torch.tensor(X[:, :, :, i + window_size:], requires_grad=False, dtype=th.float,
                                              device='cuda')
                    if i > 0:
                        rest_before = torch.tensor(X[:, :, :, :i], requires_grad=False, dtype=th.float, device='cuda')
                        input_tensor = torch.cat((rest_before, iffted, rest_after), dim=3)
                    else:
                        input_tensor = torch.cat((iffted, rest_after), dim=3)

                    model.zero_grad()
                    output = model(input_tensor)
                    output[0, -1, 0].backward()
                    window_grads.append(torch.mean(np.abs(amps_th.grad.squeeze()), dim=0).cpu().numpy()[np.newaxis])

#                 grads += np.vstack(window_grads)
                grads.append(np.vstack(window_grads))

            elif 'io' in command:
                # grads w.r.t. input
                input_tensor = torch.tensor(X, requires_grad=True, dtype=th.float, device='cuda')
                model.zero_grad()
                output = model(input_tensor)
                output[0, -1, 0].backward()
                # channels dimension
                grads.append(torch.mean(torch.abs(input_tensor.grad.squeeze()), dim=0).cpu().numpy())

            elif 'pert' in command:
                # grads w.r.t. input
                # find the model output given the input before perturbation
                with torch.no_grad():
                    input_tensor = torch.tensor(X, dtype=th.float, device='cuda')
                    model.zero_grad()
                    output_before_pert = model(input_tensor).detach().cpu().numpy()[0, -1, 0]
                    # perturb the input signal and find the output
                    for _ in range(1000):
                        amps_th, iffted, pert_values = fb_fft_with_perturbation(
                            X, amp_perturbation_additive, cfg.TRAINING.CROP_LEN, rng=rng)
                        output_after_pert = model(iffted).detach().cpu().numpy()[0, -1, 0]
                        # append perturbations and output diff from all pert. iterations and mini-batches
                        output_diff_list.append(output_after_pert - output_before_pert)
                        perturbation_list.append(np.expand_dims(pert_values.squeeze(), 2))
            else:
                raise RuntimeError('command not understood!')

        if 'pert' not in command:
#             grads /= len(data_loader) * cfg.TRAINING.BATCH_SIZE
            grads_array = np.array(grads)
            np.savetxt(f'grads_{cfg.TRAINING.MODEL}_{task}_{subject}_{rec_name}.csv', grads_array, delimiter=',')
            grads = stat_fn(np.array(grads), axis=0)
        else:
            output_diff = np.array(output_diff_list)
            perturbations = np.dstack(perturbation_list)
            grads = np.mean(np.abs(np.array([[corr(output_diff.reshape(1, -1), pert_fb.reshape(1, -1)) for pert_fb in perturbation]
                                       for perturbation in perturbations])), axis=0).squeeze()

        # mean_autocorr_x /= len(data_loader) * cfg.TRAINING.BATCH_SIZE
        # mean_autocorr_y /= len(data_loader) * cfg.TRAINING.BATCH_SIZE

        np.savetxt(os.path.join(log_dir, rec_name, f"connectivity_{output_name}_{STAT}.csv"), grads, delimiter=',')
        # np.savetxt(os.path.join(log_dir, rec_name, 'autocorr_x.csv'), mean_autocorr_x, delimiter=',')
        # np.savetxt(os.path.join(log_dir, rec_name, 'autocorr_y.csv'), mean_autocorr_y, delimiter=',')

    print('Done!')


def fb_fft(X, window_size):
    # take an np signal and perform fft and then ifft
    amps, phases = compute_fft(X)
    amps_th = th.tensor(amps, requires_grad=True, dtype=th.float, device='cuda')
    phases_th = th.tensor(phases, requires_grad=True, dtype=th.float, device='cuda')
    iffted = compute_ifft(amps_th, phases_th, window_size)
    return amps_th, iffted


def fb_fft_with_perturbation(X, pert_fn, window_size, rng):
    # take an np signal and perform fft and then ifft
    amps, phases = compute_fft(X)
    amps, phases, pert_vals = pert_fn(amps, phases, rng)

    amps_th = th.tensor(amps, requires_grad=True, dtype=th.float, device='cuda')
    phases_th = th.tensor(phases, requires_grad=True, dtype=th.float, device='cuda')
    iffted = compute_ifft(amps_th, phases_th, window_size)
    return amps_th, iffted, pert_vals


def compute_ifft(amps_th, phases_th, window_size):
    fft_coefs = amps_th.unsqueeze(-1) * th.stack((th.cos(phases_th), th.sin(phases_th)), dim=-1)
    iffted = th.irfft(fft_coefs, signal_ndim=1, signal_sizes=(window_size,)).requires_grad_()
    return iffted


def compute_fft(X):
    ffted = np.fft.rfft(X, axis=3)
    amps = np.abs(ffted)
    phases = np.angle(ffted)
    return amps, phases


if __name__ == '__main__':
    main()

