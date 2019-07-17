import os
import os.path as osp
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
import json
import yaml
from sklearn.model_selection import train_test_split
import h5py
import click
import random
from glob import glob
os.sys.path.insert(0, '..')
from models.hybrid import HybridModel as RNNs
from utils.config import cfg, merge_configs
from utils.pytorch_util import ECoGDataset
from utils.experiment_util import read_dataset

WINDOW_SIZE = 3000

def create_dataset_loader(X, y):
    return DataLoader(ECoGDataset(X, y, window=WINDOW_SIZE, stride=1, x2y_ratio=1,
                                 input_shape='ct', dummy_idx='f'),
                      batch_size=32, shuffle=False, drop_last=False,
                      pin_memory=True, num_workers=4)

@click.command(name='save_predictions')
@click.argument('configs', type=click.Path(), default=os.path.curdir)
@click.argument('dataset_dir', type=click.Path(exists=True))
@click.argument('subject', type=str)
@click.option('--log_dir', '-l', type=click.Path(), default=os.path.curdir)
@click.option('--task', '-t', type=click.Choice(['xpos', 'xvel', 'abspos', 'absvel', 'multi']), default='xpos',
              help='Task to decode. acceptable are:\n'
                   '* xpos for position decoding.\n'
                   '* xvel for velocity decoding.\n'
                   '* abspos for absolute position decoding.\n'
                   '* absvel for absolute velocity decoding.\n'
                   'default is pos')
def main(configs, dataset_dir, subject, log_dir, task):
    with open(configs, 'r') as f:
        merge_configs(yaml.load(f))

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
    else:
        raise KeyError

    assert len(datasets) > 0, 'no datasets for subject %s found!' % subject

    for dataset_path in datasets:
        year, sub, day = osp.basename(dataset_path).split('_')[1:-1]
        trials, in_channels = read_dataset(dataset_path, 'D', False)
        trials_idx = list(range(len(trials)))
        _, valid_split = train_test_split(trials_idx, test_size=0.2, shuffle=False,
                                        random_state=cfg.TRAINING.RANDOM_SEED)
        valid_trials = [trials[valid_idx] for valid_idx in valid_split]
        rec_name = '_'.join([year, day])

        weights_path = os.path.join(train_path, rec_name, 'weights_final.pt')
        assert os.path.exists(weights_path), 'No weights are detected for this recording!'
        model = RNNs(in_channels=in_channels, output_stride=int(cfg.HYBRID.OUTPUT_STRIDE))
        model.load_state_dict(torch.load(weights_path))
        model.cuda()
        model.eval()

        # TODO: hardcoded path
        with h5py.File(f'/home/fattouhm/notebooks/{year}_{sub}_{day}_{task}_predictions_{WINDOW_SIZE}.h5', 'w') as hf:
            for trial_idx, trial in enumerate(valid_trials):
                inputs, targets = trial
                time_steps = inputs.shape[1]
                offsets = time_steps - WINDOW_SIZE + 1
                predictions = np.empty((time_steps, offsets), dtype=np.float32)
                predictions[:] = np.nan
                dataset = create_dataset_loader(inputs, targets)
                for batch_offset, (X, _) in enumerate(dataset):
                    with torch.no_grad(): 
                        X = X.cuda()
                        output = model(X)
                        output = output.detach().squeeze(-1).cpu().numpy()
                
                    for sample_idx, sample_pred in enumerate(output):
                        offset_idx = batch_offset * 32 + sample_idx
                        predictions[offset_idx:offset_idx+WINDOW_SIZE, offset_idx] = sample_pred
                hf.create_dataset(f'trial{trial_idx:0>2d}', data=predictions)
    
    print('Done!')

if __name__ == '__main__':
    main()
