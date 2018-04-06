import os
os.sys.path.insert(0, '/data/schirrmr/fattouhm/datasets/')
os.sys.path.insert(0, '..')
from utils.experiment_util import *
import torch.multiprocessing
from tensorboardX import SummaryWriter
from torch import optim
from torch.nn.functional import mse_loss
import csv

subject = 'FR3'
MAX_EPOCHS = 1000
EVAL_EVERY = 50
CUDA = True
EXPERIMENT_NAME = 'acc_vs_time'
RANDOM_SEED = 10418
np.random.seed(RANDOM_SEED)

log_dir = '/home/fattouhm/hybrid/pos'
log_dir = os.path.join(log_dir, EXPERIMENT_NAME, subject)


def evaluate_hist(model, data_loader, metric, keep_state=False, cuda=False):
    model.eval()
    # loop over the dataset
    for itr, (data, target_cpu) in enumerate(data_loader):
        data, target = Variable(data), Variable(target_cpu.squeeze(0))
        if cuda:
            data, target = data.cuda(), target.cuda()

        if keep_state:
            if model.rnn_type == 'lstm':
                for h in hidden:
                    h.detach_()
            else:
                hidden.detach_()
            output, hidden = model(data, hidden)  # NxTxnum_classes
        else:
            output = model(data)

        output_size = list(output.squeeze().size())
        batch_size, seq_len = output_size[0], output_size[1]
        # compute the correlation coff. for each seq. in batch
        target = target_cpu[:, -seq_len:].numpy()
        output = output.data.cpu().numpy()
        if itr == 0:
            hist_corr = np.arctanh(metric(target[:, :], output[:, :]))
        else:
            hist_corr += np.arctanh(metric(target[:, :], output[:, :]))

    return np.arctanh(hist_corr/len(data_loader)).tolist()


def main():
    new_srate_x = 250
    new_srate_y = 250
    bin_size = int(new_srate_x / 2)  # 0.5 sec
    x2y_ratio = new_srate_x / new_srate_y
    n_sec = 1
    # window size
    window = n_sec * new_srate_x
    # we want no overlapping data
    batch_size = 16
    hidden_size = 64
    num_layers = 3
    channel_config = None
    time_config = None
    weights_dropout = ['weight_ih_l%d' % layer for layer in range(num_layers)]
    weights_dropout.extend(['weight_hh_l%d' % layer for layer in range(num_layers)])
    rnn_config = {'rnn_type': 'gru',
                  'hidden_size': hidden_size,
                  'num_layers': num_layers,
                  'dropout': 0.3,
                  'weights_dropout': weights_dropout,
                  'batch_norm': False}
    fc_config = {'num_classes': 1,
                 'fc_size': [32, 10],
                 'batch_norm': [False, False],
                 'dropout': [0.5, .3],
                 'activations': [nn.Hardtanh(-1, 1, inplace=True)] * 2}
    initializer = torch.nn.init.xavier_normal
    torch.manual_seed(RANDOM_SEED)

    crop_len = 15.0
    # relax_window = 3 # seconds
    num_relaxed_samples = 681 # int(relax_window * new_srate_x)
    stride = crop_len * new_srate_x - num_relaxed_samples
    training_loader, valid_loader, in_channels = create_dataset(subject, new_srate_x, new_srate_y, window, stride,
                                                                crop_len, batch_size)

    learning_rate = 5e-3
    wd_const = 5e-6
    model = HybridModel(in_channels=in_channels, channel_conv_config=channel_config,
                        time_conv_config=time_config, rnn_config=rnn_config,
                        fc_config=fc_config, output_stride=int(x2y_ratio))
    if CUDA:
        model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd_const)
    loss_fun = mse_loss
    metric = CorrCoeff.corrcoeff
    hist_metric = CorrCoeff(bin_size=bin_size).hist_corrcoeff

    # scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

    training_writer = SummaryWriter(log_dir + 'train')
    valid_writer = SummaryWriter(log_dir + 'valid')
    # training_writer.add_text('Model parameters', str(HybridModel.get_meta(model)))
    training_writer.add_text('Initializer', str(initializer))
    training_writer.add_text('Description', 'Adam. fixed relax window. changing crop len')
    training_writer.add_text('Learning Rate', str(learning_rate))
    training_writer.add_text('Weight Decay', str(wd_const))
    training_writer.add_text('Crop Length[sec]', str(crop_len))
    training_writer.add_text('Input srate[Hz]', str(new_srate_x))
    training_writer.add_text('Output srate[Hz]', str(new_srate_y))
    training_writer.add_text('relaxed samples', str(num_relaxed_samples))
    training_writer.add_text('Input channels', str(in_channels))

    weights_path = log_dir + 'weights.pt'

    run_experiment(model, optimizer, loss_fun, metric, training_loader, training_writer, valid_loader, valid_writer,
                   weights_path, max_epochs=MAX_EPOCHS, eval_every=100, cuda=CUDA)

    training_hist = evaluate_hist(model, training_loader, hist_metric, cuda=CUDA)
    valid_hist = evaluate_hist(model, valid_loader, hist_metric, cuda=CUDA)
    with open(log_dir+'/train_hist.csv', 'w+') as f:
        writer = csv.writer(f)
        for val in training_hist:
            writer.writerow([val])

    with open(log_dir+'/valid_hist.csv', 'w+') as f:
        writer = csv.writer(f)
        for val in valid_hist:
            writer.writerow([val])


if __name__ == '__main__':
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    main()


