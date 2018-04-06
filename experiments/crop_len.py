from utils.experiment_util import *
import torch.multiprocessing
from tensorboardX import SummaryWriter
from torch import optim

os.sys.path.insert(0, '.')
os.sys.path.insert(0, '/data/schirrmr/fattouhm/datasets/')

subject = 'FR3'
FIRST_IDX = 27
NUM_EXPERIMENTS = 3
MAX_EPOCHS = 1000
EVAL_EVERY = 50
CUDA = True
EXPERIMENT_NAME = 'crop_len'
RANDOM_SEED = 10418
np.random.seed(RANDOM_SEED)

log_dir = '/home/fattouhm/hybrid/pos'
log_dir = os.path.join(log_dir, EXPERIMENT_NAME, subject)


def main():
    new_srate_x = 250
    new_srate_y = 250
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
    relax_window = 3 # seconds
    stride = (crop_len - relax_window) * new_srate_x
    training_loader, valid_loader, in_channels = create_dataset(subject, new_srate_x, new_srate_y, window, stride,
                                                                crop_len, batch_size)
    num_relaxed_samples = int(relax_window * new_srate_x)
    weights = make_weights(crop_len * new_srate_x, num_relaxed_samples)

    weights_tensor = torch.from_numpy(weights)

    for run in range(NUM_EXPERIMENTS):
        run = run + FIRST_IDX

        # this exponent will results in a value between [-4, -2] -> learning rates = [1e-2, 1e-2]
        # r = -2*np.random.rand(1) - 2
        # learning_rate = 10 ** r[0]
        learning_rate = 5e-3
        # best lr is between 1.1e-3 to 1.8e-3
        #  learning_rate = (7 * np.random.rand(1) + 1)*1e-4 + 1e-3
        #  learning_rate = learning_rate[0]
        # range of r is [-4, -6] -> wd = 10^[-4, .., -6]
        # r = -2*np.random.rand(1) - 4
        # wd_const = 10 ** r[0]
        wd_const = 5e-6
        model = HybridModel(in_channels=in_channels, channel_conv_config=channel_config,
                            time_conv_config=time_config, rnn_config=rnn_config,
                            fc_config=fc_config, output_stride=int(x2y_ratio))
        if CUDA:
            model.cuda()
            weights_tensor = weights_tensor.cuda()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd_const)
        loss_fun = WeightedMSE(weights_tensor)
        metric = CorrCoeff(weights).weighted_corrcoef

        # scheduler = StepLR(optimizer, step_size=100, gamma=0.9)

        log_dir = log_dir + 'run%d/' % run
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
        training_writer.add_text('relax window[sec]', str(relax_window))
        training_writer.add_text('Input channels', str(in_channels))
        # training_writer.add_text('learning rate decay const', str(lr_decay_const))

        weights_path = log_dir + 'weights.pt'

        run_experiment(model, optimizer, loss_fun, metric, training_loader, training_writer, valid_loader, valid_writer,
                       weights_path, max_epochs=MAX_EPOCHS, cuda=CUDA)

def evaluate_hist(model, data_loader, loss_fun, metric, keep_state=False, writer=None, epoch=0, cuda=False):
    model.eval()

    # loop over the dataset
    avg_loss = 0
    for itr, (data, target_cpu) in enumerate(data_loader):
        # data, target = Variable(data.transpose(1, 0)), Variable(target_cpu.squeeze(0))
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
        num_classes = output_size[2] if len(output_size) > 2 else 1
        avg_loss += loss_fun(output.squeeze(), target[:, -seq_len:].squeeze()).data.cpu().numpy().squeeze()
        # compute the correlation coff. for each seq. in batch
        target = target_cpu[:, -seq_len:].numpy()
        output = output.data.cpu().numpy()
        if itr == 0:
            cum_corr = np.zeros((num_classes, 1))
            valid_corr = np.zeros((num_classes, 1))
        if num_classes == 1:
            corr = np.arctanh(metric(target[:, :], output[:, :]))
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

    if keep_state:
        avg_loss /= len(data_loader.dataset)
    else:
        avg_loss /= len(data_loader)

    if num_classes == 1:
        avg_corr = np.tanh(cum_corr.squeeze() / valid_corr.squeeze()).mean()
    else:
        avg_corr = dict()
        for i in range(num_classes):
            avg_corr['Class%d' % i] = np.tanh(cum_corr[:, i] / valid_corr[:, i]).mean()


    if writer is not None:
        writer.add_scalar('loss', avg_loss, epoch)
        # average the correlations across over iterations apply inverse fisher's transform find mean over batch
        if num_classes == 1:
                writer.add_scalar('corr', avg_corr, epoch)
        else:
                writer.add_scalars('corr', avg_corr, epoch)
    return avg_loss, avg_corr


if __name__ == '__main__':
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    main()


