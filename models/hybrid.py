from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from models.bnlstm import LSTM as BNLSTM
from models.bnlstm import BNLSTMCell
import numpy as np
from models.weight_drop import WeightDrop
from utils.config import cfg

supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}

supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())

supported_activations = {
    'tanh': nn.Hardtanh(-1, 1, inplace=True)
}

supported_init = {

}

# from https://github.com/robintibor/braindecode/blob/master/braindecode/torch_ext/modules.py
class Expression(torch.nn.Module):
    """
    Compute given expression on forward pass.

    Parameters
    ----------
    expression_fn: function
        Should accept variable number of objects of type
        `torch.autograd.Variable` to compute its output.
    """

    def __init__(self, expression_fn):
        super(Expression, self).__init__()
        self.expression_fn = expression_fn

    def forward(self, *x):
        return self.expression_fn(*x)

    def __repr__(self):
        if (hasattr(self.expression_fn, 'func') and
                hasattr(self.expression_fn, 'kwargs')):
            expression_str = "{:s} {:s}".format(
                self.expression_fn.func.__name__,
                str(self.expression_fn.kwargs))
        else:
            expression_str = self.expression_fn.__name__
        return (self.__class__.__name__ + '(' +
                'expression=' + str(expression_str) + ')')


def _log_nonlinearity():
    return Expression(lambda x: torch.log(x+1))


def _transpose_C_to_W():
    # x is expected to be Nx1xCxT
    return Expression(lambda x: x.transpose(2, 1))  # NxCx1xT


def _rnn_to_bn_transpose():
    # x is expected to be TxNxH
    return Expression(lambda x: x.permute(1, 2, 0).contiguous())  # NxHxT


def _rnn_to_fc_transpose():
    # x is expected to be TxNxH
    # x = x..contiguous()
    return Expression(lambda x: x.transpose(1, 0).contiguous())  # NxTxH


# the next two functions perform exactly the same operation. just for the sake of calrity
def _bn_to_fc_transpose():
    # x is expected to be NxHxT
    return Expression(lambda x: x.transpose(2, 1).contiguous())  # NxTxH


def _fc_to_bn_transpose():
    # x is expected to be NxTxH
    return Expression(lambda x: x.transpose(2, 1).contiguous())  # NxHxT


def _expand_last():
    # x is expected to be NxHxT
    return Expression(lambda x: x.unsqueeze(3))  # NxHxTx1


def _drop_last():
    # x is expected to be NxHxTx1
    return Expression(lambda x: x.squeeze(3))  # NxHxT


def make_rnn(input_size=0, rnn_type='lstm', normalization=False, dropout=0, weights_dropout=[], max_length=0, hidden_size=10,
             num_layers=1):
    assert rnn_type.lower() in supported_rnns, 'unknown recurrent type'+rnn_type
    if normalization == 'None':
        rnns = supported_rnns[rnn_type](input_size=input_size, hidden_size=hidden_size,
                                        num_layers=num_layers, bidirectional=False, bias=True)
    elif normalization == 'batch_norm':
        assert rnn_type.lower() == 'lstm', 'Recurrent Batch Normalization is currently not supported for '+rnn_type
        assert max_length > 0, 'a valid max length required to with batch normalization'
        rnns = BNLSTM(cell_class=BNLSTMCell, input_size=input_size, hidden_size=hidden_size,
                      num_layers=num_layers, max_length=max_length)
    else:
        raise NotImplementedError

    if dropout > 0 and weights_dropout:
        rnns = WeightDrop(rnns, weights=weights_dropout, dropout=dropout)

    return rnns


def make_fc(input_size=0, num_classes=1, batch_norm=False, dropout=[], fc_size=[], initializer=None,
            activations=None):
    assert activations in supported_activations, f'{activations} is not supported'
    activations = supported_activations[activations]
    if initializer is not None:
        assert initializer in supported_init, f'{initializer} is not supported'
        initializer = supported_init[initializer]
    if fc_size:
        fc_out = list()
        if batch_norm or dropout[0] > 0:
            fc_out.append(('trans01', _rnn_to_bn_transpose()))
            if batch_norm:
                fc_out.append(('BN0', nn.BatchNorm1d(input_size)))
            if dropout[0] > 0:
                fc_out.append(('expand0', _expand_last()))
                fc_out.append(('dropout0', nn.Dropout2d(dropout[0])))
                fc_out.append(('squeeze0', _drop_last()))
            fc_out.append(('trans02', _bn_to_fc_transpose()))
        else:
            fc_out.append(('trans0', _rnn_to_fc_transpose()))

        fc_out.append(('linear0', nn.Linear(input_size, fc_size[0], bias=not batch_norm)))
        if initializer is not None:
            initializer(fc_out[-1][1].weight)
        fc_out.append(('activation0', activations))
        if len(fc_size) > 1:
            for layer, (num_units, do) in \
                    enumerate(zip(fc_size[1:], dropout[1:]), 1):
                if batch_norm or do > 0:
                    fc_out.append(('trans%d1' % layer, _fc_to_bn_transpose()))
                    if batch_norm:
                        fc_out.append(('BN%d' % layer, nn.BatchNorm1d(fc_size[layer-1])))
                    if do > 0:
                        fc_out.append(('expand%d' % layer, _expand_last()))
                        fc_out.append(('dropout%d' % layer, nn.Dropout2d(do)))
                        fc_out.append(('squeeze%d' % layer, _drop_last()))
                    fc_out.append(('trans%d2' % layer, _bn_to_fc_transpose()))

                fc_out.append(('linear%d' % layer, nn.Linear(fc_size[layer - 1], num_units, bias=not batch_norm)))
                if initializer is not None:
                    initializer(fc_out[-1][1].weight)
                fc_out.append(('activation%d' % layer, activations))

        fc_out.append(('linear', nn.Linear(fc_size[-1], num_classes, bias=False)))
        if initializer is not None:
            initializer(fc_out[-1][1].weight)
        fully_connected = nn.Sequential(OrderedDict(fc_out))

    else:
        fc_out = list()
        if batch_norm or dropout[0] > 0:
            fc_out.append(('trans', _rnn_to_bn_transpose()))
            if batch_norm:
                fc_out.append(('BN', nn.BatchNorm1d(input_size)))
            if dropout[0] > 0:
                fc_out.append(('expand', _expand_last()))
                fc_out.append(('dropout', nn.Dropout2d(dropout[0])))
                fc_out.append(('squeeze', _drop_last()))
            fc_out.append(('detrans', _bn_to_fc_transpose()))
        else:
            fc_out.append(('trans', _rnn_to_fc_transpose()))
        fc_out.append(('output', nn.Linear(input_size, num_classes, bias=not batch_norm)))
        if initializer is not None:
            initializer(fc_out[-1][1].weight)
        fully_connected = nn.Sequential(OrderedDict(fc_out))

    return fully_connected


def make_temporal_convs(batch_norm=None, initializer=None, num_filters=None, kernel_size=None, activations=None,
                        dilations=None):
    assert num_filters is not None
    assert kernel_size is not None
    assert len(num_filters) == len(kernel_size)
    assert activations in supported_activations, f'{activations} is not supported'
    activations = supported_activations[activations]
    if initializer is not None:
        assert initializer in supported_init, f'{initializer} is not supported'
        initializer = supported_init[initializer]

    if dilations is not None:
        assert len(num_filters) == len(dilations)
    else:
        dilations = [1] * len(num_filters)

    temporal_conv = list()
    temporal_conv.append(('conv0', nn.Conv2d(1, num_filters[0], kernel_size=(1, kernel_size[0]), dilation=dilations[0],
                                             bias=not batch_norm)))
    if initializer is not None:
        initializer(temporal_conv[-1][1].weight)

    if activations is not None:
        temporal_conv.append(('activation0', activations))

    if batch_norm:
        temporal_conv.append(('BN0', nn.BatchNorm2d(num_filters[0])))
    for layer, (layer_filters, kernel, dilation) in \
            enumerate(zip(num_filters[1:], kernel_size[1:], dilations[1:]), 1):
        temporal_conv.append(('conv%d' % layer, nn.Conv2d(num_filters[layer - 1], layer_filters, dilation=dilation,
                                                          kernel_size=(1, kernel), bias=not batch_norm)))
        if initializer is not None:
            initializer(temporal_conv[-1][1].weight)
        if activations is not None:
            temporal_conv.append(('activation%d' % layer, activations))
        if batch_norm:
            temporal_conv.append(('BN%d' % layer, nn.BatchNorm2d(layer_filters)))

    temporal_conv = nn.Sequential(OrderedDict(temporal_conv))
    return temporal_conv


def make_spatial_conv(in_channels, batch_norm=False, num_filters=None, initializer=None, activations=None):
    assert num_filters is not None
    assert activations in supported_activations, f'{activations} is not supported'
    activations = supported_activations[activations]
    if initializer is not None:
        assert initializer in supported_init, f'{initializer} is not supported'
        initializer = supported_init[initializer]

    channels_conv = list()
    channels_conv.append(('conv0', nn.Conv2d(1, num_filters[0], kernel_size=(in_channels, 1),
                                             bias=not batch_norm)))
    if initializer is not None:
        initializer(channels_conv[-1][1].weight)
    if activations is not None:
        channels_conv.append(('activation0', activations))
    if batch_norm:
        channels_conv.append(('BN0', nn.BatchNorm2d(num_filters[0])))
    if len(num_filters) > 1:
        for layer, layer_filters in enumerate(num_filters[1:], 1):
            channels_conv.append(('trans%d1' % layer, _transpose_C_to_W()))
            channels_conv.append(('conv%d' % layer, nn.Conv2d(1, layer_filters,
                                                              kernel_size=(num_filters[layer - 1], 1),
                                                              bias=not batch_norm)))
            if initializer is not None:
                initializer(channels_conv[-1][1].weight)
            if activations is not None:
                channels_conv.append(('activation%d' % layer, activations))
            if batch_norm:
                channels_conv.append(('BN%d' % layer, nn.BatchNorm2d(layer_filters)))
            channels_conv.append(('trans%d2' % layer, _transpose_C_to_W()))
    else:
        channels_conv.append(('trans0', _transpose_C_to_W()))

    channel_conv = nn.Sequential(OrderedDict(channels_conv))

    return channel_conv


def make_l2pooling(window=40, stride=1):
    return nn.Sequential(OrderedDict([
        ('l2pool', torch.nn.LPPool2d(2, (1, window), stride=stride)),
        ('activation', _log_nonlinearity())
    ]))


class HybridModel(nn.Module):
    def __init__(self, in_channels=0, output_stride=0):

        super(HybridModel, self).__init__()
        self._output_stride = output_stride
        # channels conv. layers. convolutions are done over all channels in the first layer
        # and over all output filters in later layers.
        if cfg.HYBRID.SPATIAL_CONVS.ENABLED:
            channel_conv_config = dict(zip(map(lambda k: k.lower(), cfg.HYBRID.SPATIAL_CONVS.keys()), cfg.HYBRID.SPATIAL_CONVS.values()))
            channel_conv_config.pop('enabled')
            self.channel_conv = make_spatial_conv(in_channels, **channel_conv_config)
            rnn_input_size = cfg.HYBRID.SPATIAL_CONVS.num_filters[-1]
        else:
            self.channel_conv = None
            rnn_input_size = in_channels
        # convolution layers over time dimension only
        if cfg.HYBRID.TEMPORAL_CONVS.ENABLED:
            temporal_conv_config = dict(zip(map(lambda k: k.lower(), cfg.HYBRID.TEMPORAL_CONVS.keys()), cfg.HYBRID.TEMPORAL_CONVS.values()))
            temporal_conv_config.pop('ENABLED')
            self.time_conv = make_temporal_convs(**temporal_conv_config)
            rnn_input_size *= cfg.HYBRID.TEMPORAL_CONVS.num_filters[-1]
        else:
            self.time_conv = None

        if cfg.HYBRID.L2POOLING.ENABLED:
            l2pooling_config = dict(zip(map(lambda k: k.lower(), cfg.HYBRID.L2POOLING.keys()), cfg.HYBRID.L2POOLING.values()))
            l2pooling_config.pop('ENABLED')
            self.l2pooling = self.make_l2pooling(**l2pooling_config)
        else:
            self.l2pooling = None

        rnns_configs = dict(zip(map(lambda k: k.lower(), cfg.HYBRID.RNN.keys()), cfg.HYBRID.RNN.values()))
        self.rnns = make_rnn(rnn_input_size, **rnns_configs)

        linear_configs = dict(zip(map(lambda k: k.lower(), cfg.HYBRID.LINEAR.keys()), cfg.HYBRID.LINEAR.values()))
        self.fc = make_fc(cfg.HYBRID.RNN.HIDDEN_SIZE, **linear_configs)

    def forward(self, x, hidden=None):
        # x is expected to be Nx1xCxT
        if self.channel_conv is not None:
            x = self.channel_conv(x)   # NxCx1xT

        if self.time_conv is not None:
            x = self.time_conv(x)      # NxC2xC1xT2

        if self.l2pooling is not None:
            x = self.l2pooling(x)      # NxC2xC1xT3

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()   # TxNxC
        x, out_hidden = self.rnns(x, hidden)  # TxNxH

        if self._output_stride > 1:
            x = x[self._output_stride-1::self._output_stride]

        x = self.fc(x)  # NxTxH

        if hidden is not None:
            return x, out_hidden  # N x T x nClasses
        else:
            return x

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    def decribe_model(self):
       pass


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
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()


def evaluate(model, data_loader, loss_fun, metric, keep_state=False, writer=None, epoch=0, cuda=False):
    model.eval()

    # loop over the dataset
    avg_loss = 0
    for itr, (data, target_cpu) in enumerate(data_loader):
        # data, target = Variable(data.transpose(1, 0)), Variable(target_cpu.squeeze(0))
        data, target = Variable(data), Variable(target_cpu)
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

        output_size = list(output.size())
        seq_len = output_size[1] if len(output_size) > 1 else output_size[0]
        batch_size = output_size[0] if len(output_size) > 1 else 1
        num_classes = output_size[2] if len(output_size) > 2 else 1
        # print(output_size)
        # print(seq_len)
        # print(batch_size)
        # print(num_classes)
        # exit()
        avg_loss += loss_fun(output.squeeze(), target[:, -seq_len:].squeeze()).data.cpu().numpy().squeeze()
        # compute the correlation coff. for each seq. in batch
        target = target_cpu[:, -seq_len:].numpy().squeeze()
        output = output.data.cpu().numpy().squeeze()
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

    if keep_state:
        avg_loss /= len(data_loader.dataset)
    else:
        avg_loss /= len(data_loader)

    if num_classes == 1:
        avg_corr = np.tanh(cum_corr.squeeze() / valid_corr.squeeze()).mean()
    else:
        avg_corr = dict()
        for i in range(num_classes):
            avg_corr['Class%d' % i] = np.tanh(cum_corr[i] / valid_corr[i]).mean()

    if writer is not None:
        writer.add_scalar('loss', avg_loss, epoch)
        # average the correlations across over iterations apply inverse fisher's transform find mean over batch
        if num_classes == 1:
                writer.add_scalar('corr', avg_corr, epoch)
        else:
                writer.add_scalars('corr', avg_corr, epoch)
    return avg_loss, avg_corr


