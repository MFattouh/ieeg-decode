from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from bnlstm import LSTM as BNLSTM
from bnlstm import BNLSTMCell
import numpy as np
from weight_drop import WeightDrop


supported_rnns = {
    'lstm': nn.LSTM,
    'rnn': nn.RNN,
    'gru': nn.GRU
}
supported_rnns_inv = dict((v, k) for k, v in supported_rnns.items())


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


def _transpose_C_to_W():
    # x is expected to be Nx1xCxT
    return Expression(lambda x: x.transpose(2, 1))  # NxCx1xT


def _rnn_to_bn_transpose():
    # x is expected to be TxNxH
    return Expression(lambda x: x.permute(1, 2, 0).contiguous())  # NxHxT


def _rnn_to_fc_transpose():
    # x is expected to be TxNxH
    # x = x..contiguous()
    return Expression(lambda x: x.transpose(1, 0)).contiguous()  # NxTxH


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


class HybridModel(nn.Module):
    def __init__(self, in_channels=0, channel_conv_config=None, time_conv_config=None, rnn_config=None, fc_config=None,
                 output_stride=0):

        super(HybridModel, self).__init__()
        self._output_stride = output_stride
        self._hidden_layers = rnn_config['num_layers']
        self._hidden_size = rnn_config['hidden_size']
        self.rnn_type = rnn_config['rnn_type'].lower()
        # channels conv. layers. convolutions are done over all channels in the first layer
        # and over all output filters in later layers.
        if channel_conv_config is not None:
            self.channel_conv = self.make_channels_conv(in_channels, **channel_conv_config)
        else:
            self.channel_conv = None
        # convolution layers over time dimension only
        if time_conv_config is not None:
            self.time_conv = self.make_time_convs(**time_conv_config)
        else:
            self.time_conv = None

        rnn_input_size = channel_conv_config['channel_filters'][-1] if self.channel_conv is not None else in_channels
        rnn_input_size = rnn_input_size * time_conv_config['time_filters'][-1] if self.time_conv is not None\
            else rnn_input_size

        self.rnns = self.make_rnn(rnn_input_size, **rnn_config)

        self.fc = self.make_fc(rnn_config['hidden_size'], **fc_config)

    def make_rnn(self, input_size=0, rnn_type='lstm', batch_norm=False, dropout=0, weights_dropout=[], max_length=0,
                 hidden_size=10,
                 num_layers=1):
        assert rnn_type.lower() in supported_rnns, 'unknown recurrent type'+rnn_type
        if batch_norm:
            assert rnn_type.lower() == 'lstm', 'Recurrent Batch Normalization is currently not supported for '+rnn_type
            assert max_length > 0, 'a valid max length required to with batch normalization'
        if batch_norm:
            rnns = BNLSTM(cell_class=BNLSTMCell, input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, max_length=max_length)
        else:
            rnns = supported_rnns[rnn_type](input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, bidirectional=False, bias=True)

            if dropout > 0 and weights_dropout:
                rnns = WeightDrop(rnns, weights=weights_dropout, dropout=dropout)

        return rnns

    def make_fc(self, input_size=0, num_classes=1, batch_norm=[], dropout=[], fc_size=[], initializer=None,
                activations=[]):
        if fc_size:
            fc_out = list()
            if batch_norm[0] or dropout[0] > 0:
                fc_out.append(('trans01', _rnn_to_bn_transpose()))
                if batch_norm[0]:
                    fc_out.append(('BN0', nn.BatchNorm1d(input_size)))
                if dropout[0] > 0:
                    fc_out.append(('expand0', _expand_last()))
                    fc_out.append(('dropout0', nn.Dropout2d(dropout[0])))
                    fc_out.append(('squeeze0', _drop_last()))
                fc_out.append(('trans02', _bn_to_fc_transpose()))
            else:
                fc_out.append(('trans0', _rnn_to_fc_transpose()))

            fc_out.append(('linear0', nn.Linear(input_size, fc_size[0], bias=not batch_norm[0])))
            if initializer is not None:
                initializer(fc_out[-1][1].weight)
            fc_out.append(('activation0', activations[0]))
            if len(fc_size) > 1:
                for layer, (num_units, activation, bn, do) in \
                        enumerate(zip(fc_size[1:], activations[1:], batch_norm[1:], dropout[1:]), 1):
                    if bn or do > 0:
                        fc_out.append(('trans%d1' % layer, _fc_to_bn_transpose()))
                        if bn:
                            fc_out.append(('BN%d' % layer, nn.BatchNorm1d(fc_size[layer-1])))
                        if do > 0:
                            fc_out.append(('expand%d' % layer, _expand_last()))
                            fc_out.append(('dropout%d' % layer, nn.Dropout2d(do)))
                            fc_out.append(('squeeze%d' % layer, _drop_last()))
                        fc_out.append(('trans%d2' % layer, _bn_to_fc_transpose()))

                    fc_out.append(('linear%d' % layer, nn.Linear(fc_size[layer - 1], num_units, bias=not bn)))
                    if initializer is not None:
                        initializer(fc_out[-1][1].weight)
                    fc_out.append(('activation%d' % layer, activation))

            fc_out.append(('linear', nn.Linear(fc_size[-1], num_classes, bias=False)))
            if initializer is not None:
                initializer(fc_out[-1][1].weight)
            fully_connected = nn.Sequential(OrderedDict(fc_out))

        else:
            fc_out = list()
            if batch_norm[0] or dropout[0] > 0:
                fc_out.append(('trans', _rnn_to_bn_transpose()))
                if batch_norm[0]:
                    fc_out.append(('BN', nn.BatchNorm1d(input_size)))
                if dropout[0] > 0:
                    fc_out.append(('expand', _expand_last()))
                    fc_out.append(('dropout', nn.Dropout2d(dropout[0])))
                    fc_out.append(('squeeze', _drop_last()))
                fc_out.append(('detrans', _bn_to_fc_transpose()))
            else:
                fc_out.append(('trans', _rnn_to_fc_transpose()))
            fc_out.append(('output', nn.Linear(input_size, num_classes, bias=not batch_norm[0])))
            if initializer is not None:
                initializer(fc_out[-1][1].weight)
            fully_connected = nn.Sequential(OrderedDict(fc_out))

        return fully_connected

    def make_time_convs(self, batch_norm=[], initializer=None, time_filters=[], time_kernels=[], activations=[]):
        assert len(time_filters) == len(time_kernels) == len(activations) == len(batch_norm)
        time_conv = list()
        time_conv.append(('conv0', nn.Conv2d(1, time_filters[0], kernel_size=(1, time_kernels[0]),
                                             bias=not batch_norm[0])))
        if initializer is not None:
            initializer(time_conv[-1][1].weight)
        time_conv.append(('activation0', activations[0]))
        if batch_norm[0]:
            time_conv.append(('BN0', nn.BatchNorm2d(time_filters[0])))
        for layer, (num_filters, kernel, activation, bn) in \
                enumerate(zip(time_filters[1:], time_kernels[1:], activations[1:], batch_norm[1:]), 1):
            time_conv.append(('conv%d' % layer, nn.Conv2d(time_filters[layer - 1], num_filters,
                                                          kernel_size=(1, kernel), bias=not bn)))
            if initializer is not None:
                initializer(time_conv[-1][1].weight)
            time_conv.append(('activation%d' % layer, activation))
            if bn:
                time_conv.append(('BN%d' % layer, nn.BatchNorm2d(num_filters)))

        time_conv = nn.Sequential(OrderedDict(time_conv))
        return time_conv

    def make_channels_conv(self, in_channels=0, batch_norm=[], channel_filters=[], initializer=None, activations=[]):
        assert len(channel_filters) == len(activations) == len(batch_norm)
        channels_conv = list()
        channels_conv.append(('conv0', nn.Conv2d(1, channel_filters[0], kernel_size=(in_channels, 1),
                                                 bias=not batch_norm[0])))
        if initializer is not None:
            initializer(channels_conv[-1][1].weight)
        channels_conv.append(('activation0', activations[0]))
        if batch_norm[0]:
            channels_conv.append(('BN0', nn.BatchNorm2d(channel_filters[0])))
        if len(channel_filters) > 1:
            for layer, (num_filters, activation, bn) in \
                    enumerate(zip(channel_filters[1:], activations[1:], batch_norm[1:]), 1):
                channels_conv.append(('trans%d1' % layer, _transpose_C_to_W()))
                channels_conv.append(('conv%d' % layer, nn.Conv2d(1, num_filters,
                                                                  kernel_size=(channel_filters[layer - 1], 1),
                                                                  bias=not batch_norm)))
                if initializer is not None:
                    initializer(channels_conv[-1][1].weight)
                channels_conv.append(('activation%d' % layer, activation))
                if bn:
                    channels_conv.append(('BN%d' % layer, nn.BatchNorm2d(num_filters)))
                channels_conv.append(('trans%d2' % layer, _transpose_C_to_W()))
        else:
            channels_conv.append(('trans0', _transpose_C_to_W()))

        channel_conv = nn.Sequential(OrderedDict(channels_conv))

        return channel_conv

    def forward(self, x, hidden=None):
        # x is expected to be Nx1xCxT
        if self.channel_conv is not None:
            x = self.channel_conv(x)   # NxCx1xT
        if self.time_conv is not None:
            x = self.time_conv(x)      # NxC2xC1xT2

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()   # TxNxC
        x, out_hidden = self.rnns(x, hidden)  # TxNxH

        if self._output_stride > 1:
            x = x[self._output_stride-1::self._output_stride]

        x = self.fc(x).squeeze()  # NxTxH

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
        data, target = Variable(data), Variable(target_cpu.squeeze(0))
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
        seq_len = output.size()[1]
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


