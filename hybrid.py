from collections import OrderedDict

import torch
import torch.nn as nn
from torch.autograd import Variable
from bnlstm import LSTM as BNLSTM
from bnlstm import BNLSTMCell
import numpy as np

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
    return Expression(lambda x: x.permute(1, 2, 0))  # NxHxT


def _rnn_to_fc_transpose():
    # x is expected to be TxNxH
    # x = x..contiguous()
    return Expression(lambda x: x.transpose(1, 0))  # NxTxH


# the next two functions perform exactly the same operation. just for the sake of calrity
def _bn_to_fc_transpose():
    # x is expected to be NxHxT
    return Expression(lambda x: x.transpose(2, 1))  # NxTxH


def _fc_to_bn_transpose():
    # x is expected to be NxTxH
    return Expression(lambda x: x.transpose(2, 1))  # NxHxT


def _expand_last():
    # x is expected to be NxHxT
    return Expression(lambda x: x.unsqueeze(3))  # NxHxTx1


def _drop_last():
    # x is expected to be NxHxTx1
    return Expression(lambda x: x.squeeze(3))  # NxHxT


class HybridModel(nn.Module):
    def __init__(self, rnn_type='lstm', in_channels=64, channel_filters=[], time_filters=[], time_kernels=[],
                 rnn_hidden_size=768, rnn_layers=5, max_length=None, fc_size=[10], num_classes=1,
                 output_stride=0, dropout=0.1, batch_norm=False, initializer=None):

        super(HybridModel, self).__init__()
        assert rnn_type.lower() in supported_rnns, 'unknown recurrent type'+rnn_type
        if batch_norm:
            assert rnn_type.lower() == 'lstm', 'Recurrent Batch Normalization is currently not supported for '+rnn_type
            assert max_length > 0, 'a valid max length required to with batch normalization'

        # model metadata
        self._version = '0.0.1'
        self._hidden_size = rnn_hidden_size
        self._hidden_layers = rnn_layers
        self.rnn_type = supported_rnns[rnn_type.lower()]
        self._linear_hidden_size = fc_size
        self._output_stride = output_stride
        self._in_1d_filters = channel_filters
        self._in_2d_filters = time_filters
        self._in_2d_kernels = time_kernels
        self._bn = batch_norm
        self._dropout = dropout

        # channels conv. layers. convolutions are done over all channels in the first layer
        # and over all output filters in later layers.
        if channel_filters:
            channels_conv = list()
            channels_conv.append(('conv0', nn.Conv2d(1, channel_filters[0], kernel_size=(in_channels, 1),
                                                     bias=not batch_norm)))
            if initializer is not None:
                initializer(channels_conv[-1][1].weight)
            channels_conv.append(('tanh0', nn.Hardtanh(0, 20, inplace=True)))
            if batch_norm:
                channels_conv.append(('BN0', nn.BatchNorm2d(channel_filters[0])))
            if len(channel_filters) > 1:
                for layer, num_filters in enumerate(channel_filters[1:], 1):
                    channels_conv.append(('trans%d1' % layer, _transpose_C_to_W()))
                    channels_conv.append(('conv%d' % layer, nn.Conv2d(1, num_filters,
                                                                      kernel_size=(channel_filters[layer-1], 1),
                                                                      bias=not batch_norm)))
                    if initializer is not None:
                        initializer(channels_conv[-1][1].weight)
                    channels_conv.append(('tanh%d' % layer, nn.Hardtanh(0, 20, inplace=True)))
                    if batch_norm:
                        channels_conv.append(('BN%d' % layer, nn.BatchNorm2d(num_filters)))
                    channels_conv.append(('trans%d2' % layer, _transpose_C_to_W()))
            else:
                channels_conv.append(('trans0', _transpose_C_to_W()))

            # no need for batch normalization if output fed to the rnn directly since will be done on rnn input
            if batch_norm and not time_filters:
                channels_conv = channels_conv[:-1]
            self.channel_conv = nn.Sequential(OrderedDict(channels_conv))
        else:
            self.channel_conv = None

        # convolution layers over time dimension only
        rnn_input_size = channel_filters[-1] if channel_filters else in_channels
        if time_filters:
            time_conv = list()
            time_conv.append(('conv0', nn.Conv2d(1, time_filters[0], kernel_size=(1, time_kernels[0]),
                                                 bias=not batch_norm)))
            if initializer is not None:
                initializer(time_conv[-1][1].weight)
            if batch_norm:
                time_conv.append(('BN0', nn.BatchNorm2d(time_filters[0])))
            time_conv.append(('tanh0', nn.Hardtanh(0, 20, inplace=True)))
            for x in range(1, len(time_filters)):
                time_conv.append(('conv%d' % x, nn.Conv2d(time_filters[x - 1], time_filters[x],
                                                          kernel_size=(1, time_kernels[x]), bias=not batch_norm)))
                if initializer is not None:
                    initializer(time_conv[-1][1].weight)
                if batch_norm:
                    time_conv.append(('BN%d' % x, nn.BatchNorm2d(time_filters[x])))
                time_conv.append(('tanh%d' % x, nn.Hardtanh(0, 20, inplace=True)))

            rnn_input_size *= time_filters[-1]
            # remove last batch normalization layer since the bnlstm will do it
            if batch_norm:
                time_conv = time_conv[:-1]
            self.time_conv = nn.Sequential(OrderedDict(time_conv))
        else:
            self.time_conv = None

        if batch_norm:
            self.rnns = BNLSTM(cell_class=BNLSTMCell, input_size=rnn_input_size, hidden_size=rnn_hidden_size,
                               dropout=dropout, num_layers=rnn_layers, max_length=max_length)
        else:
            self.rnns = self.rnn_type(input_size=rnn_input_size, hidden_size=rnn_hidden_size, dropout=dropout,
                                      num_layers=rnn_layers, bidirectional=False, bias=False)

        if fc_size:
            fc_out = list()
            if batch_norm or dropout > 0:
                fc_out.append(('trans01', _rnn_to_bn_transpose()))
                if batch_norm:
                    fc_out.append(('BN0', nn.BatchNorm1d(rnn_hidden_size)))
                if dropout > 0:
                    fc_out.append(('expand0', _expand_last()))
                    fc_out.append(('dropout0', nn.Dropout2d(dropout)))
                    fc_out.append(('squeeze0', _drop_last()))
                fc_out.append(('trans02', _bn_to_fc_transpose()))
            else:
                fc_out.append(('trans0', _rnn_to_fc_transpose()))

            fc_out.append(('linear0', nn.Linear(rnn_hidden_size, fc_size[0], bias=not batch_norm)))
            if initializer is not None:
                initializer(fc_out[-1][1].weight)
            fc_out.append(('tanh0', nn.Hardtanh(-20, 20)))
            if len(fc_size) > 1:
                for layer, num_units in enumerate(fc_size[1:], 1):
                    if batch_norm or dropout > 0:
                        fc_out.append(('trans%d1' % layer, _fc_to_bn_transpose()))
                        if batch_norm:
                            fc_out.append(('BN%d' % layer, nn.BatchNorm1d(rnn_hidden_size)))
                        if dropout > 0:
                            fc_out.append(('expand%d' % layer, _expand_last()))
                            fc_out.append(('dropout%d' % layer, nn.Dropout2d(dropout)))
                            fc_out.append(('squeeze%d' % layer, _drop_last()))
                        fc_out.append(('trans%d2' % layer, _bn_to_fc_transpose()))

                    fc_out.append(('linear%d' % layer, nn.Linear(fc_size[layer - 1], num_units, bias=not batch_norm)))
                    if initializer is not None:
                        initializer(fc_out[-1][1].weight)
                    fc_out.append(('tanh%d' % layer, nn.Hardtanh(-20, 20)))

            fc_out.append(('linear', nn.Linear(fc_size[-1], num_classes, bias=not batch_norm)))
            if initializer is not None:
                initializer(fc_out[-1][1].weight)
            fully_connected = nn.Sequential(OrderedDict(fc_out))

        else:
            fc_out = list()
            if batch_norm or dropout > 0:
                fc_out.append('trans', _rnn_to_bn_transpose())
                if batch_norm:
                    fc_out.append('BN', nn.BatchNorm1d(rnn_hidden_size))
                if dropout > 0:
                    fc_out.append('expand', _expand_last())
                    fc_out.append(('dropout', nn.Dropout2d(dropout)))
                    fc_out.append('squeeze', _drop_last())
                fc_out.append('detrans', _bn_to_fc_transpose())
            else:
                fc_out.append('trans', _rnn_to_fc_transpose())
            fc_out.append(('output', nn.Linear(rnn_hidden_size, num_classes, bias=not batch_norm)))
            if initializer is not None:
                initializer(fc_out[-1][1].weight)
            fully_connected = nn.Sequential(OrderedDict(fc_out))

        self.fc_bn = nn.BatchNorm1d(rnn_hidden_size) if batch_norm else None
        self.fc = fully_connected

    def init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self._hidden_layers, batch_size, self._hidden_size))
        if self.rnn_type == nn.LSTM:
            return hidden, hidden
        else:
            return hidden

    def forward(self, x, hidden):
        # x is expected to be Nx1xCxT
        if self.channel_conv is not None:
            x = self.channel_conv(x)   # NxCx1xT
        if self.time_conv is not None:
            x = self.time_conv(x)      # NxC2xC1xT2

        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension
        x = x.transpose(1, 2).transpose(0, 1).contiguous()   # TxNxC
        x, hidden = self.rnns(x, hidden)  # TxNxH

        if self._output_stride > 1:
            x = x[self._output_stride-1::self._output_stride]

        print(x.size())
        x = self.fc(x)  # NxTxH

        return x, hidden  # N x T x nClasses

    @staticmethod
    def get_param_size(model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params

    @staticmethod
    def get_meta(model):
        meta = {
            "version":model._version,
            "channel_filters":model._in_1d_filters,
            "time_conv_filters":model._in_2d_filters,
            "time_conv_kernels":model._in_2d_kernels,
            "rnn_units":model._hidden_size,
            "rnn_layers":model._hidden_layers,
            "rnn_type": supported_rnns_inv[model.rnn_type],
            "fully_connected": model._linear_hidden_size,
            "output_stride":model._output_stride,
            "batch normalization": model._bn,
            "dropout": model._dropout
        }
        return meta


def train(model, data_loader, optimizer, loss_fun, keep_state=False, clip=0, cuda=False):
    model.train()
    for itr, (data, target_cpu) in enumerate(data_loader):
        data, target = Variable(data.transpose(1, 0)), Variable(target_cpu.squeeze(0))
        if cuda:
            data, target = data.cuda(), target.cuda()

        if itr == 0 or not keep_state:
            batch_size = list(data.size())[0]
            hidden = model.init_hidden(batch_size)
            if cuda:
                if model.rnn_type == nn.LSTM:
                    hidden = tuple([h.cuda() for h in hidden])
                else:
                    hidden = hidden.cuda()

        # detach to stop back-propagation to older state
        elif keep_state:
            if model.rnn_type == nn.LSTM:
                for h in hidden:
                    h.detach_()
            else:
                hidden.detach_()

        optimizer.zero_grad()
        output, hidden = model(data, hidden)  # NxTxnum_classes
        loss = loss_fun(output.squeeze(), target[:, -output.size()[1]:].squeeze())
        loss.backward()
        if clip > 0:
            torch.nn.utils.clip_grad_norm(model.parameters(), clip)

        optimizer.step()


def evaluate(model, data_loader, loss_fun, keep_state=False, writer=None, epoch=0, cuda=False):
    model.eval()
    # loop over the dataset
    avg_loss = 0
    for itr, (data, target_cpu) in enumerate(data_loader):
        data, target = Variable(data.transpose(1, 0)), Variable(target_cpu.squeeze(0))
        if cuda:
            data, target = data.cuda(), target.cuda()

        if itr == 0 or not keep_state:
            batch_size = list(data.size())[0]
            hidden = model.init_hidden(batch_size)
            if cuda:
                if model.rnn_type == nn.LSTM:
                    hidden = tuple([h.cuda() for h in hidden])
                else:
                    hidden = hidden.cuda()

        elif keep_state:
            if model.rnn_type == nn.LSTM:
                for h in hidden:
                    h.detach_()
            else:
                hidden.detach_()
        output, hidden = model(data, hidden)  # NxTxnum_classes

        size = list(output.size())
        batch_size, seq_len, num_classes = size[0], size[1], size[2]
        avg_loss += loss_fun(output.squeeze(), target[:, -seq_len:].squeeze()).data.cpu().numpy().squeeze()
        # compute the correlation coff. for each seq. in batch
        target = target_cpu.squeeze(0)[:, -seq_len:].numpy()
        output = output.data.cpu().numpy()
        if itr == 0:
            cum_corr = np.zeros((batch_size, num_classes))
            valid_corr = np.zeros((batch_size, num_classes))
        for batch_idx in range(batch_size):
            for class_idx in range(num_classes):
                # compute correlation, apply fisher's transform
                corr = np.arctanh(np.corrcoef(target[batch_idx, :, class_idx].squeeze(),
                                              output[batch_idx, :, class_idx].squeeze())[0, 1])
                if not np.isnan(corr):
                    cum_corr[batch_idx, class_idx] += corr
                    valid_corr[batch_idx, class_idx] += 1
    if writer is not None:
        writer.add_scalar('loss', avg_loss / len(data_loader.dataset), epoch)
    # average the correlations across over iterations apply inverse fisher's transform find mean over batch
    if num_classes == 1:
        avg_corr = np.tanh(cum_corr.squeeze() / valid_corr.squeeze()).mean()
        if writer is not None:
            writer.add_scalar('corr', avg_corr, epoch)
    else:
        avg_corr = dict()
        for i in range(num_classes):
            avg_corr['Class%d' % i] = np.tanh(cum_corr[:, i] / valid_corr[:, i]).mean()
        if writer is not None:
            writer.add_scalars('corr', avg_corr, epoch)
    return avg_loss, avg_corr


