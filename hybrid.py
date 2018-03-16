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


class HybridModel(nn.Module):
    def __init__(self, rnn_type='lstm', in_channels=64, channel_filters=[], time_filters=[], time_kernels=[],
                 rnn_hidden_size=768, rnn_layers=5, max_length=None, fc_size=[10], num_classes=1,
                 output_stride=0, dropout=0.1, batch_norm=False):

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

        num_1d_filters = in_channels
        if channel_filters:
            in_1d_conv = list()
            in_1d_conv.append(('layer0', nn.Conv1d(in_channels, in_channels * channel_filters[0], kernel_size=1, groups=in_channels)))
            num_1d_filters = num_1d_filters * channel_filters[0]
            if batch_norm:
                in_1d_conv.append(('BN', nn.BatchNorm1d(num_1d_filters)))
            in_1d_conv.append(('tanh', nn.Hardtanh(0, 20, inplace=True)))
            for x in range(1, len(channel_filters)):
                in_1d_conv.append(('layer%d' % x, nn.Conv1d(num_1d_filters, num_1d_filters * channel_filters[x], kernel_size=1, groups=in_channels)))
                num_1d_filters = num_1d_filters * channel_filters[x]
                if batch_norm:
                    in_1d_conv.append(('BN%d' % x, nn.BatchNorm1d(num_1d_filters)))
                in_1d_conv.append(('tanh%d' % x, nn.Hardtanh(0, 20, inplace=True)))

            self.channel_conv = nn.Sequential(OrderedDict(in_1d_conv))

        else:
            self.channel_conv = None
        rnn_input_size = num_1d_filters
        if time_filters:
            in_2d_conv = list()
            in_2d_conv.append(('layer0', nn.Conv2d(1, time_filters[0], kernel_size=(1, time_kernels[0]))))
            if batch_norm:
                in_2d_conv.append(('BN0', nn.BatchNorm2d(time_filters[0])))
            in_2d_conv.append(('tanh0', nn.Hardtanh(0, 20, inplace=True)))
            for x in range(1, len(time_filters)):
                in_2d_conv.append(('layer%d' % x, nn.Conv2d(time_filters[x - 1], time_filters[x], kernel_size=(1, time_kernels[x]))))
                if batch_norm:
                    in_2d_conv.append(('BN%d' % x, nn.BatchNorm2d(time_filters[x])))
                in_2d_conv.append(('tanh%d' % x, nn.Hardtanh(0, 20, inplace=True)))

            rnn_input_size *= time_filters[-1]
            self.time_conv = nn.Sequential(OrderedDict(in_2d_conv))
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
            # if dropout > 0:
            #     fc_out.append(('dropout0', nn.Dropout2d(dropout)))
            fc_out.append(('layer0', nn.Linear(rnn_hidden_size, fc_size[0], bias=True)))
            fc_out.append(('tanh0', nn.Hardtanh(-20, 20)))
            for x in range(1, len(fc_size)):
                # if dropout > 0:
                #     fc_out.append(('dropout%d' % x), nn.Droput2d(dropout))
                fc_out.append(('layer%d' % x , nn.Linear(fc_size[x - 1], fc_size[x], bias=True)))
                fc_out.append(('tanh%d' % x, nn.Hardtanh(-20, 20)))

            fc_out.append(('output', nn.Linear(fc_size[-1], num_classes, bias=True)))
            fully_connected = nn.Sequential(OrderedDict(fc_out))

        else:
            fc_out = list()
            # if dropout > 0:
            #     fc_out.append(('dropout', nn.Dropout2d(dropout)))
            fc_out.append(('output', nn.Linear(rnn_hidden_size, num_classes, bias=True)))
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
        # x is expected to be NxCxT
        if self.channel_conv is not None:
            x = self.channel_conv(x)
        if self.time_conv is not None:
            x = x.unsqueeze(dim=3)  # NxCx1XT
            x = x.permute(0, 3, 1, 2)  # Nx1xCxT
            x = self.time_conv(x)
            sizes = x.size()
            x = x.view(sizes[0], sizes[1] * sizes[2], sizes[3])  # Collapse feature dimension

        x = x.transpose(1, 2).transpose(0, 1).contiguous()  # TxNxC
        x, hidden = self.rnns(x, hidden)  # TxNxH

        if self._output_stride > 1:
            x = x[self._output_stride-1::self._output_stride]

        if self.fc_bn is not None:
            x = x.permute(1, 2, 0).contiguous()  #NxHxT
            x = self.fc_bn(x)
            x = x.permute(0, 2, 1).contiguous()  #NxTxH
        else:
            x = x.permute(1, 0, 2).contiguous()  #NxTxH

        x = self.fc(x).unsqueeze(1)  # Nx1xTxH

        return x.squeeze(1), hidden  # N x T x nClasses

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
            "output_stride":model._output_stride
        }
        return meta


def train(model, data_loader, optimizer, loss_fun, keep_state=False, clip=0, cuda=False):
    model.train()
    for itr, (data, target_cpu) in enumerate(data_loader):
        data, target = Variable(data.squeeze(0)), Variable(target_cpu.squeeze(0))
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
        data, target = Variable(data.squeeze(0)), Variable(target_cpu.squeeze(0))
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

