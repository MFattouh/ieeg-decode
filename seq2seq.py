import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class EncoderRNN(nn.Module):
    def __init__(self, num_channels, hidden_size, n_layers=1, keep_state=False, batch_first=False):
        super(EncoderRNN, self).__init__()

        self.num_channels = num_channels
        self.hidden_size = hidden_size
        self.hidden_layers = n_layers
        # currently input should have shape TxNxC or NxTxC
        self.gru = nn.GRU(num_channels, hidden_size, n_layers, batch_first=batch_first)

    def forward(self, ecog_inputs, hidden):
        # Note: we run this all at once (over the whole input sequence)
        output, hidden = self.gru(ecog_inputs, hidden)
        return output, hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size, cuda=False):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.cuda = cuda

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden NxH
        # encoder_outputs NxSxH
        size = encoder_outputs.size()
        N, S, H = size[0], size[1], size[2]

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros((S, N)))  # SxN
        if self.cuda:
            attn_energies = attn_energies.cuda()

        # Calculate energies for each encoder output
        for i in range(S):
            attn_energies[i] = self.score(hidden, encoder_outputs[:, i])

        # Normalize energies to weights in range 0 to 1
        return F.softmax(attn_energies, dim=0).unsqueeze(2)  # SxNx1

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            # energy = hidden.dot(encoder_output)
            energy = hidden.mm(encoder_output.transpose(0, 1)).diag(diagonal=0)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.mm(energy.transpose(0, 1)).diag(diagonal=0)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()

        # Keep parameters for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.hidden_layers = n_layers
        self.dropout_p = dropout_p

        # Define layers
        self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, hidden_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size, cuda=True)

    def forward(self, last_output, last_context, last_hidden, encoder_outputs):
        # last_output   NxH
        # last_context  NxH
        # encoder outputs SxNxH
        # encoder outputs NxSxH
        # Note: we run this one step at a time

        # Combine encoded last pred and last context, run through RNN
        rnn_input = torch.cat((last_output.unsqueeze(0), last_context.unsqueeze(0)), 2)  # 1XNx[H; H]
        rnn_output, hidden = self.gru(rnn_input, last_hidden)  # 1xNxH
        rnn_output = rnn_output.squeeze(0)  # NxH
        # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
        attn_weights = self.attn(rnn_output, encoder_outputs).permute(1, 2, 0)  # N x 1 x S
        context = attn_weights.bmm(encoder_outputs)  # N x 1 x H

        # Final output layer (next word prediction) using the RNN hidden state and context vector
        context = context.squeeze(1)        # N x S=1 x N -> N x H
        output = self.out(torch.cat((rnn_output, context), 1))  # N x H

        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights

# class AttnDecoderRNN(nn.Module):
#   Implementaion of Neural Transducer
#     def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
#         super(AttnDecoderRNN, self).__init__()
#
#         # Keep parameters for reference
#         self.attn_model = attn_model
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.n_layers = n_layers
#         self.dropout_p = dropout_p
#
#         # Define layers
#         self.RNN1 = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
#         self.RNN2 = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
#         self.out = nn.Linear(hidden_size * 2, output_size)
#
#         # Choose attention model
#         if attn_model != 'none':
#             self.attn = Attn(attn_model, hidden_size)
#
#     def forward(self, last_output, last_context, last_hidden1, last_hidden2, encoder_outputs):
#         # Note: we run this one step at a time
#
#         # Combine embedded input word and last context, run through RNN
#         rnn1_input = torch.cat((last_context, last_output), 2)
#         rnn1_output, hidden1 = self.RNN1(rnn1_input, last_hidden1)
#
#         # Calculate attention from current RNN state and all encoder outputs; apply to encoder outputs
#         attn_weights = self.attn(rnn1_output, encoder_outputs)
#         context = attn_weights.bmm(encoder_outputs)
#
#         # the second RNN uses the context and first RNN output as input
#         rnn2_input = torch.cat((context, rnn1_output), 2)
#         rnn2_output, hidden2 = self.RNN2(rnn2_input, last_hidden2)
#
#         # Return final output, hidden state, and attention weights (for visualization)
#         return rnn2_output, context, hidden1, hidden2, attn_weights


def train(encoder, decoder, fc, data_loader, encoder_optimizer, decoder_optimizer, fc_optimizer, criterion,
          keep_state=False, clip=0, teacher_forcing_ratio=0.5, cuda=False):

    encoder.train()
    decoder.train()
    fc.train()
    for itr, (data, target_cpu) in enumerate(data_loader):
        data, target = Variable(data.squeeze(0)), Variable(target_cpu.squeeze(0))
        if cuda:
            data, target = data.cuda(), target.cuda()

        target_length = target.size()[1]
        batch_size = target.size()[0]
        # if the hidden is not none then the model should keep the state from on iteration to other
        # -> detach to stop back-propagation to older state
        if keep_state and itr > 0:
            encoder_hidden.detach_()
            decoder_hidden.detach_()
            decoder_input.detach_()
            decoder_context.detach_()

        else:
            encoder_hidden = Variable(torch.zeros(encoder.hidden_layers, list(data.size())[0],
                                      encoder.hidden_size))
            if cuda:
                encoder_hidden = encoder_hidden.cuda()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        fc_optimizer.zero_grad()
        # forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(data, encoder_hidden)  # NxTxH
        # for the first time step assume last prediction is 0.
        # TODO: init. pred value should be reviewed
        if itr == 0 or not keep_state:
            decoder_input = Variable(torch.zeros(batch_size, decoder.hidden_size))
            decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))
            decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
            if cuda:
                decoder_input = decoder_input.cuda()
                decoder_context = decoder_context.cuda()
        # Choose whether to use teacher forcing
        use_teacher_forcing = np.random.rand(1) < teacher_forcing_ratio
        #     print('first decoder input', decoder_input.size())

        loss = 0
        if use_teacher_forcing:

            # Teacher forcing: Use the ground-truth target as the next input
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = \
                    decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)

                # fully connected layer for the final output
                output = fc(decoder_output)
                loss += criterion(output.squeeze(), target[:, di].squeeze())
                # loss += criterion(decoder_output, target_variable[di])
                # TODO: figure out what could be a suitable decoder input
                decoder_input = target_variable[di]  # Next target is next input

        else:
            # Without teacher forcing: use network's own prediction as the next input
            for di in range(target_length):
                decoder_output, decoder_context, decoder_hidden, decoder_attention = \
                    decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
                # print('decoder output', decoder_output.size())
                # fully connected layer for the final output
                output = fc(decoder_output)
                loss += criterion(output.squeeze(), target[:, di].squeeze())
                # loss += criterion(decoder_output, target_variable[di])

                # Get most likely word index (highest value) from output
                # topv, topi = decoder_output.data.topk(1)
                # TODO: figure out what could be a suitable decoder input
                decoder_input = decoder_output
                # decoder_input = emmbeding(Variable(topi)).squeeze(1)  # Chosen word is next input
        #             print('decoder input', decoder_input.size())

        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(fc.parameters(), clip)

        encoder_optimizer.step()
        decoder_optimizer.step()
        fc_optimizer.step()


def evaluate(encoder, decoder, fc, data_loader, criterion, keep_state=False, writer=None, epoch=0, cuda=False):
    encoder.eval()
    decoder.eval()
    fc.eval()
    # loop over the training dataset
    avg_loss = 0
    for itr, (data, target_cpu) in enumerate(data_loader):
        data, target = Variable(data.squeeze(0)), Variable(target_cpu.squeeze(0))
        if cuda:
            data, target = data.cuda(), target.cuda()

        if keep_state and itr > 0:
            encoder_hidden.detach_()
            decoder_hidden.detach_()
            decoder_input.detach_()
            decoder_context.detach_()

        else:
            encoder_hidden = Variable(torch.zeros(encoder.hidden_layers, list(data.size())[0],
                                                  encoder.hidden_size))
            if cuda:
                encoder_hidden = encoder_hidden.cuda()

        # forward pass through encoder
        encoder_outputs, encoder_hidden = encoder(data, encoder_hidden)  # NxTxH

        # for the first time step assume last prediction is 0.
        # TODO: init. pred value should be reviewed
        loss = 0
        if itr == 0 or not keep_state:
            batch_size = target.size()[0]
            target_length = target.size()[1]
            num_classes = target.size()[2]
            decoder_input = Variable(torch.zeros(batch_size, decoder.hidden_size))
            decoder_context = Variable(torch.zeros(batch_size, decoder.hidden_size))
            decoder_hidden = encoder_hidden  # Use last hidden state from encoder to start decoder
            cum_corr = np.zeros((batch_size, num_classes))
            valid_corr = np.zeros((batch_size, num_classes))
            if cuda:
                decoder_input = decoder_input.cuda()
                decoder_context = decoder_context.cuda()

        predictions = np.zeros((target_length, batch_size, num_classes))
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = \
                decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            # fully connected layer for the final output
            output = fc(decoder_output)
            predictions[di, :, :] = output.data.cpu().numpy()
            loss += criterion(output.squeeze(), target[:, di].squeeze())

            # TODO: figure out what could be a suitable decoder input
            decoder_input = decoder_output

        avg_loss += loss.data.cpu().numpy().squeeze()
        # compute the correlation coff. for each seq. in batch
        target = target_cpu.squeeze(0).numpy()
        for batch_idx in range(batch_size):
            for class_idx in range(num_classes):
                # compute correlation, apply fisher's transform
                corr = np.arctanh(np.corrcoef(target[batch_idx, :, class_idx].squeeze(),
                                              predictions[:, batch_idx, class_idx].squeeze())[0, 1])
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
