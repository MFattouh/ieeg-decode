from utils.pytorch_util import *
from models.seq2seq import *
from models.hybrid import *
import unittest
import numpy as np
from torch.utils.data import DataLoader


class TestECoGDataSet(unittest.TestCase):
    def test_length(self):
        X = np.arange(115)
        dataset = ECoGDatast(X, X, window=40, stride=40)
        self.assertEqual(len(dataset), 2, "expected 2 indices")

    def test_last_index_values(self):
        X = np.arange(115)
        y = X + 20
        dataset = ECoGDatast(X, y, window=40, stride=40)
        x_last, y_last = dataset.__getitem__(1)
        np.testing.assert_array_equal(x_last, np.arange(40, 80, dtype=np.float32), "wrong x returned")
        np.testing.assert_array_equal(y_last, np.arange(40, 80, dtype=np.float32).reshape(-1, 1)+20, "wrong y returned")

    def test_test_x2yratio(self):
        X = np.arange(115)
        y = np.arange(3)
        dataset = ECoGDatast(X, y, window=40, stride=40, x2y_ratio=40)
        self.assertEqual(len(dataset), 2, "expected 2 indices")
        x_last, y_last = dataset.__getitem__(1)
        np.testing.assert_array_equal(x_last, np.arange(40, 80, dtype=np.float32), "wrong x returned")
        np.testing.assert_array_equal(y_last.tolist(), 1, "wrong y returned")

    def test_window_is_two_strides(self):
        X = np.arange(125)
        y = np.arange(3)
        dataset = ECoGDatast(X, y, window=80, stride=40, x2y_ratio=40)
        self.assertEqual(len(dataset), 2, "expected 2 indices")
        x_last, y_last = dataset.__getitem__(1)
        np.testing.assert_array_equal(x_last, np.arange(40, 120, dtype=np.float32), "wrong x returned")
        np.testing.assert_array_equal(y_last, np.array([1, 2]).reshape(-1, 1), "wrong y returned")


class TestConcatDataset(unittest.TestCase):

    def test_batch_first(self):
        n_channels = 64
        n_samples = 125
        X = np.ones((n_samples, n_channels))
        y = X + 20
        dataset1 = ECoGDatast(X, y, window=40, stride=40)
        conc_dataset = ConcatDataset([dataset1, dataset1])
        x_dataset, y_dataset = conc_dataset.__getitem__(0)
        self.assertEqual((40, 2, n_channels), x_dataset.shape, 'batch first false does not work')
        conc_dataset = ConcatDataset([dataset1, dataset1], batch_first=True)
        x_dataset, y_dataset = conc_dataset.__getitem__(0)
        self.assertEqual((2, 40, n_channels), x_dataset.shape, 'batch first true does not work')

    def test_time_last(self):
        n_channels = 64
        n_samples = 125
        X = np.ones((n_samples, n_channels))
        y = X + 20
        dataset1 = ECoGDatast(X, y, window=40, stride=40)
        conc_dataset = ConcatDataset([dataset1, dataset1], time_last=True)
        x_dataset, y_dataset = conc_dataset.__getitem__(0)
        self.assertEqual((n_channels, 2, 40), x_dataset.shape, 'batch first false does not work')
        conc_dataset = ConcatDataset([dataset1, dataset1], batch_first=True, time_last=True)
        x_dataset, y_dataset = conc_dataset.__getitem__(0)
        self.assertEqual((2, n_channels, 40), x_dataset.shape, 'batch first true does not work')

    def test_DataLoader(self):
        n_channels = 64
        n_samples = 125
        stride = 40
        window = 40
        X1 = np.random.rand(n_samples, n_channels)
        y1 = X1[:, 0] + 0.1
        dataset1 = ECoGDatast(X1, y1, window=window, stride=stride)
        X2 = np.random.rand(n_samples, n_channels)
        y2 = X2[:, 0] + 0.1
        dataset2 = ECoGDatast(X2, y2, window=window, stride=stride)
        manual_concat_X = np.concatenate((np.expand_dims(X1, 0), np.expand_dims(X2, 0)), axis=0)
        # y1 and y2 will be expanded by the ECoDDataset getindex method
        y1 = np.expand_dims(y1, axis=1)
        y2 = np.expand_dims(y2, axis=1)
        manual_concat_y = np.concatenate((np.expand_dims(y1, 0), np.expand_dims(y2, 0)), axis=0)
        conc_dataset = ConcatDataset([dataset1, dataset2], batch_first=True, time_last=False)
        dataset_loader = iter(DataLoader(conc_dataset, batch_size=1, shuffle=False, num_workers=0))
        for idx in range(len(dataset1)):
            batch_x, batch_y = next(dataset_loader)
            np.testing.assert_almost_equal(batch_x.numpy().squeeze(0),
                                           manual_concat_X[:, idx*stride:idx*stride+window, :],
                                           err_msg='batch %d Xs does not match' % idx)

            np.testing.assert_almost_equal(batch_y.numpy().squeeze(0),
                                           manual_concat_y[:, idx*stride:idx*stride+window],
                                           err_msg='batch %d ys did not match' % idx)


class TestEncoderRNN(unittest.TestCase):
    def test_output_size(self):
        seq_len = 100
        n_channels = 64
        hidden_size = 10
        batch_size = 32
        n_layers = 2
        encoder = EncoderRNN(n_channels, hidden_size, n_layers)
        hidden = encoder.init_hidden(batch_size)
        self.assertEqual([n_layers, batch_size, hidden_size], list(hidden.size()), 'init. hidden size is wrong')
        output, hidden = encoder(Variable(
            torch.from_numpy(np.random.randn(seq_len, batch_size, n_channels).astype(np.float32))),
            hidden)
        self.assertEqual([seq_len, batch_size, hidden_size], list(output.size()), 'output size is wrong')
        self.assertEqual([n_layers, batch_size, hidden_size], list(hidden.size()), 'hidden size is wrong')


class TestHybridModel(unittest.TestCase):
    def test_BNLSTM(self):
        hidden_size = 64
        n_layers = 2
        in_channels = 62
        batch_size = 12
        window = 1000
        model = HybridModel(rnn_type='lstm', num_classes=1, rnn_hidden_size=hidden_size, rnn_layers=n_layers,
                            in_channels=in_channels,
                            channel_filters=[], time_filters=[], time_kernels=[], fc_size=[10],
                            output_stride=0, batch_norm=False, max_length=window, dropout=0.1)

        hidden = model.init_hidden(batch_size)
        dummy_input = Variable(torch.randn(batch_size, 1, in_channels, window))
        dummy_output, hidden = model(dummy_input, hidden)
        print(dummy_output.size())

# class TestSimpleEncDec(unittest.TestCase):
#     n_channels = 64
#     batch_size = 2
#     hidden_size = 10
#     encoder_test = EncoderRNN(n_channels, hidden_size, 2)  # 64 channels, 10 gru units and 2 layers
#     decoder_test = AttnDecoderRNN('general', hidden_size, hidden_size, 2)  # 10 units each takes 10 features vector
#     fc = nn.Linear(10, 1)  # combines the outputs of the decoder into a single value
#     print(encoder_test)
#     print(decoder_test)
#
#     ch_1 = np.ones((1, batch_size, n_channels), dtype=np.float32)
#     encoder_hidden = encoder_test.init_hidden(batch_size)
#     ieeg_inputs = Variable(torch.from_numpy(np.concatenate((ch_1, ch_1 * 2, ch_1 * 3), axis=0)))
#     encoder_outputs, encoder_hidden = encoder_test(ieeg_inputs, encoder_hidden)
#     decoder_attns = torch.zeros(3, batch_size, 3)
#     decoder_hidden = encoder_hidden
#     decoder_context = Variable(torch.zeros(batch_size, decoder_test.hidden_size))
#     last_pred = Variable(torch.randn((3, batch_size, hidden_size)))
#     for i in range(3):
#         decoder_output, decoder_context, decoder_hidden, decoder_attn = decoder_test(last_pred[i], decoder_context,
#                                                                                      decoder_hidden, encoder_outputs)
#         print(decoder_output.size(), decoder_hidden.size(), decoder_attn.size())
#         decoder_attns[i] = decoder_attn.squeeze(1).cpu().data


class TestCropsFromTrials(unittest.TestCase):
    def setUp(self):
        self.X = np.arange(115).squeeze()
        self.y = self.X.copy()
        self.crop_len = 30  # samples

    def test_no_strie(self):
        X_crops, y_crops = crops_from_trial(self.X, self.y, self.crop_len, normalize=False)
        self.assertEqual(len(X_crops), len(y_crops), 'should have the same number of crops')
        self.assertEqual(len(X_crops), 3, 'Should be exactly 3')
        for idx, (X_crop, y_crop) in enumerate(zip(X_crops, y_crops)):
            np.testing.assert_array_equal(self.X[idx*self.crop_len:(idx+1)*self.crop_len], X_crop, 'crop %d did not match'%idx)
            np.testing.assert_array_equal(self.y[idx*self.crop_len:(idx+1)*self.crop_len], y_crop, 'crop %d did not match'%idx)

    def test_strid(self):
        stride = 20
        X_crops, y_crops = crops_from_trial(self.X, self.y, self.crop_len, stride=stride, normalize=False)
        self.assertEqual(len(X_crops), len(y_crops), 'should have the same number of crops')
        self.assertEqual(5, len(X_crops), 'Should be exactly 5')
        for idx, (X_crop, y_crop) in enumerate(zip(X_crops, y_crops)):
            np.testing.assert_array_equal(self.X[idx*stride:idx*stride + self.crop_len], X_crop, 'crop %d did not match'%idx)
            np.testing.assert_array_equal(self.y[idx*stride:idx*stride + self.crop_len], y_crop, 'crop %d did not match'%idx)


if __name__ == '__main__':
    unittest.main()
