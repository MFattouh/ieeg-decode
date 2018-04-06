import numpy as np
import unittest
from utils.pytorch_util import CorrCoeff


class TestCorrCoeff(unittest.TestCase):
    def test_fishers_trans(self):
        batch_size = 10
        num_classes = 1
        seq_len = 1000
        output = np.random.rand(batch_size, seq_len, num_classes)
        target = np.random.rand(batch_size, seq_len, num_classes)
        normal_corr = np.corrcoef(target.reshape(1, -1).squeeze().tolist()*3, output.reshape(1, -1).squeeze().tolist()*3)[0, 1]
        for itr in range(3):
            if itr == 0:
                cum_corr = np.zeros((num_classes,1))
                valid_corr = np.zeros((num_classes,1))
                for class_idx in range(num_classes):
                    # compute correlation, apply fisher's transform
                    corr = np.arctanh(np.corrcoef(target[:, :, class_idx].reshape(1, -1).squeeze(),
                                                  output[:, :, class_idx].reshape(1, -1).squeeze())[0, 1])

                    if not np.isnan(corr):
                        cum_corr[class_idx] += corr
                        valid_corr[class_idx] += 1
            # average the correlations across over iterations apply inverse fisher's transform find mean over batch
            if num_classes == 1:
                fishers_corr = np.tanh(cum_corr.squeeze() / valid_corr.squeeze())
            else:
                fishers_corr = dict()
                for i in range(num_classes):
                    fishers_corr['Class%d' % i] = np.tanh(cum_corr[i] / valid_corr[i])

        np.testing.assert_almost_equal(normal_corr, fishers_corr, err_msg="fisher's transform corr error")

    def test_hist_corr_one_batch_3_bins(self):
        batch_size = 1
        num_classes = 1
        seq_len = 1000
        num_bins = 3
        predictions = []
        targets = []
        normal_corr = []
        for bin in range(num_bins):
            predictions.append(np.random.rand(batch_size, seq_len, num_classes))
            targets.append(np.random.rand(batch_size, seq_len, num_classes))
            normal_corr.append(np.corrcoef(targets[-1].reshape(1, -1).squeeze().tolist()*3,
                                           predictions[-1].reshape(1, -1).squeeze().tolist()*3)[0, 1])

        targets = np.concatenate(targets, axis=1)
        predictions = np.concatenate(predictions, axis=1)
        fishers_corr = CorrCoeff(bin_size=seq_len).hist_corrcoeff(targets, predictions)
        self.assertEqual(fishers_corr.shape[0], num_bins, 'wrong number of bins')
        for bin in range(num_bins):
            np.testing.assert_almost_equal(normal_corr[bin], fishers_corr[bin].tolist()[0], err_msg="fisher's transform corr error")

    def test_hist_corr_10_batches_3_bins(self):
        batch_size = 10
        num_classes = 1
        seq_len = 1000
        num_bins = 3
        predictions = []
        targets = []
        normal_corr = []
        for bin in range(num_bins):
            predictions.append(np.random.rand(batch_size, seq_len, num_classes))
            targets.append(np.random.rand(batch_size, seq_len, num_classes))
            normal_corr.append(np.corrcoef(targets[-1].reshape(1, -1).squeeze().tolist()*3,
                                           predictions[-1].reshape(1, -1).squeeze().tolist()*3)[0, 1])

        targets = np.concatenate(targets, axis=1)
        predictions = np.concatenate(predictions, axis=1)
        fishers_corr = CorrCoeff(bin_size=seq_len).hist_corrcoeff(targets, predictions)
        self.assertEqual(fishers_corr.shape[0], num_bins, 'wrong number of bins')
        for bin in range(num_bins):
            np.testing.assert_almost_equal(normal_corr[bin], fishers_corr[bin].tolist()[0], err_msg="fisher's transform corr error")

    def test_hist_corr_10_batches_3_bins_last_not_complete(self):
        batch_size = 10
        num_classes = 1
        seq_len = 1000
        num_bins = 3
        predictions = []
        targets = []
        normal_corr = []
        for bin in range(num_bins-1):
            predictions.append(np.random.rand(batch_size, seq_len, num_classes))
            targets.append(np.random.rand(batch_size, seq_len, num_classes))
            normal_corr.append(np.corrcoef(targets[-1].reshape(1, -1).squeeze().tolist()*3,
                                           predictions[-1].reshape(1, -1).squeeze().tolist()*3)[0, 1])

        predictions.append(np.random.rand(batch_size, seq_len-20, num_classes))
        targets.append(np.random.rand(batch_size, seq_len-20, num_classes))
        normal_corr.append(np.corrcoef(targets[-1].reshape(1, -1).squeeze().tolist()*3,
                                           predictions[-1].reshape(1, -1).squeeze().tolist()*3)[0, 1])
        targets = np.concatenate(targets, axis=1)
        predictions = np.concatenate(predictions, axis=1)
        fishers_corr = CorrCoeff(bin_size=seq_len).hist_corrcoeff(targets, predictions)
        self.assertEqual(fishers_corr.shape[0], num_bins, 'wrong number of bins')
        for bin in range(num_bins):
            np.testing.assert_almost_equal(normal_corr[bin], fishers_corr[bin].tolist()[0], err_msg="fisher's transform corr error")


if __name__ == '__main__':
    unittest.main()