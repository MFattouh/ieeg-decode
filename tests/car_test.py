import unittest
from utils.dataset_util import car
import numpy as np


class TestCAR(unittest.TestCase):
    def setUp(self):
        self.n_channels = 16
        self.seq_len = 100
        self.valid = np.array([True]*self.n_channels)
        self.ecog_channels_idx = np.array([False]*self.n_channels)
        self.ecog_channels_idx[[1, 2, 4]] = True
        self.hbox = np.array([1]*self.n_channels)
        self.channels = np.ones((self.n_channels, self.seq_len))
        self.channels += 1e-2*np.random.rand(self.n_channels, self.seq_len)

    def test_all_valid_one_hb(self):
        ecog_channels_after_car = car(self.channels, self.ecog_channels_idx, self.valid, self.hbox)
        mean = ecog_channels_after_car.mean(axis=1)
        std = ecog_channels_after_car.std(axis=1)
        for idx in range(np.sum(self.ecog_channels_idx)):
            np.testing.assert_almost_equal(mean[idx], 0, 0.1, 'mean should be around 0')
            np.testing.assert_almost_equal(std[idx], 1, 0.1, 'SD should be around 1')

    def test_one_nonvalid_one_hb(self):
        self.valid[0] = False
        self.channels[0, ] = 5 * np.random.rand(1, self.seq_len)

        ecog_channels_after_car = car(self.channels, self.ecog_channels_idx, self.valid, self.hbox)
        mean = ecog_channels_after_car.mean(axis=1)
        std = ecog_channels_after_car.std(axis=1)
        for idx in range(np.sum(self.ecog_channels_idx)):
            np.testing.assert_almost_equal(mean[idx], 0, 0.1, 'mean should be around 0')
            np.testing.assert_almost_equal(std[idx], 1, 0.1, 'SD should be around 1')


if __name__ == '__main__':
    print('index', np.char.find(np.chararray.lower(np.array(['heLLO', 'bLA bla'])), 'lo'))
    unittest.main()