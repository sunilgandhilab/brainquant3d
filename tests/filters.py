import numpy as np
import tifffile as tif
from pathlib import Path

import unittest

import os
import shutil

from bq3d.image_filters.filters.background_subtraction import RollingBackgroundSubtract, BackgroundSubtract
from bq3d.utils.logger import set_console_level

set_console_level(21)

# Data MUST be coped because background subtraction is done in place

class TestRollingBackgroundSubstraction(unittest.TestCase):

    def setUp(self):
        self.im_filter = RollingBackgroundSubtract()
        self.im_filter.set_inputs({'size': 3})

    def test_npy_2d_uint8(self):
        input_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_2d_signal.npy')
        copy_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_2d_signal_copy.npy')
        shutil.copy(input_path, copy_path)
        data = np.load(copy_path, mmap_mode='r+')
        self.im_filter.set_inputs({'input': data})
        generated_rolling_output = self.im_filter.run()
        correct_rolling_output = np.load('./filters/backgroundsubtraction/rolling_output/testdata_uint8_2d_signal.npy', mmap_mode='r+')
        equal = np.all(generated_rolling_output == correct_rolling_output)
        os.remove(copy_path)
        self.assertTrue(equal)

    def test_npy_3d_uint8(self):
        input_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_3d_signal.npy')
        copy_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_3d_signal_copy.npy')
        shutil.copy(input_path, copy_path)
        data = np.load(copy_path, mmap_mode='r+')
        self.im_filter.set_inputs({'input': data})
        generated_rolling_output = self.im_filter.run()
        correct_rolling_output = np.load('./filters/backgroundsubtraction/rolling_output/testdata_uint8_3d_signal.npy', mmap_mode='r+')
        equal = np.all(generated_rolling_output == correct_rolling_output)
        os.remove(copy_path)
        self.assertTrue(equal)

    def test_tiff_2d_uint8(self):
        input_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_2d_signal.tif')
        copy_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_2d_signal_copy.tif')
        shutil.copy(input_path, copy_path)
        data = tif.memmap(copy_path)
        self.im_filter.set_inputs({'input': data})
        generated_rolling_output = self.im_filter.run()
        correct_rolling_output = tif.memmap('./filters/backgroundsubtraction/rolling_output/testdata_uint8_2d_signal.tif')
        equal = np.all(generated_rolling_output == correct_rolling_output)
        os.remove(copy_path)
        self.assertTrue(equal)

    def test_tiff_3d_uint8(self):
        input_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_3d_signal.tif')
        copy_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_3d_signal_copy.tif')
        shutil.copy(input_path, copy_path)
        data = tif.memmap(copy_path)
        self.im_filter.set_inputs({'input': data})
        generated_rolling_output = self.im_filter.run()
        correct_rolling_output = tif.memmap('./filters/backgroundsubtraction/rolling_output/testdata_uint8_3d_signal.tif')
        equal = np.all(generated_rolling_output == correct_rolling_output)
        os.remove(copy_path)
        self.assertTrue(equal)


class TestBackgroundSubstraction(unittest.TestCase):

    def setUp(self):
        self.im_filter = BackgroundSubtract()

    def test_npy_2d_uint8(self):
        input_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_2d_signal.npy')
        copy_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_2d_signal_copy.npy')
        shutil.copy(input_path, copy_path)
        data = np.load(copy_path, mmap_mode='r+')
        background = np.load('./filters/backgroundsubtraction/input/testdata_uint8_2d_auto.npy')
        self.im_filter.set_inputs({'input': data, 'background': background})
        generated_regular_output = self.im_filter.run()
        correct_regular_output = np.load('./filters/backgroundsubtraction/regular_output/testdata_uint8_2d_signal.npy', mmap_mode='r+')
        equal = np.all(generated_regular_output == correct_regular_output)
        os.remove(copy_path)
        self.assertTrue(equal)

    def test_npy_3d_uint8(self):
        input_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_3d_signal.npy')
        copy_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_3d_signal_copy.npy')
        shutil.copy(input_path, copy_path)
        data = np.load(copy_path, mmap_mode='r+')
        background = np.load('./filters/backgroundsubtraction/input/testdata_uint8_3d_auto.npy')
        self.im_filter.set_inputs({'input': data, 'background': background})
        generated_regular_output = self.im_filter.run()
        correct_regular_output = np.load('./filters/backgroundsubtraction/regular_output/testdata_uint8_3d_signal.npy', mmap_mode='r+')
        equal = np.all(generated_regular_output == correct_regular_output)
        os.remove(copy_path)
        self.assertTrue(equal)

    def test_tiff_2d_uint8(self):
        input_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_2d_signal.tif')
        copy_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_2d_signal_copy.tif')
        shutil.copy(input_path, copy_path)
        data = tif.memmap(copy_path)
        background = tif.memmap('./filters/backgroundsubtraction/input/testdata_uint8_2d_auto.tif')
        self.im_filter.set_inputs({'input': data, 'background': background})
        generated_regular_output = self.im_filter.run()
        correct_regular_output = tif.memmap('./filters/backgroundsubtraction/regular_output/testdata_uint8_2d_signal.tif')
        equal = np.all(generated_regular_output == correct_regular_output)
        os.remove(copy_path)
        self.assertTrue(equal)

    def test_tiff_3d_uint8(self):
        input_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_3d_signal.tif')
        copy_path = Path('./filters/backgroundsubtraction/input/testdata_uint8_3d_signal_copy.tif')
        shutil.copy(input_path, copy_path)
        data = tif.memmap(copy_path)
        background = tif.memmap('./filters/backgroundsubtraction/input/testdata_uint8_3d_auto.tif')
        self.im_filter.set_inputs({'input': data, 'background': background})
        generated_regular_output = self.im_filter.run()
        correct_regular_output = tif.memmap('./filters/backgroundsubtraction/regular_output/testdata_uint8_3d_signal.tif')
        equal = np.all(generated_regular_output == correct_regular_output)
        os.remove(copy_path)
        self.assertTrue(equal)


if __name__ == '__main__':
    unittest.main()
