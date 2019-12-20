import numpy as np
import tifffile as tif
from pathlib import Path

import unittest

import os
import shutil

from bq3d.image_filters.filters.background_subtraction import RollingBackgroundSubtract, BackgroundSubtract
from bq3d.image_filters.filters.DoG import DoG
from bq3d.image_filters.filters.erosion import Erode
from bq3d.image_filters.filters.h_max import HMax
from bq3d.image_filters.filters.max import Max
from bq3d.image_filters.filters.projection import Project

from bq3d.utils.logger import set_console_level
set_console_level(21)

#TODO: DoG, Erode are producing blank images.

# float32
# float64
# uint8
# uint16
# int32
# uint32

# Data MUST be coped because background subtraction is done in place


def test(im_filter, input_path, correct_output_path, **extra_kwargs):
    input_path = Path(input_path)
    copy_path = Path(input_path.stem + '_copy').with_suffix('.tif')
    shutil.copy(input_path, copy_path)
    data = tif.imread(str(copy_path))
    im_filter.set_inputs({**{'input': data}, **extra_kwargs})
    generated_output = im_filter.run()
    correct_output = tif.memmap(correct_output_path)
    equal = np.all(generated_output == correct_output)
    os.remove(copy_path)
    return equal


class TestRollingBackgroundSubstraction(unittest.TestCase):
    # Only accepts uint8

    def setUp(self):
        self.im_filter = RollingBackgroundSubtract()
        self.im_filter.set_inputs({'size': 3})

    def test_2d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_2d_signal.tif',
                     './filters/backgroundsubtraction/rolling_output/out_uint8_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_3d_signal.tif',
                     './filters/backgroundsubtraction/rolling_output/out_uint8_3d_signal.tif'
                     )
        self.assertTrue(equal)


class TestBackgroundSubstraction(unittest.TestCase):
    # Only accepts uint8

    def setUp(self):
        self.im_filter = BackgroundSubtract()

    def test_2d_uint8(self):
        background = tif.memmap('./input/testdata_uint8_2d_auto.tif')
        equal = test(self.im_filter,
                     './input/testdata_uint8_2d_signal.tif',
                     './filters/backgroundsubtraction/regular_output/out_uint8_2d_signal.tif',
                     **{'background': background}
                     )
        self.assertTrue(equal)

    def test_3d_uint8(self):
        background = tif.memmap('./input/testdata_uint8_3d_auto.tif')
        equal = test(self.im_filter,
                     './input/testdata_uint8_3d_signal.tif',
                     './filters/backgroundsubtraction/regular_output/out_uint8_3d_signal.tif',
                     **{'background': background}
                     )
        self.assertTrue(equal)


class TestDoG(unittest.TestCase):

    def setUp(self):
        self.im_filter = DoG()

    def test_2d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_2d_signal.tif',
                     './filters/DoG/output/out_uint8_2d_signal.tif',
                     **{'size': (3, 3), 'sigma': (2, 2), 'sigma2': (2, 2)}
                     )
        self.assertTrue(equal)

    def test_2d_uint16(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_2d_signal.tif',
                     './filters/DoG/output/out_uint16_2d_signal.tif',
                     **{'size': (3, 3), 'sigma': (2, 2), 'sigma2': (2, 2)}
                     )
        self.assertTrue(equal)

    def test_2d_float32(self):
        equal = test(self.im_filter,
                     './input/testdata_float32_2d_signal.tif',
                     './filters/DoG/output/out_float32_2d_signal.tif',
                     **{'size': (3, 3), 'sigma': (2, 2), 'sigma2': (2, 2)}
                     )
        self.assertTrue(equal)

    def test_2d_int32(self):
        equal = test(self.im_filter,
                     './input/testdata_int32_2d_signal.tif',
                     './filters/DoG/output/out_int32_2d_signal.tif',
                     **{'size': (3, 3), 'sigma': (2, 2), 'sigma2': (2, 2)}
                     )
        self.assertTrue(equal)

    def test_2d_uint32(self):
        equal = test(self.im_filter,
                     './input/testdata_uint32_2d_signal.tif',
                     './filters/DoG/output/out_uint32_2d_signal.tif',
                     **{'size': (3, 3), 'sigma': (2, 2), 'sigma2': (2, 2)}
                     )
        self.assertTrue(equal)

    def test_2d_float64(self):
        equal = test(self.im_filter,
                     './input/testdata_float64_2d_signal.tif',
                     './filters/DoG/output/out_float64_2d_signal.tif',
                     **{'size': (3, 3), 'sigma': (2, 2), 'sigma2': (2, 2)}
                     )
        self.assertTrue(equal)

    def test_3d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_3d_signal.tif',
                     './filters/DoG/output/out_uint8_3d_signal.tif',
                     **{'size': (3, 3, 3), 'sigma': (2, 2, 2), 'sigma2': (2, 2, 2)}
                     )
        self.assertTrue(equal)

    def test_3d_uint16(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_3d_signal.tif',
                     './filters/DoG/output/out_uint16_3d_signal.tif',
                     **{'size': (3, 3, 3), 'sigma': (2, 2, 2), 'sigma2': (2, 2, 2)}
                     )
        self.assertTrue(equal)

    def test_3d_float32(self):
        equal = test(self.im_filter,
                     './input/testdata_float32_3d_signal.tif',
                     './filters/DoG/output/out_float32_3d_signal.tif',
                     **{'size': (3, 3, 3), 'sigma': (2, 2, 2), 'sigma2': (2, 2, 2)}
                     )
        self.assertTrue(equal)

    def test_3d_int32(self):
        equal = test(self.im_filter,
                     './input/testdata_int32_3d_signal.tif',
                     './filters/DoG/output/out_int32_3d_signal.tif',
                     **{'size': (3, 3, 3), 'sigma': (2, 2, 2), 'sigma2': (2, 2, 2)}
                     )
        self.assertTrue(equal)

    def test_3d_uint32(self):
        equal = test(self.im_filter,
                     './input/testdata_uint32_3d_signal.tif',
                     './filters/DoG/output/out_uint32_3d_signal.tif',
                     **{'size': (3, 3, 3), 'sigma': (2, 2, 2), 'sigma2': (2, 2, 2)}
                     )
        self.assertTrue(equal)

    def test_3d_float64(self):
        equal = test(self.im_filter,
                     './input/testdata_float64_3d_signal.tif',
                     './filters/DoG/output/out_float64_3d_signal.tif',
                     **{'size': (3, 3, 3), 'sigma': (2, 2, 2), 'sigma2': (2, 2, 2)}
                     )
        self.assertTrue(equal)


class TestErode(unittest.TestCase):

    def setUp(self):
        self.im_filter = Erode()

    def test_2d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_2d_signal.tif',
                     './filters/erosion/output/out_uint8_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_uint16(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_2d_signal.tif',
                     './filters/erosion/output/out_uint16_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_float32(self):
        equal = test(self.im_filter,
                     './input/testdata_float32_2d_signal.tif',
                     './filters/erosion/output/out_float32_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_int32(self):
        equal = test(self.im_filter,
                     './input/testdata_int32_2d_signal.tif',
                     './filters/erosion/output/out_int32_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_uint32(self):
        equal = test(self.im_filter,
                     './input/testdata_uint32_2d_signal.tif',
                     './filters/erosion/output/out_uint32_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_float64(self):
        equal = test(self.im_filter,
                     './input/testdata_float64_2d_signal.tif',
                     './filters/erosion/output/out_float64_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_3d_signal.tif',
                     './filters/erosion/output/out_uint8_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint16(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_3d_signal.tif',
                     './filters/erosion/output/out_uint16_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_float32(self):
        equal = test(self.im_filter,
                     './input/testdata_float32_3d_signal.tif',
                     './filters/erosion/output/out_float32_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_int32(self):
        equal = test(self.im_filter,
                     './input/testdata_int32_3d_signal.tif',
                     './filters/erosion/output/out_int32_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint32(self):
        equal = test(self.im_filter,
                     './input/testdata_uint32_3d_signal.tif',
                     './filters/erosion/output/out_uint32_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_float64(self):
        equal = test(self.im_filter,
                     './input/testdata_float64_3d_signal.tif',
                     './filters/erosion/output/out_float64_3d_signal.tif'
                     )
        self.assertTrue(equal)


class TestHMax(unittest.TestCase):

    def setUp(self):
        self.im_filter = HMax()
        self.im_filter.set_inputs({'hMax': 100})

    def test_2d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_2d_signal.tif',
                     './filters/h_max/output/out_uint8_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_uint16(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_2d_signal.tif',
                     './filters/h_max/output/out_uint16_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_float32(self):
        equal = test(self.im_filter,
                     './input/testdata_float32_2d_signal.tif',
                     './filters/h_max/output/out_float32_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_int32(self):
        equal = test(self.im_filter,
                     './input/testdata_int32_2d_signal.tif',
                     './filters/h_max/output/out_int32_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_uint32(self):
        equal = test(self.im_filter,
                     './input/testdata_uint32_2d_signal.tif',
                     './filters/h_max/output/out_uint32_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_float64(self):
        equal = test(self.im_filter,
                     './input/testdata_float64_2d_signal.tif',
                     './filters/h_max/output/out_float64_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_3d_signal.tif',
                     './filters/h_max/output/out_uint8_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint16(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_3d_signal.tif',
                     './filters/h_max/output/out_uint16_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_float32(self):
        equal = test(self.im_filter,
                     './input/testdata_float32_3d_signal.tif',
                     './filters/h_max/output/out_float32_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_int32(self):
        equal = test(self.im_filter,
                     './input/testdata_int32_3d_signal.tif',
                     './filters/h_max/output/out_int32_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint32(self):
        equal = test(self.im_filter,
                     './input/testdata_uint32_3d_signal.tif',
                     './filters/h_max/output/out_uint32_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_float64(self):
        equal = test(self.im_filter,
                     './input/testdata_float64_3d_signal.tif',
                     './filters/h_max/output/out_float64_3d_signal.tif'
                     )
        self.assertTrue(equal)


#TODO: Add IlluminationCorrection filter test.


class TestMax(unittest.TestCase):

    def setUp(self):
        self.im_filter = Max()

    def test_2d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_2d_signal.tif',
                     './filters/max/output/out_uint8_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_uint16(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_2d_signal.tif',
                     './filters/max/output/out_uint16_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_float32(self):
        equal = test(self.im_filter,
                     './input/testdata_float32_2d_signal.tif',
                     './filters/max/output/out_float32_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_int32(self):
        equal = test(self.im_filter,
                     './input/testdata_int32_2d_signal.tif',
                     './filters/max/output/out_int32_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_uint32(self):
        equal = test(self.im_filter,
                     './input/testdata_uint32_2d_signal.tif',
                     './filters/max/output/out_uint32_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_2d_float64(self):
        equal = test(self.im_filter,
                     './input/testdata_float64_2d_signal.tif',
                     './filters/max/output/out_float64_2d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint8(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_3d_signal.tif',
                     './filters/max/output/out_uint8_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint16(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_3d_signal.tif',
                     './filters/max/output/out_uint16_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_float32(self):
        equal = test(self.im_filter,
                     './input/testdata_float32_3d_signal.tif',
                     './filters/max/output/out_float32_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_int32(self):
        equal = test(self.im_filter,
                     './input/testdata_int32_3d_signal.tif',
                     './filters/max/output/out_int32_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_uint32(self):
        equal = test(self.im_filter,
                     './input/testdata_uint32_3d_signal.tif',
                     './filters/max/output/out_uint32_3d_signal.tif'
                     )
        self.assertTrue(equal)

    def test_3d_float64(self):
        equal = test(self.im_filter,
                     './input/testdata_float64_3d_signal.tif',
                     './filters/max/output/out_float64_3d_signal.tif'
                     )
        self.assertTrue(equal)


#TODO: Add median filters


class TestProject(unittest.TestCase):

    def setUp(self):
        self.im_filter = Project()

    def test_3d_uint8_max(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_3d_signal.tif',
                     './filters/projection/max_output/out_uint8_3d_signal.tif',
                     **{'method': 'max'}
                     )
        self.assertTrue(equal)

    def test_3d_uint16_max(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_3d_signal.tif',
                     './filters/projection/max_output/out_uint16_3d_signal.tif',
                     **{'method': 'max'}
                     )
        self.assertTrue(equal)

    def test_3d_float32_max(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_3d_signal.tif',
                     './filters/projection/max_output/out_float32_3d_signal.tif',
                     **{'method': 'max'}
                     )
        self.assertTrue(equal)

    def test_3d_uint8_min(self):
        equal = test(self.im_filter,
                     './input/testdata_uint8_3d_signal.tif',
                     './filters/projection/min_output/out_uint8_3d_signal.tif',
                     **{'method': 'min'}
                     )
        self.assertTrue(equal)

    def test_3d_uint16_min(self):
        equal = test(self.im_filter,
                     './input/testdata_uint16_3d_signal.tif',
                     './filters/projection/min_output/out_uint16_3d_signal.tif',
                     **{'method': 'min'}
                     )
        self.assertTrue(equal)


#TODO: Add SurfaceExtraction filter tests.


#TODO: Add Template filter tests.

if __name__ == '__main__':
    unittest.main()
