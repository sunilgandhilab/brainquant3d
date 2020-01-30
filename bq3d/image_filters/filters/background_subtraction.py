import cv2
import numpy as np

import uuid
from bq3d import io
import bq3d.io.TIF as tif
from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase
from bq3d.image_filters.filters._background_subtraction import subtract_background_rolling_ball

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


class RollingBackgroundSubtract(FilterBase):
    """
    Remove background via subtracting a morphological opening from the original image.

    Background removal is done z-slice by z-slice.
    Call using :meth:`filter_image` with 'RollingBackgroundSubtract' as filter.

    Attributes:
        input (array): 2D or 3D image to pass through filter.
        size (int): Size for the structure element of the morphological opening.
    """

    def __init__(self):
        self.size = None
        super().__init__()

    def _generate_output(self):
        if not self.size:
            raise RuntimeError('size not defined')

        img = self.input

        orig_shape = img.shape
        if len(orig_shape) < 3:
            img = img[np.newaxis, ...]

        # background subtraction in each slice
        for z in range(img.shape[0]):
            im = np.array(img[z], dtype=np.dtype(img.dtype).newbyteorder('L'))
            im = cv2.blur(im, (3, 3))
            img[z], _ = subtract_background_rolling_ball(im, self.size)

        img.shape = orig_shape

        return self.input

filter_manager.add_filter(RollingBackgroundSubtract())


class BackgroundSubtract(FilterBase):
    """
    Subtracts provided background image from the input image using the specified
    method.

    Background removal is done z-slice by z-slice.
    Call using :meth:`filter_image` with 'BackgroundSubtract' as filter.

    Attributes:
        input (array): 2D or 3D mage to pass through filter.
        background (str or array): Background image to subtract from data. Dimensions
            must match input dimensions.
        shift_z (int): Value to shift planes along Z.
        method (str): Method to use for background subtraction. Currently, the only
            option is 'mean'.
    """

    def __init__(self):
        self.background = None
        self.shift_z = 0
        self.method = 'mean'
        super().__init__(temp_dir=True)

    def _generate_output(self):

        if self.background is None:
            raise RuntimeError('background not defined')
        else:
            self.background = io.readData(self.background)

        if self.shift_z != 0:
            shifted = tif.writeData(self.temp_dir / f'{uuid.uuid4()}.tif', self.background, returnMemmap = True)

            for z in range(self.background.shape[0]):
                try:
                    shifted[z + self.shift_z,] = self.background[z]
                except:
                    continue  # if out of bounds
            self.background = shifted

        img = self.input

        orig_shape = img.shape
        if len(orig_shape) < 3:
            img = img[np.newaxis, ...]

        if self.method == 'mean':
            img_mean = img.mean()
            bkg_mean = self.background.mean()
            ratio = img_mean / bkg_mean
            for z in range(img.shape[0]):
                bkg = (self.background[z] * ratio).astype(img.dtype)
                sub =  img[z].astype(np.int32) - bkg
                sub[sub < 0] = 0
                img[z] = sub
        else:
            raise ValueError(f'Method {self.method} not recongnized')

        img.shape = orig_shape

        return self.input


filter_manager.add_filter(BackgroundSubtract())
