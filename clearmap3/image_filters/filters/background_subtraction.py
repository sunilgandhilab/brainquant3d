import cv2
import numpy as np

import uuid
from clearmap3 import io
import clearmap3.io.TIF as tif
from clearmap3.image_filters import filter_manager
from clearmap3.image_filters.filter import FilterBase
from clearmap3.image_filters.filters._background_subtraction import subtract_background_rolling_ball


class RollingBackgroundSubtract(FilterBase):
    """Remove background via subtracting a morphological opening from the original image
    Background removal is done z-slice by z-slice.
    Call using :meth:`filter_image` with 'background_subtract' as filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        size (int): Size for the structure element of the morphological opening.
    """

    def __init__(self):
        self.size     = None
        super().__init__()

    def _generate_output(self):
        if not self.size:
            raise RuntimeError('size not defined')

        img = self.input

        # background subtraction in each slice
        for z in range(img.shape[0]):
            im = np.array(img[z], dtype=np.dtype(img.dtype).newbyteorder('L'))
            im = cv2.blur(im, (3, 3))
            img[z], _ = subtract_background_rolling_ball(im, self.size)

        return self.input

filter_manager.add_filter(RollingBackgroundSubtract())


class BackgroundSubtract(FilterBase):
    """Remove background via subtracting a morphological opening from the original image
    Background removal is done z-slice by z-slice.
    Call using :meth:`filter_image` with 'background_subtract' as filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.
        background (str or array): background image to subtract from data.

        size (tuple): Size for the structure element of the morphological opening.
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

        if self.method == 'mean':
            img_mean = self.input.mean()
            bkg_mean = self.background.mean()
            ratio = img_mean / bkg_mean
            for z in range(self.input.shape[2]):
                bkg = (self.background[...,z] * ratio).astype(self.input.dtype)
                sub =  self.input[...,z].astype(np.int32) - bkg
                sub[sub < 0] = 0
                self.input[...,z] = sub
        else:
            raise ValueError(f'Method {self.method} not recongnized')

        return self.input


filter_manager.add_filter(BackgroundSubtract())
