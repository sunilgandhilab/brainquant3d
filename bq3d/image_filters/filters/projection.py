import cv2
import numpy as np

from bq3d import io
from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


class Project(FilterBase):
    """Generates a projection along the Z-axis using the specified method.

    Call using :meth:`filter_image` with 'Max' as filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        method (str): Method to use for projection. "max" or "min".
        mask (array): Optional binary image to use as a mask.
    """

    def __init__(self):
        self.method = 'max'
        self.mask = None
        super().__init__()

    def _generate_output(self):
        # extract brain mask
        method = self.method.lower()
        if self.mask:
            mask = io.readData(self.mask)

        if method == 'max':
            sink = np.zeros(self.input.shape[1:], dtype=self.input.dtype)
            if self.mask:
                for z in range(self.input.shape[0]):
                    slice  = np.array(self.input[z])
                    mask_s = np.array(mask[z])
                    slice[mask_s == 0] = 0

                    sink = cv2.max(sink, slice)
            else:
                for z in range(self.input.shape[0]):
                    sink = cv2.max(sink, self.input[z])

            return sink

        if method == 'min':
            max_v = np.iinfo(self.input.dtype).max
            sink = np.full(self.input.shape[1:], max_v, dtype=self.input.dtype)
            if self.mask:
                for z in range(self.input.shape[0]):
                    slice  = np.array(self.input[z])
                    mask_s = np.array(mask[z])

                    slice[mask_s == 0] = max_v

                    sink = cv2.min(sink, slice)
            else:
                for z in range(self.input.shape[0]):
                    sink = cv2.min(sink, self.input[z])
                sink = sink

            sink[sink == max_v] = 0
            return sink

filter_manager.add_filter(Project())
