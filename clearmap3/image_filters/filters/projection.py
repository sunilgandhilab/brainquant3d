import cv2
import numpy as np

from clearmap3 import io
from clearmap3.image_filters import filter_manager
from clearmap3.image_filters.filter import FilterBase

class Project(FilterBase):
    """ Calculates h-maximum transform of an image. input should be unassigned ints.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        hMax (float): h parameter of h-max transform

    """

    def __init__(self):
        self.method   = 'max'
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



