import cv2
from scipy.ndimage.filters import median_filter

from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase

class Median(FilterBase):
    """ 3D median filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        size (float or tuple): Size of sliding window to calc median. if float, will apply same window to all axes.
    """

    def __init__(self):
        self.size   = None
        super().__init__()

    def _generate_output(self):
        return median_filter(self.input, self.size)


filter_manager.add_filter(Median())

class Median2D(FilterBase):
    """ Median filtr applys slice-wise. Faster than 3D implimentation.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        size (int): Size of sliding window to calc median. most be odd integer
    """

    def __init__(self):
        self.size   = 3
        super().__init__()

    def _generate_output(self):
        for z in range(self.input.shape[0]):
            self.input[z] = cv2.medianBlur(self.input[z], self.size)

        return self.input


filter_manager.add_filter(Median2D())
