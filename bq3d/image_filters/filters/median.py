import numpy as np
import cv2

from scipy.ndimage.filters import median_filter

from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase


class Median(FilterBase):
    """3D median filter.

    Wikipedia:
        Median filtering is one kind of smoothing technique, as is linear
        Gaussian filtering. All smoothing techniques are effective at removing
        noise in smooth patches or smooth regions of a signal, but adversely
        affect edges. Often though, at the same time as reducing the noise in a
        signal, it is important to preserve the edges. Edges are of critical
        importance to the visual appearance of images, for example. For small
        to moderate levels of Gaussian noise, the median filter is demonstrably
        better than Gaussian blur at removing noise whilst preserving edges
        for a given, fixed window size.

    See `scipy.ndimage.median_filter`

    Call using :meth:`filter_image` with 'Median' as filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        size (float or tuple): Size of sliding window to calc median. If float,
            will apply same window to all axes. Default: 3.
    """

    def __init__(self):
        self.size = 3
        super().__init__()

    def _generate_output(self):
        return median_filter(self.input, self.size)

filter_manager.add_filter(Median())


class Median2D(FilterBase):
    """Median filter applys slice-wise. Faster than 3D implimentation.

    Call using :meth:`filter_image` with 'Median2D' as filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        size (int): Size of sliding window to calc median. Must be an odd
            integer. Default: 3.
    """

    def __init__(self):
        self.size = 3
        super().__init__()

    def _generate_output(self):
        img = self.input

        import ipdb; ipdb.set_trace()

        orig_shape = img.shape
        if len(orig_shape) < 3:
            img = img[np.newaxis, ...]

        for z in range(img.shape[0]):
            img[z] = cv2.medianBlur(img[z], self.size)

        img.shape = orig_shape

        return img

filter_manager.add_filter(Median2D())
