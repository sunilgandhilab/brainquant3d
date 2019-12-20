import numpy as np

from scipy.ndimage.morphology import binary_erosion

from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase
from bq3d.image_filters.filters.helpers.filterKernel import filterKernel

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


class Erode(FilterBase):
    """Erodes image with a spherical filter.

    Call using :meth:`filter_image` with 'Erode' as filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        size (float or tuple): Size of sliding window to look for max. If
            float, will apply same window to all axes. Default: (3, 3, 3).
    """

    def __init__(self):
        self.size = (3, 3, 3)
        super().__init__()

    def _generate_output(self):
        img = self.input

        orig_shape = img.shape
        if len(orig_shape) < 3:
            img = img[np.newaxis, ...]

        struct = filterKernel(ftype='sphere', size=self.size)
        data = binary_erosion(img, structure=struct)

        img.shape = orig_shape

        return data.astype(np.uint8)


filter_manager.add_filter(Erode())

