import numpy as np

from scipy.ndimage.morphology import binary_erosion

from clearmap3.image_filters import filter_manager
from clearmap3.image_filters.filter import FilterBase
from  clearmap3.image_filters.filters.helpers.filterKernel import filterKernel


class Erode(FilterBase):
    """Erodes image with a spherical filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.
        size (tuple): diameter of erosion opening.

        size (float or tuple): Size of sliding window to look for max. if float, will apply same window to all axes.
    """

    def __init__(self):
        self.size   = (3,3,3)
        super().__init__()

    def _generate_output(self):
        struct = filterKernel(ftype='sphere', size=self.size)
        data = binary_erosion(self.input, structure = struct)
        return data.astype(np.uint8)


filter_manager.add_filter(Erode())

