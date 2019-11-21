from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase
from scipy.ndimage.filters import maximum_filter

class Max(FilterBase):
    """Calculates local maxima of an image.
    See `scipy.ndimage.maximum_filter`

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        size (float or tuple): Size of sliding window to look for max. if float, will apply same window to all axes.
    """

    def __init__(self):
        self.size   = None
        super().__init__()

    def _generate_output(self):

        return maximum_filter(self.input, size= self.size)

filter_manager.add_filter(Max())

