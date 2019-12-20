from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase

from .helpers.array_manipulations import min_threshold_3d

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


class ThresholdMinimum(FilterBase):
    """ Thresholds am image setting all values below threshold to 0.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        min (float): minimum threshold.

    """

    def __init__(self):
        self.min   = 0
        super().__init__()

    def _generate_output(self):
        if len(self.input.shape) == 3:
            self.input[:] = min_threshold_3d(self.input, self.min)
            return self.input
        else:
            self.input = self.input[self.input < self.min] = 0
            return self.input


filter_manager.add_filter(ThresholdMinimum())
