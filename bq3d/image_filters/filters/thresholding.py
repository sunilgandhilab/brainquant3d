from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase

from .helpers.array_manipulations import min_threshold_3d


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
