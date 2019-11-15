from image_filters import filter_manager
from image_filters.filter import FilterBase

class Template(FilterBase):
    """template filter object
    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        size (float or tuple): Size of sliding window to look for max. if float, will apply same window to all axes.
    """

    def __init__(self):
        self.something   = None
        super().__init__()

    def _generate_output(self):

        return None

filter_manager.add_filter(Template())

