from image_filters import filter_manager
from image_filters.filter import FilterBase

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


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

