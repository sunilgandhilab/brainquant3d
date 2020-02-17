import numpy as np
from pathlib import Path

from bq3d import io
from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase
from ._standardize import _standardize

from bq3d._version import __version__
__author__     = 'Jack Zeitoun, Ricardo Azevedo'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


class Standardize(FilterBase):
    """Applies (I-mean) / SD to standardize data

    Parameters
    ----------
    self.input: ndarray (2-D, 3-D, ...) of integers
        Input self.input.
    output: ndarray
        Standardized self.input. If provided as argument, must be of datatype "float".
    """

    def __init__(self):
        super().__init__()

    def _generate_output(self):
        original_ndim = original_out_ndim = self.input.ndim
        if original_ndim == 2:
            self.input = self.input[np.newaxis, ...]

        _standardize(self.input, self.input)

        if original_ndim == 2:
            self.input.shape = self.input.shape[1:]

        return self.input


filter_manager.add_filter(Standardize())
