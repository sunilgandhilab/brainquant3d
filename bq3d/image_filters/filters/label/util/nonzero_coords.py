import numpy as np

from ._nonzero_coords import _nonzero_coords

from bq3d._version import __version__
__author__     = 'Jack Zeitoun, Ricardo Azevedo'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


def nonzero_coords(image, filename):
    """ Generates linear indices of all non zero values in array.

    Parameters
    ----------
    image: ndarray (2-D, 3-D, ...) labels (int)
        Image to be filtered.
    filename: str
        Filename for coordinates output.
    """

    original_ndim = image.ndim
    if original_ndim == 2:
        image = image[np.newaxis, ...]

    ret = _nonzero_coords(image, filename)

    if original_ndim == 2:
        image.shape = image.shape[1:]

    return ret
