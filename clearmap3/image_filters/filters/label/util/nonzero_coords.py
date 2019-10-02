import numpy as np

from ._nonzero_coords import _nonzero_coords

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
