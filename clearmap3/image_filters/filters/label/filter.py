import tifffile as tif
import numpy as np

from ._filter import _size_filter

def size_filter(image, minsize, maxsize, output, return_labels=True):
    """Filteres labeled regions in 'image' by size.

    Parameters
    ----------
    image: ndarray (2-D, 3-D, ...) labels (int)
        Image to be filtered.
    minsize: int
        Min size of labeled regions.
    maxsize: int
        Max size of labeled regions.
    output: ndarray
        Filtered image.
    return_labels: bool
        Whether or not to return the list of remaining labels.
    """

    original_ndim = image.ndim
    if original_ndim == 2:
        image = image[np.newaxis, ...]

    original_out_ndim = output.ndim
    if original_out_ndim == 2:
        output = output[np.newaxis, ...]

    ret = _size_filter(image, minsize, maxsize, output, return_labels)

    if original_ndim == 2:
        image.shape = image.shape[1:]

    if original_out_ndim == 2:
        output.shape = output.shape[1:]

    if return_labels:
        return ret
