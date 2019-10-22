import numpy as np

from ._threshold import _threshold

def threshold(image, val, output=None):
    """Performs a simple threshold on 'image'.

    Parameters
    ----------
    image: ndarray (2-D, 3-D, ...) of integers
        Image to be thresholded.
    val: integer or float
        Value to be used as the threshold.
    output: ndarray
        Binary matrix where values >= 'val' = max(dtype) and values < 'va' = 0.
    """

    original_ndim = image.ndim
    if original_ndim == 2:
        image = image[np.newaxis, ...]

    if not isinstance(output, np.ndarray):
        output = image
        original_out_ndim = output.ndim
    else:
        original_out_ndim = output.ndim
        if original_out_ndim == 2:
            output.shape = image.shape

    val = image.dtype.type(val)

    _threshold(image, output, val)

    if original_ndim == 2:
        image.shape = image.shape[1:]

    if original_out_ndim == 2:
        output.shape = output.shape[1:]
