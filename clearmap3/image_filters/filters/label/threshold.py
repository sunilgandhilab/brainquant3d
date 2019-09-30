import numpy as np

from ._threshold import _threshold

def threshold(image, val, output=
              ):
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

    val = image.dtype.type(val)

    # Ensure 3 dimensions
    if image.ndim == 2:
        image = image[np.newaxis, ...,]
        if isinstance(output, np.ndarray) and output.ndim == 2:
            output.shape = image.shape

    _threshold(image, output, val)

    # remove useless axis
    if image.shape[0] == 1:
        image.shape = image.shape[1:]
        output.shape = output.shape[1:]
