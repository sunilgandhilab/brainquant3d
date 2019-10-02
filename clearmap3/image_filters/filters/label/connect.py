import tifffile as tif
import cv2

from ._connect import _connect

def connect(image, output):
    """
    Identify connected components in thresholded 'image'.

    Parameters
    ----------
    image: ndarray (2-D, 3-D, ...)
        Binary image.
    output: ndarray
        Matrix where values >= 'val' = max(dtype) and values < 'va' = 0.
    """

    if image.ndim == 2:
        _, markers = cv2.connectedComponents(image)
        output[:] = markers
    else:
        _connect(image, output)
