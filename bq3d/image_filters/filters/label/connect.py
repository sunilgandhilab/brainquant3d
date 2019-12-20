import numpy as np
import cv2

from ._connect import _connect

from bq3d._version import __version__
__author__     = 'Jack Zeitoun, Ricardo Azevedo'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


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
