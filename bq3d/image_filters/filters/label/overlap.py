import numpy as np

from ._overlap import _overlap

from bq3d._version import __version__
__author__     = 'Jack Zeitoun, Ricardo Azevedo'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


def overlap(label_0, label_1, output):
    """Compares two labeled images and keeps labeled regions that overlap.

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

    original_label_0_ndim = label_0.ndim
    if original_label_0_ndim == 2:
        label_0 = label_0[np.newaxis, ...]

    original_label_1_ndim = label_1.ndim
    if original_label_1_ndim == 2:
        label_1 = label_1[np.newaxis, ...]

    original_out_ndim = output.ndim
    if original_out_ndim == 2:
        output = output[np.newaxis, ...]

    _overlap(label_0, label_1, output)

    if original_label_0_ndim == 2:
        label_0.shape = label_0.shape[1:]

    if original_label_1_ndim == 2:
        label_1.shape = label_1.shape[1:]

    if original_out_ndim == 2:
        output.shape = output.shape[1:]
