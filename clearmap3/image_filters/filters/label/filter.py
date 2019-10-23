import numpy as np

from ._filter import _size_filter, _label_by_size

def size_filter(image, minsize, maxsize, output):
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

    n_labels_in, counts = _size_filter(image, minsize, maxsize, output)

    if original_ndim == 2:
        image.shape = image.shape[1:]

    if original_out_ndim == 2:
        output.shape = output.shape[1:]

    return n_labels_in, counts

def label_by_size(image,  output):
    """hanges the value of all labels in a labeled image to their volume. Good for determining
    size thresholds.

    Parameters
    ----------
    image: ndarray (2-D, 3-D, ...) labels (int)
        Image to be filtered.
    output: ndarray
        Filtered image.
    """

    original_ndim = image.ndim
    if original_ndim == 2:
        image = image[np.newaxis, ...]

    original_out_ndim = output.ndim
    if original_out_ndim == 2:
        output = output[np.newaxis, ...]

    _label_by_size(image, output)

    if original_ndim == 2:
        image.shape = image.shape[1:]

    if original_out_ndim == 2:
        output.shape = output.shape[1:]

    return output