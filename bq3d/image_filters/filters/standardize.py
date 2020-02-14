import numpy as np
import tifffile as tif

from pathlib import Path

from ._standardize import _standardize

from bq3d._version import __version__
__author__     = 'Jack Zeitoun, Ricardo Azevedo'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


def standardize(image, output=None):
    """Performs a simple threshold on 'image'.

    Parameters
    ----------
    image: ndarray (2-D, 3-D, ...) of integers
        Input image.
    output: ndarray
        Standardized image. If provided as argument, must be of datatype "float".
    """

    original_ndim = image.ndim
    if original_ndim == 2:
        image = image[np.newaxis, ...]

    if not isinstance(output, np.ndarray):
        pfilename = Path(image.filename)
        out_filename = pfilename.parent.joinpath(pfilename.stem + '_standardized').with_suffix(pfilename.suffix) 
        output = tif.memmap(out_filename, dtype='float32', shape=image.shape)
        original_out_ndim = original_ndim
    else:
        original_out_ndim = output.ndim
        if original_out_ndim == 2:
            output.shape = image.shape

    _standardize(image, output)

    if original_ndim == 2:
        image.shape = image.shape[1:]

    if original_out_ndim == 2:
        output.shape = output.shape[1:]

    return output
