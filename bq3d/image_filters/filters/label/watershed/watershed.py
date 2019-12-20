import numpy as np

from ._watershed import _watershed

from bq3d._version import __version__
__author__     = 'Jack Zeitoun, Ricardo Azevedo'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


def watershed(raw_image, marker_locations, flat_neighborhood, mask_image, image_strides,
              output_image, invert=False):


    original_ndim = raw_image.ndim
    if original_ndim == 2:
        raw_image = raw_image[np.newaxis, ...]

    original_mask_ndim = mask_image.ndim
    if original_mask_ndim == 2:
        mask_image.shape = raw_image.shape

    original_out_ndim = output_image.ndim
    if original_out_ndim == 2:
        output_image.shape = raw_image.shape

    _watershed(raw_image, marker_locations, flat_neighborhood, mask_image, image_strides,
              output_image, invert=invert)

    if original_ndim == 2:
        raw_image.shape = raw_image.shape[1:]

    if original_mask_ndim == 2:
        mask_image.shape = mask_image.shape[1:]

    if original_out_ndim == 2:
        output_image.shape = output_image.shape[1:]
