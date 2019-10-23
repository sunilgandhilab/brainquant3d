import numpy as np

from ._watershed import _watershed


def watershed(raw_image, marker_locations, flat_neighborhood, mask_image, image_strides,
              compactness, output_image, wsl=False, invert=False):


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
              compactness, output_image, wsl=False, invert=False)

    if original_ndim == 2:
        raw_image.shape = raw_image.shape[1:]

    if original_mask_ndim == 2:
        mask_image.shape = mask_image.shape[1:]

    if original_out_ndim == 2:
        output_image.shape = output_image.shape[1:]
