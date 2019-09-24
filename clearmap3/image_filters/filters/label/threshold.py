import numpy as np

import os
import shutil
import time

from ._threshold import _threshold_3d

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

    #original_start = time.time()

    #no_output = False if isinstance(output, np.ndarray) else True

    ## Ensure image has 3 dimensions.
    #if image.ndim > 3:
    #    raise ValueError('"image" cannot have more than 3 dimensions')
    #elif image.ndim == 3:
    #    pass
    #elif image.ndim == 2:
    #    image = image[np.newaxis, :]
    #elif image.ndim == 1:
    #    image = image[np.newaxis, np.newaxis, :]

    #print('Mapping input...')
    #mmapped_image = np.memmap('image.mmap', dtype=image.dtype, shape=image.shape, mode='w+')
    #mmapped_image[:] = image

    #if no_output:
    #    mmapped_output = mmapped_image
    #else:
    #    # Ensure output has 3 dimensions.
    #    if output.ndim > 3:
    #        raise ValueError('"output" cannot have more than 3 dimensions')
    #    elif output.ndim == 3:
    #        pass
    #    elif output.ndim == 2:
    #        output = output[np.newaxis, :]
    #    elif output.ndim == 1:
    #        output = output[np.newaxis, np.newaxis, :]

    #    print('Mapping output...')
    #    mmapped_output = np.memmap('output.mmap', dtype=output.dtype, shape=output.shape, mode='w+')
    #    mmapped_output[:] = output

    #print('Thresholding...')
    #start = time.time()
    #_threshold_3d(mmapped_image, mmapped_output, val)
    _threshold_3d(image, output, val)

    #print('Returning output...')
    #if no_output:
    #    image[:] = mmapped_image
    #else:
    #    output[:] = mmapped_output
