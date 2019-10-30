import os
import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile as tif

import clearmap3.IO as io
from clearmap3.image_filters import filter_manager
from clearmap3.image_filters.filter import FilterBase

from .connect import connect
from .threshold import threshold
from .filter import size_filter, label_by_size
from .overlap import overlap
from .util.nonzero_coords import nonzero_coords
from .watershed.watershed import watershed
from .watershed._util import _validate_connectivity, _offsets_to_raveled_neighbors

class Label(FilterBase):
    """thresholds image using Ilastik's threshold applet

    Attributes:
         input          (array): Image to pass through filter, munt be memmapped.
         output         (array): Filter result.
         sigmas         (tuple or None): Sigma values (x, y, z) for smoothing finction.
         min_size       (float): Minimum object size during filtering.
         max_size       (float): Maximum object size during filtering.
         min_size2      (float): Minimum object size during optional second filtering.
         max_size2      (float): Maximum object size during optional second filtering.
         high_threshold (float): Probability threshold for filtering.
         low_threshold  (float): Probability threshold for optional second filtering.
         mode           (int): Options are:
                                1 : High Threshold --> Label --> Size Filter
                                3 : Mode 1 --> Low Thresh --> Label -->
                                      Compare with size filtered and keep overlap --> Size Filter (2nd Pass)
                                2 : Mode 1 --> Low Thresh --> Watershed --> Size Filter (2nd Pass)
    """

    def __init__(self):
        # Defaults
        self.sigmas = None
        self.min_size = 10
        self.max_size = 2147483647 # max int value
        self.min_size2 = 10
        self.max_size2 = 2147483647
        self.high_threshold = .7
        self.low_threshold = .2
        self.mode = 2
        super().__init__(temp_dir=True)

    def _generate_output(self):
        # label return count

        if self.sigmas:
            if self.input.ndim != len(self.sigmas):
                raise ValueError('Sigmas must have same length as image dimensions.')

            # Smooth image
            self.log.verbose('Smoothing image.')
            self.input[:] = gaussian_filter(self.input, sigma=self.sigmas)

        raw_img = self.input

        if self.mode == 3:
            # Pad image by 1 pixel in each dimension
            print('Padding image...')
            padded_img = tif.tifffile.memmap(os.path.join(self.temp_dir, 'temp_padded_img.tif'),
                                             dtype=raw_img.dtype,
                                             shape=(tuple(x+2 for x in raw_img.shape)))
            if raw_img.ndim == 3:
                padded_img[1:-1,1:-1,1:-1] = raw_img
            if raw_img.ndim == 2:
                padded_img[1:-1,1:-1] = raw_img
            raw_img = padded_img

        bin_img = tif.tifffile.memmap(os.path.join(self.temp_dir, 'temp_bin_img.tif'),
                                      dtype=np.uint8,
                                      shape=raw_img.shape)

        labeled_1_img = tif.tifffile.memmap(os.path.join(self.temp_dir, 'temp_labeled_1_img.tif'),
                                            dtype=np.int32,
                                            shape=raw_img.shape)

        # Binarize image
        self.log.debug('Thresholding')
        threshold(raw_img, self.high_threshold, bin_img)

        # Label image
        self.log.debug('Labeling')
        connect(bin_img, labeled_1_img)

        # Filter labeled regions by size (1st pass) # Mode 1: Stop after this
        self.log.debug('Size filtering')
        _, _ = size_filter(labeled_1_img, self.min_size, self.max_size, labeled_1_img)

        if self.mode == 1:
            return io.readData(labeled_1_img.filename)

        # Mode 2 two serial thresholding>label> filter runs
        elif self.mode == 2:

            labeled_2_img = tif.tifffile.memmap(os.path.join(self.temp_dir, 'temp_labeled_2_img.tif'),
                                                    dtype=np.int32,
                                                    shape=raw_img.shape)

            self.log.debug('Low thresholding')
            threshold(raw_img, self.low_threshold, bin_img)

            self.log.debug('Labeling...')
            _ = connect(bin_img, labeled_2_img)

            self.log.debug('Comparing overlap...')
            overlap(labeled_1_img, labeled_2_img, labeled_2_img)

            self.log.debug('Running final size filter...')
            _, _ = size_filter(labeled_2_img, self.min_size2, self.max_size2, labeled_2_img)

            return io.readData(labeled_1_img.filename)

        # Mode 3 two serial thresholds with identity preservation
        elif self.mode == 3:

            # Low threshold Image
            self.log.debug('Low thresholding...')
            threshold(raw_img, self.low_threshold, bin_img)

            ############################## Watershed ##############################

            # Get coordinates of all nonzero values in labeled/size-filtered image
            self.log.debug('Getting label coordinates...')
            marker_locations_filename = os.path.join(self.temp_dir, 'marker_locations.mmap')
            marker_locations = nonzero_coords(labeled_1_img, marker_locations_filename)

            connectivity, offset = _validate_connectivity(raw_img.ndim, connectivity=None,
                                                          offset=None)

            flat_neighborhood = _offsets_to_raveled_neighbors(
                raw_img.shape, connectivity, center=offset)

            image_strides = np.array(raw_img.strides, dtype=np.intp) // raw_img.itemsize

            self.log.debug('Running watershed...')
            watershed(raw_img, marker_locations, flat_neighborhood,
                                 bin_img, image_strides,
                                 labeled_1_img, # <-- Output
                                 True) # <-- Inverted watershed

            #######################################################################

            # Final size filter
            self.log.debug('Running final size filter...')
            _, _ = size_filter(labeled_1_img, self.min_size2, self.max_size2, labeled_1_img)

            if self.input.ndim == 2:
                labeled_1_img = labeled_1_img[1:-1,1:-1]
            else:
                labeled_1_img = labeled_1_img[1:-1,1:-1,1:-1]

            out = io.empty(os.path.join(self.temp_dir, 'output.tif'),
                           shape=labeled_1_img.shape,
                           dtype=labeled_1_img.dtype)
            out[:] = labeled_1_img

            return out

class LabelBySize(FilterBase):
    """Changes the value of all labels in a labeled image to their volume. Good for determining
    size thresholds.

    Attributes:
         input          (array): Labeled Image to pass through filter, munt be memmapped.
         output         (array): Filter result.
    """

    def __init__(self):
        super().__init__(temp_dir=True)

    def _generate_output(self):
        return label_by_size(self.input, self.input)

filter_manager.add_filter(Label())
filter_manager.add_filter(LabelBySize())
