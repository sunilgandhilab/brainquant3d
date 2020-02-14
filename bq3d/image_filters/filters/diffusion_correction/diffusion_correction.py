import os
import cv2
import numpy as np
import tifffile as tif

from bq3d import io
from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase

from .diffuse import diffuse
from bq3d.image_filters.filters.helpers.nonzero_coords import nonzero_coords

from bq3d._version import __version__
__author__     = 'Jack Zeitoun, Ricardo Azevedo'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


class DiffusionCorr(FilterBase):

    def __init__(self):
        # Defaults
        self.mask = None
        self.iterations = 1
        self.k_constant = 1
        self.threshold = 1/65535
        super().__init__(temp_dir=True)

    def _generate_output(self):

        # Pad image by 1 pixel in each dimension for heap
        self.log.debug('Padding image.')
        padded_mask = io.empty(os.path.join(self.temp_dir, 'temp_mask.tif'),
                                         dtype=self.mask.dtype,
                                         shape=(tuple(x+2 for x in self.mask.shape)))

        output = io.empty(os.path.join(self.temp_dir, 'temp_output.tif'),
                                         dtype=np.float32,
                                         shape=padded_mask.shape)

        if padded_mask.ndim == 3:
            padded_mask[1:-1,1:-1,1:-1] = self.mask

        x_size = padded_mask.shape[-1]
        z_size = padded_mask.shape[-1]*padded_mask.shape[-2]

        # find seeds to diffuse from
        seeds = []
        for z in range(padded_mask.shape[0]):
            contours, _ = cv2.findContours(padded_mask[z],
                                           cv2.RETR_LIST,
                                           cv2.CHAIN_APPROX_NONE)

            # cv2.drawContours(output[z], contours, -1, 1, 1)
            # linear positions of seeds
            for cont in contours:
                for coord in cont:
                    seeds.append(z*z_size + coord[0][1]*x_size + coord[0][0])

        ## draw coords
        #sh = output.shape
        #output.shape = sh[0] * sh[1] * sh[2]
        #for i in seeds:
        #    output[i] = 1

        diffuse(padded_mask, seeds, output, self.iterations, self.k_constant, self.threshold)



        return padded_mask

filter_manager.add_filter(DiffusionCorr())
