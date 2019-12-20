import random
import logging
import numpy as np

from bq3d import io

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"

log = logging.getLogger(__name__)

def grays_to_rand_rgb(source, output):

    ftype = io.dataFileNameToType(source)

    if ftype == 'TIF':

        log.info(f'Generating random RGB imgage for {ftype}')
        data = io.readData(source).astype(int)
        max_label = int(np.max(data))
        # create lut
        lut_r = np.array(random.sample(list(range(0, max_label)), max_label))
        lut_g = np.array(random.sample(list(range(0, max_label)), max_label))
        lut_b = np.array(random.sample(list(range(0, max_label)), max_label))
        # downsample to 8bit
        lut_r = ((255 / max_label) * lut_r).astype(int)
        lut_g = ((255 / max_label) * lut_g).astype(int)
        lut_b = ((255 / max_label) * lut_b).astype(int)
        # add 0 value
        lut_r = np.insert(lut_r, 0, 0)
        lut_g = np.insert(lut_g, 0, 0)
        lut_b = np.insert(lut_b, 0, 0)

        rgb_output = np.zeros(data.shape + (3,), dtype='uint8')
        rgb_output[...,0] = lut_r[data]
        rgb_output[...,1] = lut_g[data]
        rgb_output[...,2] = lut_b[data]

        io.writeData(output, rgb_output, rgb = True)

    else:
        raise RuntimeError(f'Conversion to random RGB not supported for {ftype}')

    return output
