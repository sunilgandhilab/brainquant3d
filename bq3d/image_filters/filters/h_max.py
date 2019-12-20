import numpy as np

from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase

from bq3d.image_filters.filters.helpers.greyReconstruction import reconstruct

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


class HMax(FilterBase):
    """Calculates h-maximum transform of an image.

    Wikipedia:
        The h-maxima transform is a morphological operation used to filter local
        maxima of an image based on local contrast information. First all local
        maxima are defined as connected pixels in a given neighborhood with
        intensity level greater than pixels outside the neighborhood. Second,
        all local maxima that have height f lower or equal to
        a given threshold are suppressed. The height f of the remaining maxima
        is decreased by h.

    Call using :meth:`filter_image` with 'HMax' as filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        hMax (float): h parameter of h-max transform

    """

    def __init__(self):
        self.hMax = None
        super().__init__()

    def _generate_output(self):
        img = self.input

        orig_shape = img.shape
        if len(orig_shape) < 3:
            img = img[np.newaxis, ...]

        seed = img.copy()
        seed[seed >= self.hMax] = seed[seed >= self.hMax] - self.hMax
        res = reconstruct(seed, img)

        res.shape = orig_shape

        return res


filter_manager.add_filter(HMax())


