import numpy as np

from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase

from scipy.ndimage.filters import correlate
from bq3d.image_filters.filters.helpers.filterKernel import filterKernel

class DoG(FilterBase):
    """Convolve image with a difference of Gaussians kernel.

    Wikipedia:
        "As a feature enhancement algorithm, the difference of Gaussians can be
        utilized to increase the visibility of edges and other detail present in
        a digital image. A wide variety of alternative edge sharpening filters
        operate by enhancing high frequency detail, but because random noise also
        has a high spatial frequency, many of these sharpening filters tend to
        enhance noise, which can be an undesirable artifact. The difference of
        Gaussians algorithm removes high frequency detail that often includes
        random noise, rendering this approach one of the most suitable for
        processing images with a high degree of noise. A major drawback to
        application of the algorithm is an inherent reduction in overall image
        contrast produced by the operation."

    This filter essentially subtracts a gaussian-blured version of the image from
    a second gaussian-blurred version of the same image.

    Call using :meth:`filter_image` with 'DoG' as filter.

    Note:
        "size", "sigma", and "sigma2" must have the same number of values as
        dimensions in the original image.

    Attributes:
        input (array): Image to pass through filter.
        size (tuple): Size of the kernel.
        sigma (tuple): Sigma values for first gaussian.
        sigma2 (tuple): Sigma values for second gaussian.
    """

    def __init__(self):
        self.size   = None
        self.sigma  = None
        self.sigma2 = None

        super().__init__()

    def _generate_output(self):

        img = self.input.astype('float32')  # always convert to float for downstream processing

        orig_shape = img.shape

        if len(orig_shape) < 3:
            img = img[np.newaxis, ...]
            self.size = self.size + (self.size[-1],)
            self.sigma = self.sigma + (self.sigma[-1],)
            self.sigma2 = self.sigma2 + (self.sigma2[-1],)

        if self.size:
            fdog = filterKernel(ftype='DoG', size=self.size, sigma=self.sigma, sigma2=self.sigma2)
            fdog = fdog.astype('float32')
            img = correlate(img, fdog)
            img[img < 0] = 0

        img.shape = orig_shape

        return img

filter_manager.add_filter(DoG())

