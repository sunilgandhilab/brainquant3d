from clearmap3.image_filters import filter_manager
from clearmap3.image_filters.filter import FilterBase

from scipy.ndimage.filters import correlate
from  clearmap3.image_filters.filters.helpers.filterKernel import filterKernel

class DoG(FilterBase):
    """Remove background via subtracting a morphological opening from the original image
    Background removal is done z-slice by z-slice.
    Call using :meth:`filter_image` with 'background_subtract' as filter.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        size (tuple): Size for the structure element of the morphological opening.
    """

    def __init__(self):
        self.size   = None
        self.sigma  = None
        self.sigma2 = None

        super().__init__()

    def _generate_output(self):

        img = self.input.astype('float32')  # always convert to float for downstream processing

        if self.size:
            fdog = filterKernel(ftype='DoG', size=self.size, sigma=self.sigma, sigma2=self.sigma2)
            fdog = fdog.astype('float32')
            img = correlate(img, fdog)
            img[img < 0] = 0

        return img

filter_manager.add_filter(DoG())

