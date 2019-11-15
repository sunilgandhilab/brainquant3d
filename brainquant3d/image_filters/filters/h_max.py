from clearmap3.image_filters import filter_manager
from clearmap3.image_filters.filter import FilterBase

from clearmap3.image_filters.filters.helpers.greyReconstruction import reconstruct

class HMax(FilterBase):
    """ Calculates h-maximum transform of an image.

    Attributes:
        input (array): Image to pass through filter.
        output (array): Filter result.

        hMax (float): h parameter of h-max transform

    """

    def __init__(self):
        self.hMax   = None
        super().__init__()

    def _generate_output(self):
            return reconstruct(self.input - self.hMax, self.input)



filter_manager.add_filter(HMax())


