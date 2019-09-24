import logging
import numpy as np

from scipy import ndimage as ndi

import clearmap3.IO as io
from clearmap3.utils.timer import Timer
from clearmap3.utils.logger import log_parameters
log = logging.getLogger(__name__)


def label_props(img, labels, centerMethod = 'Geo', intensityMethod = 'Sum', areaMethod = True):
    """Joins a list of coordiantes from distributed processing based on their IDs into a single list.
        Also. Handles duplicate ID entries bbased on
    Arguments:
        img (np.array): raw image
        labels (np.array): labeled image
        centerMethod (str or bool): 'Max' sets brightst pixel as center coordinate
        intensityMethod (str or bool): 'Sum' takes total label intensity.
        areaMethod (bool): choose whether to calculate area.
    Returns:
       array: label coordinates (list of tuples), intensities (list), sizes (list)
    """
    img = io.readData(img)
    labels = io.readData(labels)

    timer = Timer()
    log_parameters(centerMethod = centerMethod, intensityMethod = intensityMethod, areaMethod = areaMethod)

    # get label properties
    regions = region_props(labels, img)
    # get relavant properties
    coordinates = []
    intensities = []
    areas = []
    for region in regions:
        # get center coordinates
        if centerMethod == 'Max':
            coord = region.max_coord()
        elif centerMethod == 'Geo':
            coord = tuple(int(i) for i in region.centroid())
        # get label intensity
        if intensityMethod == 'Sum':
            inten = region.sum_intensity()
        # get label area
        if areaMethod:
            area = region.area()

        # add props to output
        coordinates.append(coord)
        intensities.append(inten)
        areas.append(area)

    timer.log_elapsed()

    return coordinates, intensities, areas, labels


def region_props(label_image, intensity_image=None):
    """ Measure properties of labeled image regions.
    Similar to skimage.regionprops but handles memory more efficiently by removing cacheing of arrays.

    Args:
        label_image (np.ndarray): labeled image
        intensity_image (np.ndarray): raw image

    Returns:
        (list) list of RegionProperties objects for each label
    """

    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2d and 3d images are supported.')

    if isinstance(intensity_image, np.ndarray):
        if not intensity_image.shape == label_image.shape:
            raise ValueError('Label and intensity image must have the same shape.')

    if not np.issubdtype(label_image.dtype, np.integer):
        raise TypeError('Label image must be integer type.')

    regions = []
    objects = ndi.find_objects(label_image)

    for i, sl in enumerate(objects):
        if sl is None:
            continue

        label = i + 1

        props = RegionProperties(sl, label, label_image, intensity_image)
        regions.append(props)

    return regions


class RegionProperties(object):

    def __init__(self, im_slice, label, label_image, intensity_image=None):
        """ a Region object that can be used to pull various metrics from a label in an image.

        Arguments:
            im_slice (list): slice of full image containing the label
            label (int): value of region corresponding to its label value
            label_image (np.ndarray): full labeled image
            intensity_image (np.ndarray): full intensity image
        """

        self.label = label  # int of label
        self.slice = im_slice  # bbox
        self._label_image = label_image[im_slice]
        if isinstance(intensity_image, np.ndarray):
            self._intensity_image = intensity_image[im_slice]
        else:
            self._intensity_image = None

    def area(self):
        """
        Returns:
            (np.ndarray) labeled global coordinates in [[x,y,z],...]
        """
        return np.sum(self.image())

    def centroid(self):
        """
        Returns:
            (np.ndarray) geometric center in [[x,y,z],...]
        """
        return tuple(self.coords().mean(axis=0))

    def coords(self):
        """
        Returns:
            (np.ndarray) labeled global coordinates in [[x,y,z],...]
        """
        indices = np.nonzero(self.image())
        return np.vstack([indices[i] + self.slice[i].start
                          for i in range(self._label_image.ndim)]).T

    def image(self):
        """
        Returns:
            (np.ndarray) Image masked with the correct label.
        """
        return self._label_image == self.label

    def max_intensity(self):
        return np.max(self._intensity_image[self.image()])

    def max_coord(self):
        return np.unravel_index(self._intensity_image[self.image()].argmax(), self._intensity_image.shape)

    def mean_intensity(self):
        return np.mean(self._intensity_image[self.image()])

    def min_intensity(self):
        return np.min(self._intensity_image[self.image()])

    def sum_intensity(self):
        return np.sum(self._intensity_image[self.image()])

