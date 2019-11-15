import logging
import numpy as np

from scipy import ndimage as ndi

from bq3d import io
from bq3d.utils.timer import Timer
from bq3d.utils.logger import log_parameters
log = logging.getLogger(__name__)


def label_props(img, labels, props):
    """Joins a list of coordiantes from distributed processing based on their IDs into a single list.
        Also. Handles duplicate ID entries bbased on
    Arguments:
        img (np.array): raw image
        labels (np.array): labeled image
        props (list): list of strings where each string is the attibute from region_props to export
    Returns:
       array: label coordinates (list of tuples), intensities (list), sizes (list)
    """

    img = io.readData(img)
    labels = io.readData(labels)

    timer = Timer()
    log_parameters(props=props)

    # get label properties
    regions = region_props(labels, img)

    # get relavant properties
    res = []
    for prop in props:
        prop_res = []
        for region in regions:
            method = getattr(region, prop)
            prop_res.append(method())
        res.append(prop_res)

    timer.log_elapsed()

    return res


def region_props(label_image, intensity_image=None):
    """ Measure properties of labeled image regions.
    Similar to skimage.regionprops but more memory efficient by not cacheing of arrays.

    Args:
        label_image (np.ndarray): labeled image
        intensity_image (np.ndarray): raw image

    Returns:
        (list) list of RegionProperties objects for each label
    """

    if label_image.ndim not in (2, 3):
        raise TypeError('Only 2d and 3d images are supported.')

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

    def max(self):
        return np.max(self._intensity_image[self.image()])

    def max_coord(self):
        return np.unravel_index(self._intensity_image[self.image()].argmax(), self._intensity_image.shape)

    def mean(self):
        return np.mean(self._intensity_image[self.image()])

    def min(self):
        return np.min(self._intensity_image[self.image()])

    def sum(self):
        return np.sum(self._intensity_image[self.image()])

