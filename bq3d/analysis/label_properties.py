import logging
import math
from typing import Union, List
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage as ndi
from skimage import measure

from bq3d import io
from bq3d.utils.timer import Timer
from bq3d.utils.logger import log_parameters

from bq3d._version import __version__

__author__ = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__ = "Copyright 2019, Gandhi Lab"
__license__ = 'BY-NC-SA 4.0'
__version__ = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__ = 'ricardo-re-azevedo@gmail.com'
__status__ = "Development"

log = logging.getLogger(__name__)


def label_props(img, labels: Union[str, Path, np.array], props: Union[str, Path, np.array]):
    """Joins a list of coordiantes from distributed processing based on their IDs into a single
    list.
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
            try:  # JZ DEBUG
                prop_res.append(method())
            except Exception as e:
                log.error(e, exc_info=True)

        res.append(prop_res)

    timer.log_elapsed()

    return res


def region_props(label_image: np.array, intensity_image: np.array = None):
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

        try:
            props = RegionProperties(sl, label, label_image, intensity_image)
        except Exception as e:
            log.error(e, exc_info=True)

        regions.append(props)

    log.info(f'Objects Detected: {len(regions)}')
    return regions


class RegionProperties(object):

    def __init__(self, im_slice: tuple, label: Union[int, list], label_image, intensity_image=None):
        """ a Region object that can be used to pull various metrics from a label in an image.
        Arguments:
            im_slice (list): slice of full image containing the label
            label (int): value of region corresponding to its label value
            label_image (np.ndarray): full labeled image
            intensity_image (np.ndarray): full intensity image
        """

        self.label = label  # int of label
        self.slice = im_slice  # bbox
        self.origin = tuple(s.start for s in self.slice)
        self.full_label_image = label_image
        self.full_intensity_image = intensity_image

        self.label_image = label_image[im_slice]
        if isinstance(intensity_image, np.ndarray):
            self.intensity_image = intensity_image[im_slice]
        else:
            self.intensity_image = None

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

    def moments(self) -> np.array:
        """
        Returns:
            (np.array) Spatial moments up to 3rd order:
            m_ij = sum{ array(row, col) * row^i * col^j }
            where the sum is over the row, col coordinates of the region.
        """
        m = measure._moments.moments(self.image().astype(np.uint8), 3)
        return m

    def local_centroid(self):
        """
        Returns:
            (np.ndarray) geometric center in [x,y,z] relative to bounding box
        """
        m = self.moments
        return tuple(m[tuple(np.eye(self.label_image.ndim, dtype=int))] /
                     m[(0,) * self.label_image.ndim])

    def local_coords(self):
        """
        Returns:
            (np.ndarray) labeled global coordinates in [[x,y,z],...] relative to bounding box
        """
        return np.nonzero(self.image())

    def coords(self):
        """
        Returns:
            (np.ndarray) labeled global coordinates in [[x,y,z],...]
        """
        indices = self.local_coords()
        return np.vstack([indices[i] + self.slice[i].start
                          for i in range(self.label_image.ndim)]).T

    def image(self):
        """
        Returns:
            (np.ndarray) Image masked with the correct label.
        """
        if isinstance(self.label, list):
            return np.isin(self.label_image, self.label)
        return self.label_image == self.label

    def surface(self, approx_method=cv2.CHAIN_APPROX_NONE):
        """ get coordinates of surface.
        Returns:
            (np.ndarray) coordinates belonging to region surface in [[x,y,z],...].
        """
        im = self.image().astype(np.uint8)
        ndim = im.ndim
        # probably can make higher performance version
        contours = []
        if ndim == 2:
            cont, _ = cv2.findContours(im.T, cv2.RETR_LIST, approx_method)
            return np.vstack(cont) + self.origin
        if ndim == 3:
            for z in range(im.shape[0]):
                cont, _ = cv2.findContours(im[z].T, cv2.RETR_LIST, approx_method)
                if len(cont) == 0:
                    continue
                join_contours = np.vstack(cont)
                join_contours = np.insert(join_contours, 0, z, axis=2)
                contours.append(join_contours)
            contours = np.vstack(contours).squeeze()
            return contours + self.origin

        else:
            raise ValueError(f'image of ndim {ndim} not supported')

    def surface_z(self, z, approx_method=cv2.CHAIN_APPROX_NONE):
        """ get coordinates of at some z.
        Returns:
            (list) List of contours. with each contour in format [[y,z],...].
        """

        im = self.image().astype(np.uint8)
        local_z = z - self.origin[0]

        if im.ndim != 3:
            ValueError(f'Method only supports 3d images')
        cont, _ = cv2.findContours(im[local_z].T, cv2.RETR_LIST, approx_method)
        abs_cont_yx = [c.squeeze() + self.origin[1:] for c in cont]
        return abs_cont_yx

    def surface_area(self):
        """
        Returns:
            (float): total serfice area of region.
        """
        ndim = self.label_image.ndim
        if ndim == 2:
            return measure.perimeter(self.label_image, neighbourhood=8)
        elif ndim == 3:
            peri = 0
            im = self.image()
            for z in range(self.label_image.shape[0]):
                peri += measure.perimeter(im[z], neighbourhood=8)
            return peri
        else:
            raise ValueError(f'RegionProperties.surface area does not support ndim {ndim}')

    def svr(self):
        """
        Returns:
            (float): surface are to volume ratio. region surface area / volume""
        """
        return self.surface_area() / self.area()

    def moments_central(self):
        """
        Returns:
            (np.ndarray) Central moments (translation invariant) up to 3rd order:
             mu_ij = sum{ array(row, col) * (row - row_c)^i * (col - col_c)^j }
             where the sum is over the row, col coordinates of the region, and row_c
             and col_c are the coordinates of the regions centroid.
        """
        mu = measure._moments.moments_central(self.image().astype(np.uint8),
                                              self.local_centroid(), order=3)
        return mu

    def inertia_tensor(self):
        """
        Returns:
            (np.ndarray) Inertia tensor of the region for the rotation around its mass.
        """
        mu = self.moments_central
        return measure._moments.inertia_tensor(self.image(), mu)

    def inertia_tensor_eigvals(self):
        """
        Returns:
            (np.ndarray) The eigenvalues of the inertia tensor in decreasing order.
        """
        return measure._moments.inertia_tensor_eigvals(self.image(),
                                                       T=self.inertia_tensor())

    def compactness(self):
        """
        Returns:
            (float): ratio of object area to area of circle/sphere with same perimeter.
        """

        ndim = self.label_image.ndim
        if ndim == 2:
            return (4 * math.pi * self.area()) / (self.surface_area() ** 2)
        elif ndim == 3:
            return self.area() / ((self.surface_area() ** (3 / 2)) * 0.09403)
        else:
            raise ValueError(f'RegionProperties.surface area does not support ndim {ndim}')

    def max(self):
        return np.max(self.intensity_image[self.image()])

    def max_coord(self):
        return np.unravel_index(self.intensity_image[self.image()].argmax(),
                                self.intensity_image.shape)

    def mean(self):
        return np.mean(self.intensity_image[self.image()])

    def min(self):
        return np.min(self.intensity_image[self.image()])

    def sum(self):
        return np.sum(self.intensity_image[self.image()])


def join_regions(regions: List[RegionProperties]) -> RegionProperties:
    """ Joins a list of RegionProperties into a larger region. All must have the same label_image
    and intensioty_image.
    Args:
        regions: regions to join
    Returns:
        RegionProperties instance for the grouped regions.
    """

    # im_slice (list): slice of full image containing the label
    # label (int): value of region corresponding to its label value
    # label_image (np.ndarray): full labeled image
    # intensity_image (np.ndarray): full intensity image

    slices = []
    for s in range(len(regions[0].slice)):
        start = min([r.slice[s].start for r in regions])
        stop = max([r.slice[s].stop for r in regions])
        slices.append(slice(start, stop))
    slices = tuple(slices)

    labels = []
    for r in regions:
        if isinstance(r.label, list):
            labels.extend(r.label)
        else:
            labels.append(r.label)

    return RegionProperties(slices, labels, regions[0].full_label_image,
                            intensity_image=regions[0].full_intensity_image)
