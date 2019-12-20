"""
Converts point data into voxel image data for visulaization and analysis
"""

import numpy as np
import math

from bq3d import io
import bq3d.analysis._voxelization as vox
from bq3d.analysis.colocalization import point_distance

import logging

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"

log = logging.getLogger(__name__)


def voxelize(points, dataSize = None, sink = None, method = 'Spherical', size = (5,5,5), weights = None):
    """Converts a list of points into an volumetric image array

    Arguments:
        points (array): point data array
        dataSize (tuple or str): size of final image in xyz. If str, will use the size of the passed image.
        sink (str, array or None): the location to write or return the resulting voxelization image, if None return array
        method (str or None): method for voxelization: 'Spherical', 'Rectangular' or 'Pixel'
        size (tuple): size parameter for the voxelization
        weights (array or None): weights for each point, None is uniform weights
    Returns:
        (array): volumetric data of smeared out points
    """

    log.verbose('voxelizing points')
    points = io.readPoints(points)

    if dataSize is None:
        dataSize = tuple(int(math.ceil(points[:,i].max())) for i in range(points.shape[1]))
    elif isinstance(dataSize, str):
        dataSize = io.dataSize(dataSize)

    if method.lower() == 'spherical':
        if weights is None:
            data = vox.voxelizeSphere(points.astype('float'), dataSize[0], dataSize[1], dataSize[2], size[0], size[1], size[2])
        else:
            data = vox.voxelizeSphereWithWeights(points.astype('float'), dataSize[0], dataSize[1], dataSize[2], size[0], size[1], size[2], weights)

    elif method.lower() == 'rectangular':
        if weights is None:
            data = vox.voxelizeRectangle(points.astype('float'), dataSize[0], dataSize[1], dataSize[2], size[0], size[1], size[2])
        else:
            data = vox.voxelizeRectangleWithWeights(points.astype('float'), dataSize[0], dataSize[1], dataSize[2], size[0], size[1], size[2], weights)

    elif method.lower() == 'pixel':
        data = voxelizePixel(points, dataSize, weights)
    else:
        raise RuntimeError('voxelize: mode: %s not supported!' % method)

    if data.dtype == np.float64:
        log.warning('Converting dtype float64 to int32 for output. This may result in loss of info.')
        data = data.astype('int32')

    if sink:
        return io.writeData(sink, data, returnMemmap = True)
    else:
        return data


def voxelizePixel(points,  dataSize = None, weights = None):
    """Mark pixels/voxels of each point in an image array
    
    Arguments:
        points (array): point data array
        dataSize (tuple or None): size of the final output data, if None size is determined by maximal point coordinates
        weights (array or None): weights for each points, if None weights are all 1s.
    
    Returns:
        (array): volumetric data with with points marked in voxels
    """
    
    if dataSize is None:
        dataSize = tuple(int(math.ceil(points[:,i].max())) for i in range(points.shape[1]))
    elif isinstance(dataSize, str):
        dataSize = io.dataSize(dataSize)

    points = np.rint(points).astype('int64')

    if weights is None:
        vox = np.zeros(dataSize, dtype=np.int16)
        for i in range(points.shape[0]):
            if 0 < points[i,0] < dataSize[0] and 0 < points[i,1] < dataSize[1] and 0 < points[i,2] < dataSize[2]:
                vox[points[i,0], points[i,1], points[i,2]] += 1

    elif isinstance(weights, int):
        vox = np.zeros(dataSize, dtype=np.int16) # TODO: dtype should depend on weight value passed
        for i in range(points.shape[0]):
            if 0 < points[i,0] < dataSize[0] and 0 < points[i,1] < dataSize[1] and 0 < points[i,2] < dataSize[2]:
                vox[points[i,0], points[i,1], points[i,2]] += weights

    elif isinstance(weights, np.ndarray):
        vox = np.zeros(dataSize, dtype=weights.dtype)
        for i in range(points.shape[0]):
            if 0 < points[i,0] < dataSize[0] and 0 < points[i,1] < dataSize[1] and 0 < points[i,2] < dataSize[2]:
                vox[points[i,0], points[i,1], points[i,2]] += weights[i]
    else:
        RuntimeError('VoxelizePixel: only Bool, Int, and arrays are valid weight values.')
    return vox

def distance_map(image:np.ndarray, coords:np.ndarray):
    """ Generates a map with size of image with the minimum distance to a coordinate.
    Ignores pixels in image with value 0.

    Args:
        image: (np.ndarray) image used to determine map size
        coords: (np.ndarray) coordinates to calculate distance from in form [[x,y,z],...]

    Returns:
        np.array: map where each value is the minimum distance to a coordinate. returns 32bit float.
    """

    dmap = np.zeros(image.shape, dtype = np.float32)
    nonzero_vox = np.array(np.nonzero(image)).T

    dists = point_distance(nonzero_vox, coords)

    for i, pt in enumerate(nonzero_vox):
        dmap[tuple(pt)] = dists[i]

    return dmap


def labels_to_coords(label_im:np.array):
    """ Converts each label in an image to a set of coordinates

    Args:
        label_im (np.array): labeled image

    Returns:
        (dict) dict of coordinated in format { 'label_id': [[x,y,z],...] }
    """
    im = io.readData(label_im)

    region_coords = {}
    for z in range(im.shape[2]):
        print(z)
        regions = regionprops(im[..., z])
        for reg in regions:
            coord = list(np.insert(reg.coords(), 2, z, 1))

            if reg.label in region_coords:
                region_coords[reg.label].extend(coord)
            else:
                region_coords[reg.label] = coord

    for id in region_coords:
        region_coords[id] = np.array(region_coords[id])

    return region_coords
