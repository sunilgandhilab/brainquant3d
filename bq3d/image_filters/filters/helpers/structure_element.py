# -*- coding: utf-8 -*-
"""
Routines to generate various structure elements


Structured elements defined by the ``setype`` key include: 

.. _StructureElementTypes:

Structure Element Types
-----------------------

=============== =====================================
Type            Descrition
=============== =====================================
``sphere``      Sphere structure
``disk``        Disk structure
=============== =====================================

Note:
    To be extended!

"""


import numpy as np
from scipy import ndimage as ndi
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

def structure_element(setype = 'Disk', sesize = (3,3)):
    """Creates specific 2d and 3d structuring elements
      
    Arguments:
        setype (str): structure element type, see :ref:`StructureElementTypes`
        sesize (array or tuple): size of the structure element
    
    Returns:
        array: structure element
    """
    
    ndim = len(sesize)
    if ndim == 2:
        return structure_element_2d(setype, sesize)
    else:
        return structure_element_3d(setype, sesize)


def structure_element_offsets(sesize):
    """Calculates offsets for a structural element given its size
    
    Arguments:
        sesize (array or tuple): size of the structure element
    
    Returns:
        array: offsets to center taking care of even/odd ummber of elements
    """
    
    sesize = np.array(sesize)
    ndim = len(sesize)

    o = np.array(((0,0),(0,0), (0,0)))

    for i in range(0, ndim):   
        if sesize[i] % 2 == 0:
            o[i,0] = sesize[i]/2
            o[i,1] = sesize[i]/2 - 1 + 1
        else:
            o[i,0] = (sesize[i]-1)/2
            o[i,1] = o[i,0] + 1

    return o.astype('int')


def structure_element_2d(setype = 'Disk', sesize = (3,3)):
    """Creates specific 2d structuring elements
    
    Arguments:
        setype (str): structure element type, see :ref:`StructureElementTypes`
        sesize (array or tuple): size of the structure element
    
    Returns:
        array: structure element
    """
    
    setype = setype.lower()

    if len(sesize) != 2:
        raise Exception('structureElement2D: sesize is not 2d')

    o = structure_element_offsets(sesize)
    omax = o.min(axis=1)
    sesize = np.array(sesize)

    if setype == 'sphere':
        g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1]]
        add = ((sesize + 1) % 2) / 2.
        x = g[0,:,:,:] + add[0]
        y = g[1,:,:,:] + add[1]

        se = 1 - (x * x / (omax[0] * omax[0]) + y * y / (omax[1] * omax[1]))
        se[se < 0] = 0
        return se / se.sum()

    elif setype == 'disk':
       
        g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1]]
        add = ((sesize + 1) % 2) / 2.

        x = g[0,:,:] + add[0]
        y = g[1,:,:] + add[1]

        se = 1 - (x * x / (omax[0] * omax[0]) + y * y / (omax[1] * omax[1]))
        se[se >= 0] = 1
        se[se < 0] = 0
        return se.astype('int')

    else:
        return np.ones(sesize)


def structure_element_3d(setype = 'Disk', sesize = (3,3,3)):
    """Creates specific 3d structuring elements
        
    Arguments:
        setype (str): structure element type, see :ref:`StructureElementTypes`
        sesize (array or tuple): size of the structure element
    
    Returns:
        array: structure element
    """
    
    setype = setype.lower()

    if len(sesize) != 3:
        raise Exception('structureElement3D: sesize is not 3d')

    o = structure_element_offsets(sesize)
    omax = o.max(axis=1)
    sesize = np.array(sesize)

    if setype == 'sphere':
        g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1], -o[2,0]:o[2,1]]
        add = ((sesize + 1) % 2) / 2.
        x = g[0,:,:,:] + add[0]
        y = g[1,:,:,:] + add[1]
        z = g[2,:,:,:] + add[2]

        se = 1 - (x * x / (omax[0] * omax[0]) + y * y / (omax[1] * omax[1]) + z * z / (omax[2] * omax[2]))
        se[se < 0] = 0
        return se / se.sum()

    elif setype == 'disk':
        
        g = np.mgrid[-o[0,0]:o[0,1], -o[1,0]:o[1,1], -o[2,0]:o[2,1]]
        add = ((sesize + 1) % 2) / 2.
        x = g[0,:,:,:] + add[0]
        y = g[1,:,:,:] + add[1]
        z = g[2,:,:,:] + add[2]

        se = 1 - (x * x / (omax[0] * omax[0]) + y * y / (omax[1] * omax[1]) + z * z / (omax[2] * omax[2]))
        se[se < 0] = 0
        se[se > 0] = 1
        return se.astype('int')

    else:
        return np.ones(sesize)
        

def structure_element_binary(image_dim, connectivity=1, offset=None):
    """Convert any valid connectivity to a structuring element and offset.

    Parameters
    ----------
    image_dim : int
        The number of dimensions of the input image.
    connectivity : int, array
        The neighborhood connectivity. An integer is interpreted as in
        ``scipy.ndimage.generate_binary_structure``, as the maximum number
        of orthogonal steps to reach a neighbor. An array is directly
        interpreted as a structuring element and its shape is validated against
        the input image shape. 1 is 6-way connectivity, 3 is 18-way
    offset : tuple of int, or None
        The coordinates of the center of the structuring element.

    Returns
    -------
    c_connectivity : array of bool
        The structuring element corresponding to the input `connectivity`.
    offset : array of int
        The offset corresponding to the center of the structuring element.

    Raises
    ------
    ValueError:
        If the image dimension and the connectivity or offset dimensions don't
        match.
    """

    if np.isscalar(connectivity):
        c_connectivity = ndi.generate_binary_structure(image_dim, connectivity)
    else:
        c_connectivity = np.array(connectivity, bool)
        if c_connectivity.ndim != image_dim:
            raise ValueError("Connectivity dimension must be same as image")

    if offset is None:
        if any([x % 2 == 0 for x in c_connectivity.shape]):
            raise ValueError("Connectivity array must have an unambiguous "
                             "center")

        offset = np.array(c_connectivity.shape) // 2

    return c_connectivity, offset


def _offsets_to_raveled_neighbors(image_shape, structure, center, order='C'):
    """Compute offsets to a samples neighbors if the image would be raveled.

    Parameters
    ----------
    image_shape : tuple
        The shape of the image for which the offsets are computed.
    structure : ndarray
        A structuring element determining the neighborhood expressed as an
        n-D array of 1's and 0's.
    center : sequence
        Tuple of indices specifying the center of `selem`.

    Returns
    -------
    offsets : ndarray
        Linear offsets to a samples neighbors in the raveled image, sorted by
        their Euclidean distance from the center.

    Examples
    --------
    >>> _offsets_to_raveled_neighbors((4, 5), np.ones((4, 3)), (1, 1))
    array([-5, -1,  1,  5, -6, -4,  4,  6, 10,  9, 11])
    """
    structure = structure.copy()  # Don't modify original input
    structure[tuple(center)] = 0  # Ignore the center; it's not a neighbor
    connection_indices = np.transpose(np.nonzero(structure))
    offsets = (np.ravel_multi_index(connection_indices.T, image_shape,
                                    order=order) -
               np.ravel_multi_index(center, image_shape, order=order))
    squared_distances = np.sum((connection_indices - center) ** 2, axis=1)
    return offsets[np.argsort(squared_distances)]

    
     
