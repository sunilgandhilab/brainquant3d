"""
Plotting routines for overlaying labels, tilings, and sectioning of 3d data sets

Supported functionality:

    * plot volumetric data as a sequence of tiles via :func:`plotTiling`
    * overlay points on images via :func:`overlayPoints` and 
      :func:`plotOverlayPoints`
    * overlay labeled images on gray scale images via :func:`overlayLabel` and 
      :func:`plotOverlayLabel`

"""


import numpy
import tifffile as tif
import matplotlib as mpl

import clearmap3.IO as io
import clearmap3.analysis.voxelization as vox


import logging
log = logging.getLogger(__name__)


def overlay_label(dataSource, labelSource, output = None,  alpha = False, labelColorMap = 'jet', x = all, y = all, z = all):
    """Overlay a gray scale image with colored labeled image
    
    Arguments:
        dataSouce (str or array): volumetric image data
        labelSource (str or array): labeled image to be overlayed on the image data
        output (str or None): destination for the overlayed image
        alpha (float or False): transparency
        labelColorMap (str or object): color map for the labels
        x, y, z (all or tuple): sub-range specification
    
    Returns:
        (array or str): figure handle
        
    See Also:
        :func:`overlayPoints`
    """ 
    
    label = io.readData(labelSource, x= x, y = y, z = z)
    image = io.readData(dataSource, x= x, y = y, z = z)

    lmax = label.max()
    if lmax <= 1:
        carray = numpy.array([[1,0,0,1]])
    else:
        cm = mpl.cm.get_cmap(labelColorMap)
        cNorm  = mpl.colors.Normalize(vmin=1, vmax = int(lmax))
        carray = mpl.cm.ScalarMappable(norm=cNorm, cmap=cm)
        carray = carray.to_rgba(numpy.arange(1, int(lmax + 1)))

    if not alpha:
        carray = numpy.concatenate(([[0,0,0,1]], carray), axis = 0)
    else:
        carray = numpy.concatenate(([[1,1,1,1]], carray), axis = 0)

    cm = mpl.colors.ListedColormap(carray)
    carray = cm(label)
    carray = carray.take([0,1,2], axis = -1)

    if not alpha:
        cimage = (label == 0) * image
        cimage = numpy.repeat(cimage, 3)
        cimage = cimage.reshape(image.shape + (3,))
        cimage = cimage.astype(carray.dtype)
        cimage += carray
    else:
        cimage = numpy.repeat(image, 3)
        cimage = cimage.reshape(image.shape + (3,))
        cimage = cimage.astype(carray.dtype)
        cimage *= carray

    return io.writeData(output, cimage)


def overlay_points(pointSource, dataSource, output = None, overlay = True, x = all, y = all, z = all):
    """Overlay points on 3D data and return as color image
    
    Arguments:
        pointSource (str or array): point data to be overlayed on the image data
        dataSource (str or array): volumetric image data. if None just output points as an image
        overlay (bool): if False will not overlay and just output points as an image.
        x, y, z (all or tuple): sub-range specification
    
    Returns:
        (str or array): image overlayed with points
        
    See Also:
        :func:`overlayLabel`
    """

    points = (io.readPoints(pointSource, x = x, y = y, z = z, shift = True)).astype(int)

    if overlay:
        X, Y, Z = io.dataSize(dataSource)
        datatype = io.readData(dataSource, x = x, y = y, z = 0).dtype

        if io.isMappable(output):
            output = tif.tifffile.memmap(output, dtype=datatype, shape=(Z, 2, Y, X), imagej=True) # TZCYXS FIJI
        elif io.isFileExpression(output):
            output = [tif.tifffile.memmap(output, dtype=datatype, shape=(2, Y, X), imagej=True) for z in range(Z)] # sequence of memmap files
        elif output is None:
            output = numpy.zeros((Z, 2, Y, X), dtype=datatype)
        else:
            RuntimeError ('output format not compatable with overlayPoints: ' + output)

        for z in range(Z):
            print(('Overlaying {}...'.format(z)))
            output[z][0][:] = io.readData(dataSource, x=x, y=y, z=z).squeeze().T
            z_points = points[points[:,-1] == z][:, :2]
            output[z][1][[*z_points[:, ::-1].T]] = 1
        return output

    else:
        shape = io.dataSize(dataSource)
        cimage = vox.voxelize(points, shape, output = output, method='Pixel', weights=65535) # TODO: weight should depend on bit depth of dataSource
        return cimage
