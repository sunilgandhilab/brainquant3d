import os
import shutil
import numpy as np
import tifffile as tif
from bq3d import io
from bq3d.utils.chunking import range_to_slices

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

def dataSize(filename, **args):
    """Returns size of data in tif file
    
    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
    
    Returns:
        tuple: data size
    """

    t = tif.tifffile.memmap(filename)
    s = t.shape

    return io.dataSizeFromDataRange(s, **args)


def dataZSize(filename, z = None):
    """Returns z size of data in tif file
    
    Arguments:
        filename (str): file name as regular expression
        z (tuple): z data range specification
    
    Returns:
        int: z data size
    """
    
    t = tif.TiffFile(filename)

    d2 = t.pages[0].shape
    if len(d2) == 3:
      return io.toDataSize(d2[0], r = z)

    d3 = len(t.pages)
    if d3 > 1:
        return io.toDataSize(d3, r = z)
    else:
        return None

def getDataType(filename):
    """gets dtype of data in file

    Arguments:
        filename (str): file name

    Returns:
        dtype: data type
    """
    return io.readData(filename).dtype


def readData(filename, x = None, y = None, z = None, returnMemmap = True, **kwargs):
    """Read data from a single tif image or stack
    
    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
    
    Returns:
        array: image data
    """

    if returnMemmap:
        data = tif.tifffile.memmap(filename, **kwargs)
    else:
        data = tif.imread(filename)

    return io.dataToRange(data, x = x, y = y, z = z)


def empty(filename, shape, dtype, **kwargs):

    return tif.tifffile.memmap(filename, shape=shape, dtype=dtype, **kwargs)


def writeData(filename, data, rgb = False, substack = None, returnMemmap = True):
    """Write image data to tif file
    
    Arguments:
        filename (str): file name 
        data (array): image data in x,y,z
        rgb (bool): if true will save RGB image. channels should be last array axis
        returnMemmap (bool): returns array rather than file name
    Returns:
        str or np.array: output file name or memory mapped array
    """

    fn, fe = os.path.splitext(filename) # initial write to partial filename to prevent creating incomplete files
    d = len(data.shape) # fiji wants 'TZCYXS'


    if data.dtype == '>u2':
        dtype = data.flat[0].dtype
    else:
        dtype = data.dtype

    if substack:
        sub = range_to_slices(substack)
        data_map = io.readData(filename)
        data_map[sub] = data

    else:
        if not rgb:
            if d == 2: # XY
                data_map = tif.tifffile.memmap(fn, dtype=dtype, shape=data.shape)
            elif d == 3: # XYZ
                data_map = tif.tifffile.memmap(fn, dtype=dtype, shape=data.shape, bigtiff = True) #imageJ = true not work for int32
            elif d == 4: #XYZC
                data_map = tif.tifffile.memmap(fn, dtype=dtype, shape=data.shape, imagej = True)
            else:
                raise RuntimeError('writing {} dimensional data to tif not supported!'.format(len(data.shape)))
        else:
            if d == 3: # XYC
                data_map = tif.tifffile.memmap(fn, dtype=dtype, shape=data.shape, imagej = True) #imageJ = true not work for int32
            elif d == 4: # XYZS
                data_map = tif.tifffile.memmap(fn, dtype=dtype, shape=data.shape, imagej = True)
            else:
                raise RuntimeError('writing {} dimensional data to tif not supported!'.format(len(data.shape)))

        data_map[:] = data
        shutil.move(fn, filename)

    if returnMemmap:
        return readData(filename)
    else:
        return filename


def copyData(source, sink, x=None, y=None, z=None, returnMemmap = True):
    """Copy a data file from source to sink
    
    Arguments:
        source (str): file name pattern of source
        sink (str): file name pattern of sink
        returnMemmap (bool): returns the result as an array
    Returns:
        str: file name of the copy
    """
    out_type = io.dataFileNameToType(sink)
    if out_type == 'TIF':
        if isinstance(source, np.memmap) and x==y==y==z==None:
            shutil.copyfile(source.filename, sink)
        else:
            Xsize, Ysize, Zsize = io.dataSize(source)
            # cropped size
            Xsize = io.toDataSize(Xsize, r=x)
            Ysize = io.toDataSize(Ysize, r=y)
            Zsize = io.toDataSize(Zsize, r=z)
            im = io.readData(source, x = x, y= y, z= z)
            out = io.writeData(sink, im, returnMemmap = returnMemmap)

        if returnMemmap:
            return io.readData(sink)
        else:
            return sink
    else:
        raise RuntimeError('copying from TIF to {} not yet supported.'.format(out_type))
