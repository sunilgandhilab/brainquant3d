# -*- coding: utf-8 -*-
"""
IO interface to read microscope and point data

This is the main module to distribute the reading and writing of individual data formats to the specialized sub-modules.

See :mod:`bq3d.io` for details.

"""

import sys

import os
import re
import numpy
import importlib
import shutil
from pathlib import Path

from bq3d.utils.chunking import range_to_slices

import logging
log = logging.getLogger(__name__)
self = sys.modules[__name__]

pointFileExtensions = ['csv', 'txt', 'npy', 'vtk', 'ims', 'json']
"""list of extensions supported as a point data file"""

pointFileTypes = ['CSV', 'NPY', 'VTK', 'Imaris', 'JSON']
"""list of point data file types"""

pointFileExtensionToType = {'csv' : 'CSV', 'txt' : 'CSV', 'npy' : 'NPY', 'vtk' : 'VTK',
                            'ims' : 'Imaris', 'json': 'json'}
"""map from point file extensions to point file types"""

dataFileExtensions = ['tif', 'tiff', 'mhd', 'raw', 'zraw', 'ims', 'nrrd', 'npy', 'NPY', 'csv',
                      'json']
"""list of extensions supported as a image data file"""

dataFileTypes = ['FileList', 'TIF', 'RAW', 'NRRD', 'Imaris', 'NPY', 'CSV','json']
"""list of image data file types"""

mappableFileTypes = ['TIF', 'NPY']
"""list of file types compatable with bq3dmemory mapping"""

dataFileExtensionToType = { 'tif' : 'TIF', 'tiff' : 'TIF', 'raw' : 'RAW', 'zraw' : 'RAW',
                            'mhd' : 'RAW', 'nrrd': 'NRRD', 'ims' : 'Imaris', 'npy' : 'NPY',
                            'json': 'json', 'csv' : 'CSV'}
"""map from image file extensions to image file types"""


##############################################################################
# Basic file queries
##############################################################################

def fileExtension(filename):
    """Returns file extension if exists
    
    Arguments:
        filename (str): file name
        
    Returns:
        str: file extension or None
    """

    if isinstance(filename, Path):
        filename = filename.as_posix()
    elif not isinstance(filename, str):
        return None

    fext = filename.split('.')
    if len(fext) < 2:
        return None
    else:
        return fext[-1]


def isFile(source):
    """Checks if filename is a real file, returns false if it is directory or regular expression
    
    Arguments:
        source (str): source file name
        
    Returns:
        bool: true if source is a real file   
    """
    
    if not isinstance(source, str):
        return False

    if os.path.exists(source):
        if os.path.isdir(source):
            return False
        else:
            return True
    else:
        return False


def isFileExpression(source):
    """Checks if filename is a regular expression denoting a file list
    
    Arguments:
        source (str): source file name
        
    Returns:
        bool: True if source is regular expression with a digit label in format \d{3,}
    """    
    
    if not isinstance(source, str):
        return False

    if isFile(source):
        return False
    else:
        m = re.search('{[\d,]+}', source)
        if m is None:
            return False
        else:
            return True


def isDataFile(source):
    """Checks if a file is a valid image data file
     
    Arguments:
        source (str): source file name
        
    Returns:
        bool: true if source is an image data file
    """   
    
    if not isinstance(source, str):
        return False

    fext = fileExtension(source)
    if fext in dataFileExtensions:
        return True
    else:
        return False


def isMappable(source):
    """Checks if a file is compatable with bq3dmemmap

    Arguments:
        source (str): source file name

    Returns:
        bool: true if source is memory mappable
    """

    typ = dataFileNameToType(source)
    if typ in mappableFileTypes:
        return True
    else:
        return False

def getDataType(source, **kwargs):
    """gets dtype of data in file

    Arguments:
        filename (str): file name

    Returns:
        dtype: data type
    """

    if isinstance(source, str):
        mod = dataFileNameToModule(source)
        return mod.getDataType(source, **kwargs)
    else:
        raise RuntimeError('Could not determine datatype of input file')

def createDirectory(filename):
    """Creates the directory of the file if it does not exists
     
    Arguments:
        filename (str): file name
        
    Returns:
        str: directory name
    """       
    
    dirname, fname = os.path.split(filename)
    if not os.path.exists(dirname) and dirname != '':
        os.makedirs(dirname)

    return dirname


def pointFileNameToType(filename):
    """Returns type of a point file
    
    Arguments:
        filename (str): file name
        
    Returns:
        str: point data type in :const:`pointFileTypes`
    """       


    fext = fileExtension(filename)
    if fext in pointFileExtensions:
        return pointFileExtensionToType[fext]
    else:
       raise RuntimeError('Cannot determine type of point file %s with extension %s' % (filename, fext))


def dataFileNameToType(filename):
    """Returns type of a image data file
    
    Arguments:
        filename (str): file name
        
    Returns:
        str: image data type in :const:`dataFileTypes`
    """      
    
    if isFileExpression(filename):
        return 'FileList'
    else:
        fext = fileExtension(filename)
        if fext in dataFileExtensions:
            return dataFileExtensionToType[fext]
        else:
           raise RuntimeError('Cannot determine type of data file %s with extension %s' % (filename, fext))


def dataFileNameToModule(filename):
    """Return the module that handles io for a data file
        
    Arguments:
        filename (str): file name
        
    Returns:
        object: sub-module that handles a specific data type
    """          
    
    ft = dataFileNameToType(filename)
    return importlib.import_module('bq3d.io.' + ft)


def pointFileNameToModule(filename):
    """Return the module that handles io for a point file
        
    Arguments:
        filename (str): file name
        
    Returns:
        object: sub-module that handles a specific point file type
    """ 

    ft = pointFileNameToType(filename)
    return importlib.import_module('bq3d.io.' + ft)


##############################################################################
# Data Sizes and Ranges
##############################################################################

    
def dataSize(source, x = None, y = None, z = None, **args):
    """Returns array size of the image data needed when read from file and reduced to specified ranges
       
    Arguments:
        source (array or str): source data
        x,y,z (tuple or None): range specifications, ``None`` is no cropping
        
    Returns:
        tuple: size of the image data after reading and range reduction
    """
    
    if isinstance(source, str):
        mod = dataFileNameToModule(source)
        return mod.dataSize(source, x = x, y = y, z = z, **args)
    elif isinstance(source, numpy.ndarray):
        return dataSizeFromDataRange(source.shape, x = x, y = y, z = z)
    elif isinstance(source, tuple):
        return dataSizeFromDataRange(source, x = x, y = y, z = z)
    else:
        raise RuntimeError('dataSize: argument not a string, tuple or array!')


def dataZSize(source, z = None, **args):
    """Returns size of the array in the third dimension, None if 2D data
           
    Arguments:
        source (array or str): source data
        z (tuple or None): z-range specification, ``None`` is no cropping
        
    Returns:
        int: size of the image data in z after reading and range reduction
    """ 
      
    if isinstance(source, str):
        mod = dataFileNameToModule(source)
        return mod.dataZSize(source, z = z, **args)
    elif isinstance(source, numpy.ndarray):
        if len(source.shape) > 2: 
            return toDataSize(source.shape[2], r = z)
        else:
            return None
    elif isinstance(source, tuple):
        if len(source) > 2: 
            return toDataSize(source[2], r = z)
        else:
            return None
    else:
        raise RuntimeError('dataZSize: argument not a string, tuple or array!')


def toDataRange(size, r = None):
    """Converts range r to numeric range (min,max) given the full array size
       
    Arguments:
        size (tuple): source data size
        r (tuple or None): range specification, ``None`` is no cropping
        
    Returns:
        tuple: absolute range as pair of integers
    
    See Also:
        :func:`toDataSize`, :func:`dataSizeFromDataRange`
    """     

    if r is None:
        return 0, size

    if isinstance(r, int) or isinstance(r, float):
        r = (r, r +1)

    if r[0] is None:
        r = (0, r[1])
    if r[0] < 0:
        if -r[0] > size:
            r = (0, r[1])
        else:
            r = (size + r[0], r[1])
    if r[0] > size:
        r = (size, r[1])

    if r[1] is None:
        r = (r[0], size)
    if r[1] < 0:
        if -r[1] > size:
            r = (r[0], 0)
        else:
            r = (r[0], size + r[1])
    if r[1] > size:
        r = (r[0], size)

    if r[0] > r[1]:
        r = (r[0], r[0])

    return r


def toDataSize(size, r = None):
    """Converts full size to actual size given range r
    
    Arguments:
        size (tuple): data size
        r (tuple or None): range specification, ``None`` is no cropping
        
    Returns:
        int: data size
    
    See Also:
        :func:`toDataRange`, :func:`dataSizeFromDataRange`
    """
    dr = toDataRange(size, r = r)
    return int(dr[1] - dr[0])


def dataSizeFromDataRange(dataSize, x = None, y = None, z = None):
    """Converts full data size to actual size given ranges for x,y,z
    
    Arguments:
        dataSize (tuple): data size
        x,y,z (tuple or None): range specifications, ``None`` is no cropping
        
    Returns:
        tuple: data size as tuple of integers
    
    See Also:
        :func:`toDataRange`, :func:`toDataSize`
    """    
    
    dataSize = list(dataSize)
    n = len(dataSize)
    if n > 0:
        dataSize[0] = toDataSize(dataSize[0], r = z)
    if n > 1:
        dataSize[1] = toDataSize(dataSize[1], r = y)
    if n > 2:
        dataSize[2] = toDataSize(dataSize[2], r = x)

    return tuple(dataSize)


def dataToRange(data, x = None, y = None, z = None):
    """Reduces data to specified ranges
    
    Arguments:
        data (array): full data array
        x,y,z (tuple or None): range specifications, ``None`` is no cropping
        
    Returns:
        array: reduced data
    
    See Also:
        :func:`dataSizeFromDataRange`
    """  

    shape = data.shape
    d = data.ndim
    rr = []

    if d > 3:
        raise RuntimeError('dataToRange: dimension %d to big' % d)
    if d > 2:
        rr.append(toDataRange(shape[-3], r = z))
    if d > 1:
        rr.append(toDataRange(shape[-2], r = y))
    if d > 0:
        rr.append(toDataRange(shape[-1], r = x))

    return numpy.squeeze(data[range_to_slices(rr)])


##############################################################################
# Read / Write Data
##############################################################################

def readData(source, **args):
    """Read data from one of the supported formats
    
    Arguments:
        source (str, array or None): full data array, if numpy array simply reduce its range
        x,y,z (tuple or None): range specifications, ``None`` is full range
        args: further arguments specific to image data format reader
    
    Returns:
        array: data as numpy array
    
    See Also:
        :meth:`writeData`
    """
    if isinstance(source, Path):
        source = source.as_posix()

    if isinstance(source, (str, numpy.memmap)):
        log.debug(f'Reading {source}')

    if source is None:
        return None
    elif isinstance(source, str):
        mod = dataFileNameToModule(source)
        return mod.readData(source, **args)
    elif isinstance(source, numpy.ndarray) or isinstance(source, numpy.memmap):
        return dataToRange(source, **args)
    else:
        log.exception('readData: cannot infer format of the requested data/file.')
        raise RuntimeError


def empty(filename, shape, dtype, **kwargs):
    """ Create an empty image
    """

    mod = dataFileNameToModule(filename)
    os.makedirs(Path(filename).parent, exist_ok=True)
    return mod.empty(filename, shape, dtype, **kwargs)


def writeData(sink, data, **kwargs):
    """Write data to one of the supported formats
    
    Arguments:
        sink (str, array or None): the destination for the data, if None the data is returned directly
        data (array, DataFrame or None): data to be written
        args: further arguments specific to image data format writer
            TIF: returnMemmap: (Bool) returns the written data as a memmapped array
    
    Returns:
        array, str or None: data or file name of the written data
    See Also:
        :func:`readData`
    """
    if isinstance(sink, Path):
        sink = sink.as_posix()

    if sink is None: # dont write but return the data
        log.debug('writeData recieved sink of None')
        return data
    elif isinstance(sink, str):

        mod = dataFileNameToModule(sink)
        createDirectory(sink)
        return mod.writeData(sink, data, **kwargs)
    else:
        return sink

def copyFile(source, sink):
    """Copy a file from source to sink
    
    Arguments:
        source (str): file name of source
        sink (str): file name of sink
    
    Returns:
        str: name of the copied file
    
    See Also:
        :func:`copyData`, :func:`convertData`
    """ 
    
    shutil.copy(source, sink)
    return sink


def copyData(source, sink, **kwargs):
    """Copy a data file from source to sink, which can consist of multiple files

    Will also handle file type conversions and crop for some file types
    
    Arguments:
        source (str): file name of source
        sink (str): file name of sink
    
    Returns:
        str: name of the copied file
    
    See Also:
        :func:`copyFile`, :func:`convertData`
    """     

    if isinstance(source, Path):
        source = source.as_posix()
    if isinstance(sink, Path):
        sink = sink.as_posix()

    mod = dataFileNameToModule(source)
    return mod.copyData(source, sink, **kwargs)


def convertData(source, sink, **kwargs):
    """Transforms data from source format to sink format
    
    Arguments:
        source (str): file name of source
    
    Returns:
        str: name of the copied file
        
    Warning:
        Not optimized for large image data sets
    
    See Also:
        :func:`copyFile`, :func:`copyData`
    """


    if source is None:
        return None

    elif isinstance(source, numpy.ndarray):
        if sink is None:
            return dataToRange(source, **kwargs)
        elif isinstance(sink,  str):
            data = dataToRange(source, **kwargs)
            return writeData(sink, data)
        else:
            raise RuntimeError('convertData: unknown sink!')

    elif isinstance(source, str):
        if sink is None:
            return readData(source, **kwargs)
        if isinstance(sink, str):
            return copyData(source, sink, **kwargs)
        else:
            raise RuntimeError('convrtData: unknown sink!')

def toMultiChannelData(*args):
    """Concatenate single channel arrays to one multi channel array
    
    Arguments:
        args (arrays): arrays to be concatenated
    
    Returns:
        array: concatenated multi-channel array
    """
    
    data = numpy.array(args)
    return data.rollaxis(data, 0, data.ndim)


##############################################################################
# Read / Write Points
##############################################################################


def pointsToCoordinates(points):
    """Converts a (coordiantes, properties) tuple to the coordinates only
    
    Arguments:
        points (array or tuple): point data to be reduced to coordinates
    
    Returns:
        array: coordiante data
        
    Notes:
        Todo: Move this to a class that handles points and their meta data
    """
    
    if isinstance(points, tuple):
        return points[0]
    else:
        return points


def pointsToProperties(points):
    """Converts a (coordiante, properties) tuple to the properties only
    
    Arguments:
        points (array or tuple): point data to be reduced to properties
    
    Returns:
        array: property data
        
    Notes:
        Todo: Move this to a class that handles points and their meta data
    """    
    
    if isinstance(points, tuple) and len(points) > 1:
        return points[1]
    else:
        return None


def pointsToCoordinatesAndProperties(points):
    """Converts points in various formats to a (coordinates, properties) tuple
    
    Arguments:
        points (array or tuple): point data to be converted to (coordinates, properties) tuple
    
    Returns:
        tuple: (coordinates, properties) tuple
        
    Notes:
        Todo: Move this to a class that handles points and their meta data
    """  
    
    if isinstance(points, tuple):
        if len(points) == 0:
            return None, None
        elif len(points) == 1:
            return points[0], None
        elif len(points) == 2:
            return points
        else:
            raise RuntimeError('points not a tuple of 0 to 2 elements!')
    else:
        return points, None


def pointsToCoordinatesAndPropertiesFileNames(filename, propertiesPostfix = '_intensities'):
    """Generates a tuple of filenames to store coordinates and properties data separately
    
    Arguments:
        filename (str): point data file name
        propertiesPostfix (str): postfix on file name to indicate property data
    
    Returns:
        tuple: (file name, file name for properties)
        
    Notes:
        Todo: Move this to a class that handles points and their meta data
    """  
    
    if isinstance(filename, str):
        return filename, filename[:-4] + propertiesPostfix + filename[-4:]
    elif isinstance(filename, tuple):
        if len(filename) == 1:
            if filename[0] is None:
                return None, None
            elif isinstance(filename[0], str):
                return filename[0], filename[0][:-4] + propertiesPostfix + filename[0][-4:]
            else:
                raise RuntimeError('pointsFilenames: invalid filename specification!')
        elif len(filename) == 2:
            return filename
        else:
            raise RuntimeError('pointsFilenames: invalid filename specification!')
    elif filename is None:
        return None, None
    else:
        raise RuntimeError('pointsFilenames: invalid filename specification!')


def pointShiftFromRange(dataSize, x = None, y = None, z = None):
    """Calculate shift of points given a specific range restriction
    
    Arguments:
        dataSize (str): data size of the full image
        x,y,z (tuples or None): range specifications
    
    Returns:
        tuple: shift of points from original origin of data to origin of range reduced data
    """
    
    if isinstance(dataSize, str):
        dataSize = self.dataSize(dataSize)
    dataSize = list(dataSize)

    d = len(dataSize)
    rr = []
    if d > 0:
        rr.append(toDataRange(dataSize[0], r = x))
    if d > 1:
        rr.append(toDataRange(dataSize[1], r = y))
    if d > 2:
        rr.append(toDataRange(dataSize[2], r = z))
    if d > 3 or d < 1:
        raise RuntimeError('shiftFromRange: dimension %d to big' % d)

    return [r[0] for r in rr]


def pointsToRange(points, dataSize = None, x = None, y = None, z = None, shift = False):
    """Restrict points to a specific range
    
    Arguments:
        points (array or str): point source
        dataSize (str): data size of the full image
        x,y,z (tuples or None): range specifications
        shift (bool): shift points to relative coordinates in the reduced image
    
    Returns:
        tuple: points reduced in range and optionally shifted to the range reduced origin
    """
    

    if x is None and y is None and z is None:
        return points

    istuple = isinstance(points, tuple)
    (points, properties) = pointsToCoordinatesAndProperties(points)

    if points is None:
        if istuple:
            return points, properties
        else:
            return points

    if not isinstance(points, numpy.ndarray):
        raise RuntimeError('pointsToRange: points not None or numpy array!')

    d = points.shape[1]

    if dataSize is None:
        dataSize = points.max(axis=0)
    elif isinstance(dataSize, str):
        dataSize = self.dataSize(dataSize)

    rr = []
    if d > 0:
        rr.append(self.toDataRange(dataSize[0], r = x))
    if d > 1:
        rr.append(self.toDataRange(dataSize[1], r = y))
    if d > 2:
        rr.append(self.toDataRange(dataSize[2], r = z))
    if d > 3 or d < 1:
        raise RuntimeError('pointsToRange: dimension %d to big' % d)

    if d > 0:
        ids = numpy.logical_and(points[:,0] >= rr[0][0], points[:,0] < rr[0][1])
    if d > 1:
        ids = numpy.logical_and(numpy.logical_and(ids, points[:,1] >= rr[1][0]), points[:,1] < rr[1][1])
    if d > 2:
        ids = numpy.logical_and(numpy.logical_and(ids, points[:,2] >= rr[2][0]), points[:,2] < rr[2][1])

    points = points[ids, :]

    if shift:
        sh = [r[0] for r in rr]
        points = points - sh

    if not properties is None:
        properties = properties[ids]

    if istuple:
        return points, properties
    else:
        return points


def readPoints(source, **args):
    """Read a list of points from csv or vtk
    
    Arguments:
        source (str, array, tuple or None): the data source file
        args: further arguments specific to point data format reader
    
    Returns:
        array or tuple or None: point data of source
    
    See Also:
        :func:`writePoints`
    """ 
    
    istuple = isinstance(source, tuple)

    if source is None:
        source = (None, None)
    elif isinstance(source, numpy.ndarray):
        source = (source, None)
    elif isinstance(source, str):
        source = (source, None)
    elif isinstance(source, tuple):
        if len(source) == 0:
            source = (None, None)
        elif len(source) == 1: 
            if source[0] is None:
                source = (None, None)
            elif isinstance(source[0], numpy.ndarray):
                source = (source[0], None)
            elif isinstance(source[0], str):
                source = pointsToCoordinatesAndPropertiesFileNames(source, **args)
            else:
                raise RuntimeError('readPoints: cannot infer format of the requested data/file.')
        elif len(source) == 2:
            if not((source[0] is None or isinstance(source[0], str) or isinstance(source[0], numpy.ndarray)) and 
                   (source[1] is None or isinstance(source[1], str) or isinstance(source[0], numpy.ndarray))):
               raise RuntimeError('readPoints: cannot infer format of the requested data/file.')
        else:
            raise RuntimeError('readPoints: cannot infer format of the requested data/file.')
    else:
        raise RuntimeError('readPoints: cannot infer format of the requested data/file.')

    if source[0] is None:
        points = None
    elif isinstance(source[0], numpy.ndarray):
        points = source[0]
    elif isinstance(source[0], str):
        mod = self.pointFileNameToModule(source[0])
        points = mod.readPoints(source[0])

    if source[1] is None:
        properties = None
    elif isinstance(source[1], numpy.ndarray):
        properties = source[1]
    elif isinstance(source[1], str):
        mod = self.pointFileNameToModule(source[1])
        properties = mod.readPoints(source[1])

    if istuple:
        return self.pointsToRange((points, properties), **args)
    else:
        return self.pointsToRange(points, **args)


def writePoints(sink, points, **args):
    """Write a list of points to csv, vtk or ims files
    
    Arguments:
        sink (str or None): the destination for the point data
        points (array or tuple or None): the point data, optionally as (coordinates, properties) tuple
        args: further arguments specific to point data format writer
    
    Returns:
        str: output filename
    
    See Also:
        :func:`readPoints`
    """ 
    
    mod = self.pointFileNameToModule(sink)
    abs_path = Path(sink).absolute()
    if not Path(abs_path.parent).is_dir():
        os.mkdir(abs_path.parent)

    ret = mod.writePoints(abs_path.as_posix(), points)

    return ret


def writeTable(filename, table):
    """Writes a numpy array with column names to a csv file.
    
    Arguments:
        filename (str): filename to save table to
        table (annotated array): table to write to file
        
    Returns:
        str: file name
    """
    with open(filename,'w') as f:
        for sublist in table:
            f.write(', '.join([str(item) for item in sublist]))
            f.write('\n')
        f.close()

    return filename

