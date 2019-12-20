# -*- coding: utf-8 -*-
"""
Simple Interface to read RAW/MHD files e.g. created by elastix

Example:
    >>> import os, numpy
    >>> import bq3d
    >>> import bq3d.io.RAW as raw
    >>> filename = os.path.join(brainquant3dPath, 'Test/Data/Raw/test.mhd') 
    >>> raw.dataSize(filename)
    (20, 50, 10)
"""
#:copyright: Copyright 2015 by Christoph Kirst, The Rockefeller University, New York City
#:license: GNU, see LICENSE.txt for details.

import os
import numpy

import vtk
from vtk.util.numpy_support import vtk_to_numpy

from bq3d import io
import imp

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
    """Read data size from raw/mhd image
    
    Arguments:
        filename (str):  imaris file name
        x,y,z (tuple or all): range specifications
    
    Returns:
        int: raw image data size
    """  
    
    imr = vtk.vtkMetaImageReader()
    imr.SetFileName(filename)
    imr.Update()

    im = imr.GetOutput()
    dims = im.GetDimensions()
    #dims = list(dims)
    #dims[0:2] = [dims[1], dims[0]]
    #dims = tuple(dims)
    return io.dataSizeFromDataRange(dims, **args)


def dataZSize(filename, z = None, **args):
    """Read z data size from raw/mhd image
        
    Arguments:
        filename (str):  imaris file name
        z (tuple or None): range specification
    
    Returns:
        int: raw image z data size
    """  
    
    imr = vtk.vtkMetaImageReader()
    imr.SetFileName(filename)
    imr.Update()

    im = imr.GetOutput()
    dims = im.GetDimensions()
    
    if len(dims) > 2:
        return io.toDataSize(dims[2], r = z)
    else:
        return None


def readData(filename, x = None, y = None, z = None):
    """Read data from raw/mhd image
    
    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
    
    Returns:
        array: image data
    """
    
    imr = vtk.vtkMetaImageReader()
    imr.SetFileName(filename)
    imr.Update()
    
    im = imr.GetOutput()
    dims = im.GetDimensions()

    sc = im.GetPointData().GetScalars()
    img = vtk_to_numpy(sc)
    
    dims = list(dims)
    dims[0:3] = [dims[2], dims[1], dims[0]]

    imgs = list(img.shape)
    if len(imgs) > 1:
        imgs.pop(0)
        dims = dims + imgs

    img = img.reshape(dims)
    #img = img.transpose([1,2,0])
    tp = [2,1,0]
    tp = tp + [i for i in range(3, len(dims))]
    img = img.transpose(tp)

    return io.dataToRange(img, x = x, y = y, z = z)


def writeHeader(filename, meta_dict):
    """Write raw header mhd file
    
    Arguments:
        filename (str): file name of header
        meta_dict (dict): dictionary of meta data
    
    Returns:
        str: header file name
    """       
    
    
    header = ''
    # do not use tags = meta_dict.keys() because the order of tags matters
    tags = ['ObjectType','NDims','BinaryData',
       'BinaryDataByteOrderMSB','CompressedData','CompressedDataSize',
       'TransformMatrix','Offset','CenterOfRotation',
       'AnatomicalOrientation',
       'ElementSpacing',
       'DimSize',
       'ElementType',
       'ElementDataFile',
       'Comment','SeriesDescription','AcquisitionDate','AcquisitionTime','StudyDate','StudyTime']
    for tag in tags:
        if tag in list(meta_dict.keys()):
            header += '%s = %s\n'%(tag,meta_dict[tag])
    f = open(filename,'w')
    f.write(header)
    f.close()
    
    return filename


def writeRawData(filename, data):
    """Write the data into a raw format file.

    Arguments:
        filename (str): file name as regular expression
        data (array): data to write to raw file
    
    Returns:
        str: file name of raw file
    """   
    
    rawfile = open(filename,'wb')
    d = len(data.shape)
    if d <= 2:
        #data.tofile(rawfile)
        data.transpose([1,0]).tofile(rawfile)
    elif d == 3:
        #data.transpose([2,0,1]).tofile(rawfile)
        data.transpose([2,1,0]).tofile(rawfile)
    elif d== 4:
        #data.transpose([3,2,0,1]).tofile(rawfile)
        data.transpose([3,2,1,0]).tofile(rawfile)
    else:
        raise RuntimeError('writeRawData: image dimension %d not supported!' % d)

    rawfile.close()

    return filename


def writeData(filename, data, **args):
    """ Write  data into to raw/mhd file pair

    Arguments:
        filename (str): file name as regular expression
        data (array): data to write to raw file
    
    Returns:
        str: file name of mhd file
    """ 
    
    fext = io.fileExtension(filename)
    if fext == "raw":
        fname = filename[:-3] + 'mhd'
    else:
        fname = filename

    assert(fname[-4:]=='.mhd')
    
    meta_dict = {'ObjectType': 'Image',
                 'BinaryData': 'True',
                 'BinaryDataByteOrderMSB': 'False'}

    numpy_to_datatype   = {numpy.dtype('int8')    : "MET_CHAR",
                           numpy.dtype('uint8')   : "MET_UCHAR",
                           numpy.dtype('int16')   : "MET_SHORT", 
                           numpy.dtype('uint16')  : "MET_USHORT",
                           numpy.dtype('int32')   : "MET_INT",
                           numpy.dtype('uint32')  : "MET_UINT",
                           numpy.dtype('int64')   : "MET_LONG",
                           numpy.dtype('uint64')  : "MET_ULONG",
                           numpy.dtype('float32') : "MET_FLOAT", 
                           numpy.dtype('float64') : "MET_DOUBLE",
                           }    
                           
    dtype = data.dtype
    meta_dict['ElementType'] = numpy_to_datatype[dtype]

    dsize = list(data.shape)
    #dsize[0:2] = [dsize[1],dsize[0]]  #fix arrays represented as (y,x,z)
    
    meta_dict['NDims'] = str(len(dsize))
    meta_dict['DimSize'] = ' '.join([str(i) for i in dsize])
    meta_dict['ElementDataFile'] = os.path.split(fname)[1].replace('.mhd','.raw')
    writeHeader(fname, meta_dict)
    
    pwd = os.path.split(fname)[0]
    if pwd:
        data_file = pwd +'/' + meta_dict['ElementDataFile']
    else:
        data_file = meta_dict['ElementDataFile']

    writeRawData(data_file, data)
    
    return fname


def copyData(source, sink, **kwargs):
    """Copy a raw/mhd file pair from source to sink
    
    Arguments:
        source (str): file name of source
        sink (str): file name of sink
    
    Returns:
        str: file name of the copy
    """     
    
    sourceExt = io.fileExtension(source)
    sinkExt   = io.fileExtension(sink)

    if sinkExt == 'raw' or sinkExt == 'zraw':
        sources = [source]
        sinks = [sink, sink[:-3] + 'mhd']

        if sourceExt == 'raw' or sourceExt == 'zraw':
            sources.append(source[:-3] + 'mhd')
        elif sourceExt == 'mhd':
            if os.path.exists(source[:-3] + 'raw'):
                sources.append(source[:-3] + 'raw')
            elif os.path.exists(source[:-3] + 'zraw'):
                sources.append(source[:-3] + 'zraw')
            else:
                raise RuntimeError('copyData: raw or zraw matching mhd not found')
        else:
            raise RuntimeError('copyData: {} to {} not supported'.format(sourceExt, sinkExt))

        for i in range(2):
            io.copyFile(sources[i], sinks[i])

        return sink

    elif sinkExt == 'mhd':
        sources = [source]
        sinks = []

        if sourceExt == 'raw':
            sources.append(source[:-3] + 'mhd')
            sinks.append(sink[:-3] + 'raw')
            sinks.append(sink)
        elif sourceExt == 'zraw':
            sources.append(source[:-3] + 'mhd')
            sinks.append(sink[:-3] + 'zraw')
            sinks.append(sink)
        elif sourceExt == 'mhd':
            if os.path.exists(source[:-3] + 'raw'):
                sources.append(source[:-3] + 'raw')
                sinks.append(sink)
                sinks.append(sink[:-3] + 'raw')
            elif os.path.exists(source[:-3] + 'zraw'):
                sources.append(source[:-3] + 'zraw')
                sinks.append(sink)
                sinks.append(sink[:-3] + 'zraw')
            else:
                raise RuntimeError('copyData: raw or zraw matching mhd not found')
        else:
            raise RuntimeError('copyData: {} to {} not supported'.format(sourceExt, sinkExt))

        for i in range(2):
            io.copyFile(sources[i], sinks[i])

        return sink

    elif sinkExt == 'tif':
        data = readData(source, **kwargs)
        return io.writeData(sink, data)

    else:
        raise RuntimeError('copyData: {} to {} not supported'.format(sourceExt, sinkExt))
