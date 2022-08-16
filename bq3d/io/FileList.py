"""
Interface to read/write image stacks saved as a list of files

The filename is given as regular expression as described 
`here <https://docs.python.org/2/library/re.html>`_.
"""
import numpy
import os
import re
from pathlib import Path
import tifffile as tif
import multiprocessing


import bq3d
from bq3d import io
from bq3d.utils.files import sort

import logging

from bq3d._version import __version__
__author__     = 'Jack Zeitoun, Ricardo Azevedo'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"

log = logging.getLogger(__name__)

def readFileList(filename, z = None):
    """Returns list of files that match the regular expression
    
    Arguments:
        filename (str): file name as regular expression
        z (tuple)     : range of files to read in
    
    Returns:
        str, list: path of files, file names that match the regular expression
    """
    
    #get path        
    (fpath, fname) = os.path.split(filename)
    fnames = os.listdir(fpath)
    # generate list
    searchRegex = re.compile(fname).search    
    fl = sort([l for l in fnames for m in (searchRegex(l),) if m])

    if not fl:
        raise Exception('no files found in ' + fpath + ' match ' + fname + ' !')
    # crop along z
    if z != None:
        fl = fl[z[0]:z[1]] #TODO: check if correct indexing

    return fpath, fl


def splitFileExpression(filename):
    """Split the regular expression at the digit place holder

    Arguments:
        filename (str): file name as regular expression. regex should be in format '\d{3,4}'
        fileext (str or None): file extension to use if filename is a fileheader only

    Returns:
        tuple: file header (text prior to regex), file extension(text following regex), digit format
    """

    m = re.match('(.+)(\\\\d{[\d,]+})(.+)', filename)

    if m:
        digits = max(list(map(int, re.findall('\d', m.group(2)))))

        fileheader = m.group(1)
        digitfrmt = "%." + str(digits) + "d"
        fileext = m.group(3)
    else:
        raise ValueError(f'{filename} does not contain a valid RegEx.')

    return fileheader, fileext, digitfrmt


def fileExpressionToFileName(filename, z):
    """Insert a number into the regular expression

    Arguments:
        filename (str): file name as regular expression
        z (int or str): z slice index or string to insert

    Returns:
        str: file name
    """

    (fileheader, fileext, digitfrmt) = splitFileExpression(filename)
    if isinstance(z, str):
       return fileheader + z + fileext
    else:
      return fileheader + (digitfrmt % z) + fileext


def dataSize(filename, **args):
    """Returns size of data stored as a file list
    
    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
    
    Returns:
        tuple: data size
    """
    
    fp, fl = readFileList(filename)
    nz = len(fl)

    d2 = io.dataSize(os.path.join(fp, fl[0]))
    if not len(d2) == 2:
        raise RuntimeError("FileList: importing multiple files of dim %d not supported!" % len(d2))

    dims = (nz,) + d2
    return io.dataSizeFromDataRange(dims, **args)


def dataZSize(filename, z = None, **args):
    """Returns size of data stored as a file list
    
    Arguments:
        filename (str): file name as regular expression
        z (tuple): z data range specification
    
    Returns:
        int: z data size
    """
    
    fp, fl = readFileList(filename)
    nz = len(fl)
    return io.toDataSize(nz, r = z)

def getDataType(filename):
    """gets dtype of data in file

    Arguments:
        filename (str): file name

    Returns:
        dtype: data type
    """
    return io.readData(filename, z=0, returnMemmap=False).dtype  #Tif.readData(returnMemmap)


def readDataFiles(filename, x = None, y = None, z = None, **args):
    """Read data from individual images assuming they are the z slices

    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
    
    Returns:
        array: image data
    """
    

    fpath, fl = readFileList(filename)
    nz = len(fl)

    #read first image to get data size and type
    rz = io.toDataRange(nz, r = z)
    sz = io.toDataSize(nz, r = z)
    fn = os.path.join(fpath, fl[rz[0]])
    img = io.readData(fn, x = x, y = y, returnMemmap = False)
    nxy = img.shape
    data = numpy.zeros((sz,) + nxy, dtype = img.dtype)
    data[0,:,:] = img

    for i in range(rz[0]+1, rz[1]):
        log.info("CAUTION: UNTESTED CODE. Double check the validity of your result if you run into this part of the code.")
        fn = os.path.join(fpath, fl[i])
        data[i-rz[0],:,:] = io.readData(fn, x = x, y = y, returnMemmap = False)

    return data


def readData(filename, **args):
    """Read image stack from single or multiple images
    
    Arguments:
        filename (str): file name as regular expression
        x,y,z (tuple): data range specifications
    
    Returns:
        array: image data
    """
    
    if os.path.exists(filename):
         return io.readData(filename, **args)
    else:
         return readDataFiles(filename, **args)


def empty(filename, shape, dtype, **kwargs):
    fileheader, fileext, digitfrmt = splitFileExpression(filename)
    nz = shape[0]

    for i in range(nz):
        fname = fileheader + (digitfrmt % i) + fileext
        io.empty(fname, shape[:-1], dtype)


def writeData(filename, data, startIndex = 0, rgb = False, substack = None, **kwargs):
    """Write image stack to single or multiple image files
    
    Arguments:
        filename (str): file name as regular expression
        data (array): image data
        startIndex (int): index of first z-slice
    
    Returns:
        str: file name as regular expression
    """
    
    # create directory if not exists
    io.createDirectory(filename)

    # check for the \d{xx} part of the regular expression -> if not assume file header
    fileheader, fileext, digitfrmt = splitFileExpression(filename)
    d = data.ndim

    if d == 2:
        fname = fileheader + (digitfrmt % startIndex) + fileext
        io.writeData(fname, data, substack=substack)
        return fname
    else:

        if substack:
            startIndex = substack[0][0]
            nz = substack[0][1] - startIndex
            substack = substack[1:]
        else:
            nz = data.shape[0]

        if rgb:
            if nz == 3:
                fname = fileheader + (digitfrmt % startIndex) + fileext
                io.writeData(fname, data, rgb=True, substack=substack)
                return fname
            else:
                raise RuntimeError('Image does not have correct dimensionality for RGB. format should be XYS')
        else:
            for i in range(nz):
                fname = fileheader + (digitfrmt % (i + startIndex)) + fileext
                log.verbose(f'writing to {fname} at {substack}')
                io.writeData(fname, data[i], substack=substack)
                log.verbose(f'done writing to {fname} at {substack}')
            return filename


def copyData(source, sink, processes = 1, x = None, y = None, z = None, **kwargs):
    """Copy a data file from source to sink when for entire list of files
    
    Arguments:
        source (str): file name pattern of source
        sink (str): file name pattern of sink
        processes (int): number of processes to be used when writing files in parallel

    Returns:
        str: file name pattern of the copy
    Notes:
        TODO: replace cropData with this. currently still using because cropData is more flexible
        TODO: could simplify by not splitting up the regex files
    """

    if isinstance(source, Path):
        source = source.as_posix()
    if isinstance(sink, Path):
        sink = sink.as_posix()

    fp, fl = readFileList(source, z = z) # crops is z by only reading files in range
    out_type = io.dataFileNameToType(sink)

    if out_type == 'FileList':
        # setup inputs for pool
        files = []
        z_idx = []
        for i,fn in enumerate(fl):
            files.append(os.path.join(fp, fn))
            z_idx.append(i)

        f_chunks  = [files[i::processes] for i in range(processes)]
        z_chunks  = [z_idx[i::processes] for i in range(processes)]


        # setup pool
        args = [(sources, z_chunks[i], sink, x, y) for i,sources in enumerate(f_chunks)]

        if processes == 1:
            _parallelCopyToFileList(*args)
        else:
            pool = multiprocessing.Pool(processes)
            pool.map(_parallelCopyToFileList, args)
            pool.close()

    elif out_type == 'TIF':
        # get datasize
        Zsize, Ysize, Xsize = io.dataSize(source)
        # cropped size
        Xsize = io.toDataSize(Xsize, r=x)
        Ysize = io.toDataSize(Ysize, r=y)
        Zsize = io.toDataSize(Zsize, r=z)
        # setup inputs for pool
        data_type   = getDataType(os.path.join(fp, fl[0]))
        files       = [os.path.join(fp, i) for i in fl]
        idxs        = list(range(len(files)))
        z_f_chunks  = [files[i::processes] for i in range(processes)]
        z_i_chunks  = [idxs[i::processes] for i in range(processes)]
        im = io.empty(sink, dtype=data_type, shape=(Zsize, Ysize, Xsize))
        args = [(z_f_chunks[i], idxs, sink, x, y) for i, idxs in enumerate(z_i_chunks)]

        # setup pool
        if processes == 1:
            _parallelCopyToTif(*args)
        else:
            pool = multiprocessing.Pool(processes)
            pool.map(_parallelCopyToTif, args)
            pool.close()

    return sink

def _parallelCopyToFileList(args):
    """copies FileList to FileList in parallel"""
    sources, source_idxs, sink, Xrng, Yrng = args

    for i,file in enumerate(sources):
        log.debug(f'copyData: copying {file} to {sink}')
        im = tif.imread(file)
        if not Xrng == Yrng == None:
            im = io.dataToRange(im, x=Xrng, y=Yrng)

        io.writeData(sink, im, startIndex = source_idxs[i])

def _parallelCopyToTif(args):
    """copies FileList to Tif in parallel"""
    sources, idxs, sink, Xrng, Yrng = args
    output = io.readData(sink)

    for i, idx in enumerate(idxs):
        file = sources[i]
        log.debug(f'copyData: copying {file} to {sink}')
        im = tif.imread(file)
        if not Xrng == Yrng == None:
            output[idx] = io.dataToRange(im, x=Xrng, y=Yrng)
        else:
            output[idx] = im
