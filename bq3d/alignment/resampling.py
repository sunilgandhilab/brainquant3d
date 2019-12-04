# -*- coding: utf-8 -*-
"""
This module provides methods to resample and reorient image data.

Resampling the data is usually necessary as the first step in image registration to match the
resolution and orientation of the reference object. Subsequently, when registering cell
coordinates extracted from image data. resampling may be nescessary to bring points into the
reference frame.

Orientation:
The *orientation* parameter is a tuple of intergers from 0 to n, where n is the number of axes in
the image, that specifies the permutation of the axes similar to *numpy.transpose*. Additionally,
a minus sign in front of a value indicates that the axis is to be mirrored.

    >>> orientation = (2,-1) # swap x and y axes and invert the new y axis.

Note:
The module assumes that images in arrays are arranged as
    * [x,y] or
    * [x,y,z]

"""

import os
import math
import numpy

import multiprocessing
import tempfile
import shutil
import cv2

import bq3d
from bq3d import io
import bq3d.io.FileList as fl

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


def invert_orientation(orientation: tuple):
    """Returns the mirrored permuation.

    Arguments:
        orientation (tuple): orientation specification
    Returns:
        tuple: mirriredorientation sequence
    """

    if orientation:
        return tuple(ax * -1 for ax in orientation)

    return None


def orientationToPermuation(orientation: tuple):
    """return the permuation from an orientation excluding mirroring.

    Arguments:
        orientation (tuple): orientation specification
    Returns:
        tuple: premutation sequence
    See Also:
        `Orientation`_
    """

    if orientation:
        return tuple(int(abs(i)) - 1 for i in orientation)
    else:
        return tuple((1,2,3))

def orientResolution(resolution, orientation):
    """Permutes a resolution tuple according to the given orientation.

    Arguments:
        resolution (tuple): resolution specification
        orientation (tuple or str): orientation specification

    Returns:
        tuple: oriented resolution sequence

    See Also:
        `Orientation`_
    """
    if resolution is None:
        return None

    per = orientationToPermuation(orientation)
    # print orientation, per, resolution
    return tuple(resolution[i] for i in per)


def orientResolutionInverse(resolution, orientation):
    """Permutes a resolution tuple according to the inverse of a given orientation.

    Arguments:
        resolution (tuple): resolution specification
        orientation (tuple or str): orientation specification

    Returns:
        tuple: oriented resolution sequence

    See Also:
        `Orientation`_
    """

    if resolution is None:
        return None

    per = orientationToPermuation(invert_orientation(orientation))
    return tuple(resolution[i] for i in per)


def orientDataSize(dataSize, orientation):
    """Permutes a data size tuple according to the given orientation.

    Arguments:
        dataSize (tuple): resolution specification
        orientation (tuple or str): orientation specification

    Returns:
        tuple: oriented dataSize sequence

    See Also:
        `Orientation`_
    """

    return orientResolution(dataSize, orientation)


def orientDataSizeInverse(dataSize, orientation):
    """Permutes a dataSize tuple according to the inverse of a given orientation.

    Arguments:
        dataSize (tuple): dataSize specification
        orientation (tuple or str): orientation specification

    Returns:
        tuple: oriented dataSize sequence

    See Also:
        `Orientation`_
    """

    return orientResolutionInverse(dataSize, orientation)


def resampleDataSize(dataSizeSource, dataSizeSink=None, resolutionSource=None, resolutionSink=None, orientation=None):
    """Calculate scaling factors and data sizes for resampling.

    Arguments:
        dataSizeSource (tuple): data size of the original image
        dataSizeSink (tuple or None): data size of the resmapled image
        resolutionSource (tuple or None): resolution of the source image
        resolutionSink (tuple or None): resolution of the sink image
        orientation (tuple or str): re-orientation specification

    Returns:
        tuple: data size of the source
        tuple: data size of the sink
        tuple: resolution of source
        tuple: resolution of sink

    See Also:
        `Orientation`_
    """

    # determine data sizes if not specified
    if dataSizeSink is None:
        if resolutionSource is None or resolutionSink is None:
            raise RuntimeError('resampleDataSize: data size and resolutions not defined!')

        # orient resolution of source to resolution of sink to get sink data size
        resolutionSourceO = orientResolution(resolutionSource, orientation)
        dataSizeSourceO = orientDataSize(dataSizeSource, orientation)

        # calculate scaling factor
        dataSizeSink = tuple([int(math.ceil(dataSizeSourceO[i] * resolutionSourceO[i] / resolutionSink[i])) for i in
                              range(len(dataSizeSource))])

    if dataSizeSource is None:
        if resolutionSource is None or resolutionSink is None:
            raise RuntimeError('resampleDataSize: data size and resolutions not defined!')

        # orient resolution of source to resolution of sink to get sink data size
        resolutionSourceO = orientResolution(resolutionSource, orientation)

        # calculate source data size
        dataSizeSource = tuple([int(math.ceil(dataSizeSink[i] * resolutionSink[i] / resolutionSourceO[i])) for i in
                                range(len(dataSizeSink))])
        dataSizeSource = orientDataSizeInverse(dataSizeSource, orientation)

    # calculate effecive resolutions
    if resolutionSource is None:
        if resolutionSink is None:
            resolutionSource = (1, 1, 1)
        else:
            dataSizeSourceO = orientDataSize(dataSizeSource, orientation)
            resolutionSource = tuple(
                float(dataSizeSink[i]) / dataSizeSourceO[i] * resolutionSink[i] for i in range(len(dataSizeSource)))
            resolutionSource = orientResolutionInverse(resolutionSource, orientation)

    dataSizeSourceO = orientDataSize(dataSizeSource, orientation)

    resolutionSourceO = orientResolution(resolutionSource, orientation)
    resolutionSink = tuple(
        float(dataSizeSourceO[i]) / float(dataSizeSink[i]) * resolutionSourceO[i] for i in range(len(dataSizeSource)))

    return dataSizeSource, dataSizeSink, resolutionSource, resolutionSink


def parse_interpolation(interpolation: str):
    """Converts interpolation given as string to cv2 interpolation equivalent

    Arguments:
        interpolation (str or object): interpolation string or cv2 object

    Returns:
        object: cv2 interpolation type
    """
    if interpolation is None:
        interpolation = cv2.INTER_NEAREST

    elif isinstance(interpolation, str):
        if interpolation == 'nn':
            interpolation = cv2.INTER_NEAREST
        elif interpolation == 'linear':
            interpolation = cv2.INTER_LINEAR
        elif interpolation == 'area':
            interpolation = cv2.INTER_AREA

    return interpolation


def resampleXY(source, dataSizeSink, zList=[0], sink=None, interpolation='linear'):
    """Resample an image stack along the a 2d slice
    This routine is used for resampling a large stack in parallel in xy or xz direction.

    Arguments:
        source (str or array): 2d image source
        dataSizeSink (tuple): size of the resmapled image
        zList (int): z planes to crop
        sink (str or None): location for the resmapled image
        interpolation (int): CV2 interpolation method to use

    Returns:
        array or str: resampled data or file name

    """

    for i in zList:
        data = io.readData(source, z=i)
        log.verbose(f'resample XY: Resampling plane {i} to size: ({dataSizeSink[0]}, '
                    f'{dataSizeSink[1]})')
        # note: cv2.resize reverses x-Y axes
        sink[i] = cv2.resize(data, (dataSizeSink[2], dataSizeSink[1]),
                             interpolation=interpolation)

    return sink


def _resampleXYParallel(arg):
    """Resampling helper function to use for parallel resampling of image slices"""

    fileSource = arg[0]
    fileSink = arg[1]
    dataSizeSink = arg[2]
    interpolation = arg[3]
    zChunks = arg[4]

    sink = io.readData(fileSink)
    resampleXY(fileSource, zList=zChunks, sink=sink, dataSizeSink=dataSizeSink, interpolation=interpolation)


def resampleData(source, sink=None, orientation=None, dataSizeSink=None, resolutionSource=(.91, .91, 8.3),
                 resolutionSink=(25, 25, 25), processingDirectory=bq3d.config.temp_dir,
                 processes=bq3d.config.processes,
                 cleanup=True, interpolation='linear', **kwargs):
    """Resample data of source in resolution and orientation

    Arguments:
        source (str or array): image to be resampled
        sink (str or None): destination of resampled image
        orientation (tuple): orientation specified by permuation and change in sign of (1,2,3)
        dataSizeSink (tuple or None): target size of the resampled image
        resolutionSource (tuple): resolution of the source image (in length per pixel)
        resolutionSink (tuple): resolution of the resampled image (in length per pixel)
        processingDirectory (str or None): directory in which to perform resmapling in parallel, None a temporary directry will be created
        processes (int): number of processes to use for parallel resampling
        cleanup (bool): remove temporary files
        interpolation (str): method to use for interpolating to the resmapled image

    Returns:
        (array or str): data or file name of resampled image
    Notes:
        * resolutions are assumed to be given for the axes of the intrinsic
          orientation of the data and reference as when viewed by matplotlib or ImageJ
        * orientation: permuation of 1,2,3 with potential sign, indicating which
          axes map onto the reference axes, a negative sign indicates reversal
          of that particular axes
        * only a minimal set of information to detremine the resampling parameter
          has to be given, e.g. dataSizeSource and dataSizeSink
    """

    log.info(f'interpolation method: {interpolation}')
    log.info(f'Number of processes: {processes}')

    interpolation = parse_interpolation(interpolation)

    dataSizeSource = io.dataSize(source)
    if isinstance(dataSizeSink, str):
        dataSizeSink = io.dataSize(dataSizeSink)

    # orient actual resolutions onto reference resolution
    dataSizeSource, dataSizeSink, resolutionSource, resolutionSink = resampleDataSize(dataSizeSource=dataSizeSource,
                                                                                      dataSizeSink=dataSizeSink,
                                                                                      resolutionSource=resolutionSource,
                                                                                      resolutionSink=resolutionSink,
                                                                                      orientation=orientation)
    dataSizeSinkI = orientDataSizeInverse(dataSizeSink, orientation)

    # setup intermediate output
    if processingDirectory is None:
        processingDirectory = tempfile.mkdtemp()
    else:
        io.createDirectory(processingDirectory)

    resampledXYFile = os.path.join(processingDirectory, 'resampleXY.tif')
    data_type = io.getDataType(source)
    resampledXY = io.empty(resampledXYFile, dtype=data_type,
                                      shape=(dataSizeSource[0], dataSizeSinkI[1], dataSizeSinkI[2]),
                                      imagej=True)

    nZ = dataSizeSource[0]

    # resample in XY
    # chunk for each process
    Zlist = list(range(nZ))
    chunks = [Zlist[i::processes] for i in range(processes)]

    argdata = [(source, resampledXYFile, dataSizeSinkI, interpolation, chunk) for chunk in chunks]
    if processes == 1:
        _resampleXYParallel(argdata[0])
    else:
        pool = multiprocessing.Pool(processes=processes)
        pool.map(_resampleXYParallel, argdata)
        pool.close()

    # rescale in z
    resampledXY = io.readData(resampledXYFile)
    resampledData = numpy.zeros((dataSizeSinkI[0], dataSizeSinkI[1], dataSizeSinkI[2]),
                                dtype=data_type)

    for i in range(dataSizeSinkI[1]):  # faster if iterate over y
        if i % 50 == 0:
            log.verbose(("resampleData: Z: Resampling %d/%d" % (i, dataSizeSinkI[0])))
        resampledData[:, i] = cv2.resize(resampledXY[:, i], (dataSizeSinkI[2],
                                                                   dataSizeSinkI[0]),
                                            interpolation=interpolation)

    if cleanup:
        shutil.rmtree(processingDirectory)

    if not orientation is None:

        # reorient
        per = orientationToPermuation(orientation)
        resampledData = resampledData.transpose(per)

        # reverse orientation after permuting e.g. (-2,1) brings axis 2 to first axis and we can reorder there
        if orientation[0] < 0:
            resampledData = resampledData[::-1, :, :]
        if orientation[1] < 0:
            resampledData = resampledData[:, ::-1, :]
        if orientation[2] < 0:
            resampledData = resampledData[:, :, ::-1]

    log.verbose("resampleData: resampled data size: " + str(resampledData.shape))

    return io.writeData(sink, resampledData)


def resampleDataInverse(sink, source=None, dataSizeSource=None, orientation=None, resolutionSource=(4.0625, 4.0625, 3),
                        resolutionSink=(25, 25, 25),
                        processingDirectory=None, processes=bq3d.config.processes, cleanup=True,
                        interpolation='linear', **args):
    """Resample data inversely to :func:`resampleData` routine

    Arguments:
        sink (str or None): image to be inversly resampled (=sink in :func:`resampleData`)
        source (str or array): destination for inversly resmapled image (=source in :func:`resampleData`)
        dataSizeSource (tuple or None): target size of the resampled image
        orientation (tuple): orientation specified by permuation and change in sign of (1,2,3)
        resolutionSource (tuple): resolution of the source image (in length per pixel)
        resolutionSink (tuple): resolution of the resampled image (in length per pixel)
        processingDirectory (str or None): directory in which to perform resmapling in parallel, None a temporary directry will be created
        processes (int): number of processes to use for parallel resampling
        cleanup (bool): remove temporary files
        interpolation (str): method to use for interpolating to the resmapled image

    Returns:
        (array or str): data or file name of resampled image
    Notes:
        * resolutions are assumed to be given for the axes of the intrinsic
          orientation of the data and reference as when viewed by matplotlib or ImageJ
        * orientation: permuation of 1,2,3 with potential sign, indicating which
          axes map onto the reference axes, a negative sign indicates reversal
          of that particular axes
        * only a minimal set of information to detremine the resampling parameter
          has to be given, e.g. dataSizeSource and dataSizeSink
    """

    # assume we can read data fully into memory
    resampledData = io.readData(sink)

    dataSizeSink = resampledData.shape

    if isinstance(dataSizeSource, str):
        dataSizeSource = io.dataSize(dataSizeSource)

    dataSizeSource, dataSizeSink, resolutionSource, resolutionSink = resampleDataSize(dataSizeSource=dataSizeSource,
                                                                                      dataSizeSink=dataSizeSink,
                                                                                      resolutionSource=resolutionSource,
                                                                                      resolutionSink=resolutionSink,
                                                                                      orientation=orientation)

    dataSizeSinkI = orientDataSizeInverse(dataSizeSink, orientation)

    # flip axes back and permute inversely
    if not orientation is None:
        if orientation[0] < 0:
            resampledData = resampledData[::-1, :, :]
        if orientation[1] < 0:
            resampledData = resampledData[:, ::-1, :]
        if orientation[2] < 0:
            resampledData = resampledData[:, :, ::-1]

        # reorient
        peri = invert_orientation(orientation)
        peri = orientationToPermuation(peri)
        resampledData = resampledData.transpose(peri)

    # upscale in z
    interpolation = parse_interpolation(interpolation)

    resampledDataXY = numpy.zeros((dataSizeSinkI[0], dataSizeSinkI[1], dataSizeSource[2]), dtype=resampledData.dtype)

    for i in range(dataSizeSinkI[0]):
        if i % 25 == 0:
            log.vebose("resampleDataInverse: processing %d/%d" % (i, dataSizeSinkI[0]))

        # cv2.resize takes reverse order of sizes !
        resampledDataXY[i] = cv2.resize(resampledData[i], (dataSizeSource[2],
                                                                 dataSizeSinkI[1]),
                                              interpolation=interpolation)

    # upscale x, y in parallel
    if io.isFileExpression(source):
        files = source
    else:
        if processingDirectory is None:
            processingDirectory = tempfile.mkdtemp()
        files = os.path.join(sink[0], 'resample_\d{4}.tif')

    io.writeData(files, resampledDataXY)

    nZ = dataSizeSource[0]
    pool = multiprocessing.Pool(processes=processes)
    argdata = []
    for i in range(nZ):
        argdata.append((source, fl.fileExpressionToFileName(files, i), dataSizeSource, interpolation, i, nZ))
    pool.map(_resampleXYParallel, argdata)

    if io.isFileExpression(source):
        return source
    else:
        data = io.convertData(files, source)

        if cleanup:
            shutil.rmtree(processingDirectory)

        return data


def resamplePoints(source, sink=None, dataSizeSource=None, dataSizeSink=None, orientation=None,
                   resolutionSource=(4.0625, 4.0625, 3), resolutionSink=(25, 25, 25), **args):
    """Resample Points to map from original data to the coordinates of the resampled image

    The resampling of points here corresponds to he resampling of an image in :func:`resampleData`

    Arguments:
        pointSource (str or array): image to be resampled
        pointSink (str or None): destination of resampled image
        orientation (tuple): orientation specified by permuation and change in sign of (1,2,3)
        dataSizeSource (str, tuple or None): size of the data source
        dataSizeSink (tuple or None): target size of the resampled image
        resolutionSource (tuple): resolution of the source image (in length per pixel)
        resolutionSink (tuple): resolution of the resampled image (in length per pixel)

    Returns:
        (array or str): data or file name of resampled points
    Notes:
        * resolutions are assumed to be given for the axes of the intrinsic
          orientation of the data and reference as when viewed by matplotlib or ImageJ
        * orientation: permuation of 1,2,3 with potential sign, indicating which
          axes map onto the reference axes, a negative sign indicates reversal
          of that particular axes
        * only a minimal set of information to detremine the resampling parameter
          has to be given, e.g. dataSizeSource and dataSizeSink
    """
    log.info('resampling points')

    # size of data source
    if isinstance(dataSizeSource, str):
        dataSizeSource = io.dataSize(dataSizeSource)

    if isinstance(dataSizeSink, str):
        dataSizeSink = io.dataSize(dataSizeSink)

    dataSizeSource, dataSizeSink, resolutionSource, resolutionSink = resampleDataSize(dataSizeSource=dataSizeSource,
                                                                                      dataSizeSink=dataSizeSink,
                                                                                      resolutionSource=resolutionSource,
                                                                                      resolutionSink=resolutionSink,
                                                                                      orientation=orientation)

    points = io.readPoints(source)

    dataSizeSinkI = orientDataSizeInverse(dataSizeSink, orientation)

    # scaling factors
    scale = [float(dataSizeSource[i]) / float(dataSizeSinkI[i]) for i in range(3)]

    repoints = points.copy()
    for i in range(3):
        repoints[:, i] = repoints[:, i] / scale[i]

    # permute for non trivial orientation
    if not orientation is None:
        per = orientationToPermuation(orientation)
        repoints = repoints[:, per]

        for i in range(3):
            if orientation[i] < 0:
                repoints[:, i] = dataSizeSink[i] - repoints[:, i]

    if sink:
        return io.writePoints(sink, repoints)
    else:
        return repoints
