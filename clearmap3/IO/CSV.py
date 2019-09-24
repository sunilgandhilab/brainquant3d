# -*- coding: utf-8 -*-
"""
Interface to write csv files of cell coordinates / intensities

The module utilizes the csv file writer/reader from numpy.
    (5, 3)
"""

import numpy as np
import pandas as pd

import clearmap3.IO as io

import logging
log = logging.getLogger(__name__)


def writePoints(filename, points, **args):
    """Write point data to csv file
    
    Arguments:
        filename (str): file name
        points (array): point data
    
    Returns:
        str: file name
    """
    
    np.savetxt(filename, points, delimiter=',', newline='\n', fmt='%.5e')
    return filename


def readPoints(filename, **args):
    """Read point data to csv file
    
    Arguments:
        filename (str): file name
        args: arguments for :func:`~clearmap3.IO.pointsToRange`
    
    Returns:
        str: file name
    """

    points = np.genfromtxt(filename, delimiter=',')
    return io.pointsToRange(points, **args)


def writeData(filename, data, **kwargs):
    """Write data to csv file

    Arguments:
        filename (str): file name
        data (array or dataframe): data to write

    Returns:
        str: file name
    """

    if isinstance(data, np.ndarray):
        np.savetxt(filename, data, delimiter=',', newline='\n', fmt='%.5e')
    if isinstance(data, pd.DataFrame):
        data.to_csv(filename)
    return filename
