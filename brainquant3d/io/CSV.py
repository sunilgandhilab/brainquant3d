# -*- coding: utf-8 -*-
"""
Interface to write csv files of cell coordinates / intensities

The module utilizes the csv file writer/reader from numpy.
    (5, 3)
"""

import numpy as np
import pandas as pd

from clearmap3 import io

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
    data = pd.read_csv(filename)

    #imagej format
    if {'Slice', 'Y', 'X'} <= set(data.columns):
        return np.array([data['Slice'], data['Y'], data['X']]).T

    else:
        raise ValueError(f'could not infer points format from {filename}')


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
