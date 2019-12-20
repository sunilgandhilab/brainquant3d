import json
import numpy as np

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


def writePoints(filename, data):

    if isinstance(data, np.ndarray):
        out = {}
        out['z'] = data[0]
        out['y'] = data[1]
        out['x'] = data[2]

    with open(filename, 'w') as f:
        json.dump(data, f)

    return filename


def readPoints(filename):

    with open(filename) as json_file:
        data = json.load(json_file)

    if not {'z', 'y', 'x'} <= set(data):
        raise ValueError('z,y,x must be rpesent as keys in dict json')

    return np.array([data['z'], data['y'], data['x']]).T


def writeData(filename, data):

    with open(filename, 'w') as f:
        json.dump(data, f)

    return filename


def readData(filename):

    with open(filename) as json_file:
        data = json.load(json_file)

    return data
