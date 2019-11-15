import json
import numpy as np

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