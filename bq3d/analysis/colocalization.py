import numpy as np
from typing import Union
from scipy.spatial.distance import cdist

from bq3d import config

def grouped_distances(points:np.array, groups: dict, return_minimum = False):
    """ gets the minimum euclidian distance of each point to each coordinate group.

    Args:
        points (np.array): coords in [[x,y,z],...]
        groups (dict): groups of coords in {'group': [[x,y,z],...], ...}
        return_minimum (bool): return the minimum distance between a point and any group.

    Returns:

    """
    distances = dict.fromkeys(groups.keys(),[])
    for id, group in groups.items():
        print(id)
        distances[id] = list(point_distance(points, group))

    if return_minimum:
        minimums = []
        for i, pt in enumerate(points):
           minimums.append(min([distances[key][i] for key in distances]))

        return distances, minimums
    else:
        return distances

def point_distance(point:Union[np.array,list], group:np.array, chunksize_gb:int = config.thread_ram_max):
    """ Returns minimum euclidian distance between a coordinate and group of coordinates.

    Args:
        point: (np.array, list): coord to test distance in [x,y,z]
        group: (np.array): coords to test against as [[x,y,z],...]. can be a single point.
        chunksize_gb: (int): max chunking size in Gb to limit RAM usage.

    Returns:
        (float) calculated minimum distances. 0 if point in the group
    """

    result_size_bytes = (point.size / 3) * (group.size / 3) * np.dtype(np.double).itemsize
    chunksize_bytes = (10 ** 9 * chunksize_gb)

    if result_size_bytes > chunksize_bytes:  # sink for cdist is double
        n_chunks = np.ceil(result_size_bytes / chunksize_bytes)
        chunks = np.array_split(point, n_chunks)

        results = []
        for chunk in chunks:
            res = cdist(chunk, group).min(axis=1)
            results.extend(res)

        return np.array(results)

    else:
        return cdist(point, group).min(axis=1)