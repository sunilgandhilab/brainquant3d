# -*- coding: utf-8 -*-
"""
This is the main routine to run the individual routines to detect cells om
volumetric image data.
"""
import os
import numpy as np
import shutil
import uuid
from multiprocessing import Pool

from bq3d import config
from bq3d import io
from bq3d.utils.timer import Timer
from bq3d.utils.files import unique_temp_dir
from bq3d.utils.chunking import chunk_ranges
from bq3d.stack_processing.parallelization import processSubStack

import logging
from bq3d.utils.logger import set_console_level

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"

log = logging.getLogger(__name__)


def detect_cells(source, flow, processes=config.processes, log_level='info', **parameter):
    """Detect cells in data
    
    This is a main script to start running the cell detection.    
    
    Arguments:
        source (str or array): Image source
        flow (tuple): sequence of filters to call
        processes (int): parallel processes to spawn
        log_level (str): log level to print to console. can use 'verbose',
         a custom level higher than info but lower than debug.
        **parameter (dict): parameter for the image procesing sub-routines
    
    Returns:
        
    """
    timer = Timer()
    set_console_level(log_level)

    result = process_flow(source, flow=flow, processes=processes, **parameter)

    timer.log_elapsed("Total Cell Detection")
    return result

def process_flow(source,
                 flow,
                 x=None,
                 y=None,
                 z=None,
                 overlap=10,
                 min_sizes=(30, 30, 30),
                 aspect_ratio=(1, 10, 10),
                 output_properties = [],
                 size=config.thread_ram_max,
                 sink=None,
                 processes=config.processes):
    """ Runs a workflow.

    Args:
        source (str): path to image file to analyse.
        flow (tuple): images filters to run in sequential order.
            Entries should be a dict and will be passed to *bq3d.image_filters.filter_image*.
            The input image to each filter will the be output of the pevious filter.
        x (tuple): x range to analyse.
        y (tuple): y range to analyse.
        z (tuple): z range to analyse.
        overlap (int) :overlap between chunks in voxels
        min_sizes (tuple): minimum voxel size of chunk along each axis
        aspect_ratio (tuple): ratio to maintain between axis
        output_properties: (list): properties to include in output. See
        label_properties.region_props for more info
        size (int): max total size of substack in Gb
        sink (tuple): files to save detected cell info to.
            Coordinates will be saved to first file and properties segmented properties to second.
        processes (int): number of processes to use

    Returns:
        (tuple): coordinates, properties as tuple of np.ndarrays.
    """


    temp_dir = unique_temp_dir('bq3d')
    os.makedirs(temp_dir)
    source_fn = temp_dir / (str(uuid.uuid4()) + '.tif')
    log.verbose(f'Copying raw data to: {source_fn}')

    if len(output_properties) > 0 and output_properties[0] != 'centroid':
        output_properties.insert(0,'centroid')

    source = io.copyData(source, source_fn, x=x, y=y, z=z)

    unique_chunks, overlap_chunks = chunk_ranges(source, overlap=overlap, min_sizes=min_sizes,
                                                 aspect_ratio=aspect_ratio, size=size)
    log.verbose(f'Number of chunks: {len(unique_chunks)}')

    argdata = [(flow, output_properties, source, overlap_chunks[i], unique_chunks[i], temp_dir)
               for i in
               range(len(overlap_chunks))]
    try:
        if processes == 1:
            results = [processSubStack(*arg) for arg in argdata]
        else:
            pool = Pool(processes=processes, maxtasksperchild=1)
            results = pool.starmap(processSubStack, argdata)
            pool.close()
    except Exception as err:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise err

    # join results
    results = join_points(results, unique_chunks, overlap_chunks)
    results = jsonify_points(output_properties, results)

    shutil.rmtree(temp_dir, ignore_errors=True)
    if sink:
        io.writePoints(sink, results)

    return results


def join_points(results, unique_ranges, overlap_ranges):
    """Joins a list of points obtained from processing a stack in chunks. converts coordinates to absolute based on range.
    Only keeps points in given range.

    Arguments:
        results (list): list of point results from the individual sub-processes.
        unique_ranges (list): list of chunk indices of the unique portion of each chunk in form [[x_start,x_end],
        [y_start,y_end],[z_start,z_end])
        overlap_ranges: (list) list of chunk indices of the full chunk with overlap in form [[
        x_start,x_end], [y_start,y_end],[z_start,z_end])

    Returns:
       tuple: joined points, joined intensities
    """

    nchunks = len(results)

    all_coords  = []
    all_props   = []
    for r in results:
        if len(r) > 0:
            all_coords.append(r[0])
            all_props.append(r[1:])

    filtered_data = None

    for i in range(nchunks):
        coords = np.array(all_coords[i])
        props  = np.array(all_props[i]).T

        if len(coords) > 0:
            # covert to abs coordinates
            min_coord = tuple(rng[0] for rng in unique_ranges[i])
            max_coord = tuple(rng[1] for rng in unique_ranges[i])
            coords = coords + tuple(rng[0] for rng in overlap_ranges[i])

            # generate mask
            mask = np.logical_and(coords >= min_coord, coords < max_coord)
            mask = np.all(mask, axis=1)

            # join data
            data = np.concatenate((coords, props), axis=1)
            data = data[mask]

            if not isinstance(filtered_data, np.ndarray):
                filtered_data = data
            else:
                filtered_data = np.concatenate((filtered_data, data), axis=0)

    if not isinstance(filtered_data, np.ndarray):
        return np.zeros((0, 3))
    else:
        return filtered_data.T.tolist()


def jsonify_points(keys, values):

    res = {}
    i = 0
    for k in keys:
        if k == 'centroid':
            res['z'] = values[i]
            res['y'] = values[i+1]
            res['x'] = values[i+2]
            i += 3
        else:
            res[k] = values[i]
            i += 1

    if i != len(values):
        raise ValueError('Too many point properties for the number of keys')

    return res
