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

from clearmap3 import config
import clearmap3.IO as io
from clearmap3.utils.timer import Timer
from clearmap3.utils.files import unique_temp_dir
from clearmap3.stack_processing.chunking import chunk_ranges
from clearmap3.stack_processing.parallelization import processSubStack

import logging
from clearmap3.utils.logger import set_console_level
log = logging.getLogger(__name__)


def detect_cells(source, flow, sink=None, processes=config.processes, log_level='info', **parameter):
    """Detect cells in data
    
    This is a main script to start running the cell detection.    
    
    Arguments:
        source (str or array): Image source
        flow (tuple): sequence of filters to call
        sink (str or None): destination for the results.
        processes (int): parallel processes to spawn
        log_level (str): log level to print to console. can use 'verbose',
         a custom level higher than info but lower than debug.
        **parameter (dict): parameter for the image procesing sub-routines
    
    Returns:
        
    """
    timer = Timer()
    set_console_level(log_level)

    result = process_flow(source, sink=sink, flow=flow, processes=processes, **parameter)

    timer.log_elapsed("Total Cell Detection")
    return result

def process_flow(source,
                 flow,
                 x=None,
                 y=None,
                 z=None,
                 overlap=10,
                 min_sizes=(30, 30, 30),
                 aspect_ratio=(10, 10, 1),
                 size=config.thread_ram_max,
                 sink=None,
                 processes=config.processes):
    """ Runs a workflow.

    Args:
        source (str): path to image file to analyse.
        flow (tuple): images filters to run in sequential order.
            Entries should be a dict and will be passed to *clearmap3.image_filters.filter_image*.
            The input image to each filter will the be output of the pevious filter.
        x (tuple): x range to analyse.
        y (tuple): y range to analyse.
        z (tuple): z range to analyse.
        overlap (int) :overlap between chunks in voxels
        min_sizes (tuple): minimum voxel size of chunk along each axis
        aspect_ratio (tuple): ratio to maintain between axis
        size (int): max total size of substack in Gb
        sink (tuple): files to save detected cell info to.
            Coordinates will be saved to first file and properties segmented properties to second.
        processes (int): number of processes to use

    Returns:
        (tuple): coordinates, properties as tuple of np.ndarrays.
    """

    temp_dir = unique_temp_dir('clearmap')
    os.makedirs(temp_dir)
    source = io.copyData(source, temp_dir / (str(uuid.uuid4()) + '.tif'), x=x, y=y, z=z, processes=1)

    unique_chunks, overlap_chunks = chunk_ranges(source, overlap=overlap, min_sizes=min_sizes,
                                                 aspect_ratio=aspect_ratio, size=size)
    log.verbose(f'Number of chunks: {len(unique_chunks)}')

    argdata = [(flow, source, overlap_chunks[i], unique_chunks[i], temp_dir) for i in
               range(len(overlap_chunks))]
    try:
        if processes == 1:
            results = [processSubStack(*arg) for arg in argdata]
        else:
            pool = Pool(processes=processes, maxtasksperchild=1)
            results = pool.starmap(processSubStack, argdata)
    except Exception as err:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise err


    results = join_points(results, unique_chunks)

    shutil.rmtree(temp_dir, ignore_errors=True)
    if sink:
        io.writePoints(sink, results)

    return results


def join_points(results, absolute_ranges):
    """Joins a list of points obtained from processing a stack in chunks. converts coordinates to absolute based on range.
    Only keeps points in given range.

    Arguments:
        results (list): list of point results from the individual sub-processes
        absolute_ranges (list or None): list of all sub-stack information, see :ref:`SubStack`

    Returns:
       tuple: joined points, joined intensities
    """

    nchunks     = len(results)
    pointlist   = [results[i][0] for i in range(nchunks)]
    intensities = [results[i][1] for i in range(nchunks)]

    filtered_results  = []
    filtered_resultsi = []

    for i in range(nchunks):
        cts = pointlist[i]
        cti = intensities[i]

        if cts.size > 0:
            # covert to abs coordinates
            min_coord = tuple(rng[0] for rng in absolute_ranges[i])
            max_coord = tuple(rng[1] for rng in absolute_ranges[i])
            cts = cts + min_coord

            # remove points out of range
            mask = np.logical_and(cts >= min_coord, cts < max_coord)
            mask = np.all(mask, axis=1)
            filtered_results.append(cts[mask])

            # remove points in intensity array as well
            filtered_resultsi.append(cti[mask])

    if not results:
        return np.zeros((0, 3))
    else:
        return np.concatenate(filtered_results), np.concatenate(filtered_resultsi)
