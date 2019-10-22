import math
import numpy as np
from typing import Union
from itertools import product

import clearmap3.IO as io
from clearmap3 import config

def range_to_slices(ranges:list):
    """ converts ranges to slice object

    Args:
        ranges (list): list of ranges in format [[start,stop],[start,stop],]

    Returns:
        list of sloce objects
    """

    return tuple(slice(*r, None) for r in ranges)


def ranges_from_sizes(sizes:list):
    """ generates ranges as (start,stop) given a list of chunk sizes.

    Args:
        sizes (list of chink sizes):

    Returns:
        (list): list of ranges as (start,stop)
    """

    indices = []
    offset = 0
    for r in sizes:
        indices.append([offset, offset+r]) # 1st al included, second excluded
        offset = offset+r

    return indices


def add_overlap(ranges:list, overlap, shape:tuple):
    """ adds overlap to each range. Will not extend past bounds of image.

    Args:
        ranges (list): list of ranges as (start,stop)
        overlap (int): amount of overlap to add
        shape (tuple): shape of original image

    Returns:
        (list): range with overlap added.
    """

    new_rngs = []
    for chunk in ranges:
        new_chunk = tuple([max(rng[0] - overlap, 0), min(rng[1] + overlap, shape[ax])] for ax, rng in enumerate(chunk))
        new_rngs.append(new_chunk)

    return new_rngs


def unique_slice(img_rng, unique_rng):
    """ removes overlap from an image substack

    Args:
        img_rng (list, tuple): list of ranges of the substack in full image as (start,stop) for each axis
        unique_rng (list, tuple): list of ranges of the unique portion of the substack in full image

    Returns:
        (np.ndarray): cropped image with overlap removed
    """

    new_ranges = []
    for ax, rng in enumerate(unique_rng):
        start_idx = rng[0] - img_rng[ax][0]
        stop_idx = start_idx + (rng[1] - rng[0])
        new_ranges.append([start_idx, stop_idx])

    return range_to_slices(new_ranges)


def chunk_ranges(source:Union[str, np.ndarray],
                  overlap:int = 10,
                  min_sizes:tuple = (30,30,30),
                  aspect_ratio:tuple = (1,10,10),
                  size:float = config.thread_ram_max, **kwargs):
    """ chunks image into equally sized 3d chunks.

    Args:
        source (str,np.ndarray): image to chunkimage
        overlap (int): overlap between chunks in voxels
        min_sizes (tuple): minimum voxel size of chunk along each axis
        aspect_ratio (tuple): ratio to maintain between axis
        size (int): max total size of substack in Gb

    Returns:
        (list) list of chunk indices each index is in form [[x_start,x_end],[y_start,y_end],[z_start,z_end])


    """
    source = io.readData(source)

    if not source.ndim == 3:
        raise ValueError('source must be 3 dimensional')

    if not len(aspect_ratio) == source.ndim:
        raise ValueError('aspect_ratio must have same number of dimensions as source')

    if any([overlap > s for s in min_sizes]):
        raise ValueError('overnlap cannot be smaller thn any value in min_size')

    chunk_size = size * (1.28 * 10**8) / source.itemsize # in voxels


    indices = []
    if chunk_size < source.size:
        c = (chunk_size / np.prod(aspect_ratio)) ** (1/3) # common denominator for each axis

        for a, axis in enumerate(source.shape):
            # bounding chunks won't be perfectly optimized because assumning the max unique size
            # of a chunk is size - 2*overlap
            max_size = math.floor(c * aspect_ratio[a]) - (2 * overlap)

            if min_sizes[a] > max_size:
                raise ValueError(f'min_size along axis {a} too large. results in substack larger than max chunk.')

            # if does not chunk evenly. equalize chunk sizes by moving voxels from equal chunks to the unequal chunk
            if axis % max_size:
                n_chunks = math.ceil(axis / max_size)
                chunk_size = axis // n_chunks

                if chunk_size < min_sizes[a]:
                    raise ValueError(f'cannot chunk along axis {a}. min_size too large or size to small')

                chunk_sizes = [chunk_size] * (n_chunks - 1)
                chunk_sizes.append(axis - np.sum(chunk_sizes)) # contains remainder of voxels from rounding

            else:
                chunk_sizes = [max_size] * (axis / max_size)

            indices.append(ranges_from_sizes(chunk_sizes))

        unique_chunks = list(product(*indices))
        overlap_chunks = add_overlap(unique_chunks, overlap, source.shape)

        return unique_chunks, overlap_chunks
    else:
        return [[[0, ax] for ax in source.shape]] * 2

def chunk_ranges_z_only(source:Union[str, np.ndarray],
                  overlap:int = 10,
                  min_size:tuple = 30,
                  size:float = config.thread_ram_max, **kwargs):
    """ chunks image into equally sized 3d chunks.

    Args:
        source (str,np.ndarray): image to chunkimage
        overlap (int): overlap between chunks in voxels
        min_size (tuple): minimum voxel size of chunk along z
        size (int): max total size of substack in Gb

    Returns:
        (list) list of chunk indices each index is in form [[x_start,x_end],[y_start,y_end],[z_start,z_end])


    """
    source = io.readData(source)

    if not source.ndim == 3:
        raise ValueError('source must be 3 dimensional')


    if overlap > min_size:
        raise ValueError('overnlap cannot be smaller thn any value in min_size')

    chunk_size = size * (1.28 * 10**8) / source.itemsize # in voxels

    indices = []
    if chunk_size < source.size:

        a = source.shape[0]
        # bounding chunks won't be perfectly optimized because assumning the max unique size
        # of a chunk is size - 2*overlap
        max_size = math.floor(source.size / chunk_size) - (2 * overlap)

        if min_size > max_size:
            raise ValueError(f'min_size along axis {a} too large. results in substack larger than max chunk.')

        # if does not chunk evenly. equalize chunk sizes by moving voxels from equal chunks to
        # the unequal chunk

        if a % max_size:
            n_chunks = math.ceil(a / max_size)
            chunk_size = a // n_chunks

            if chunk_size < min_size:
                raise ValueError(f'cannot chunk along axis {a}. min_size too large or size to small')

            chunk_sizes = [chunk_size] * (n_chunks - 1)
            chunk_sizes.append(a - np.sum(chunk_sizes))  # contains remainder of voxels from
            # rounding

        else
            chunk_sizes = [max_size] * (a / max_size)

        indices.append(ranges_from_sizes(chunk_sizes))

        unique_chunks = [[ranges_from_sizes(source.shape[1]),
                          ranges_from_sizes(source.shape[2]),
                          z] for z in indices]

        overlap_chunks = add_overlap(indices, overlap, source.shape)
        overlap_chunks = [[ranges_from_sizes(source.shape[1]),
                          ranges_from_sizes(source.shape[2]),
                          z] for z in overlap_chunks]

        return unique_chunks, overlap_chunks
    else:
        return [[[0, ax] for ax in source.shape]] * 2