import numpy as np
from ._diffuse import _diffuse

from bq3d.image_filters.filters.helpers.structure_element import structure_element_binary, \
    _offsets_to_raveled_neighbors


def diffuse(mask, seeds, output, iterations=1, k_const=1, threshold=0):

    connectivity, offset = structure_element_binary(mask.ndim, connectivity=3,
                                                  offset=None)

    flat_neighborhood = _offsets_to_raveled_neighbors(
        mask.shape, connectivity, center=offset)

    image_strides = np.array(mask.strides, dtype=np.intp) // mask.itemsize

    if not isinstance(seeds, np.ndarray):
        seeds = np.array(seeds)

    # assume concentration is max outside tissue
    output[mask == 0] = 1

    for i in range(iterations):
        print(f'iteration: {i}')
        _diffuse(mask, output, seeds, flat_neighborhood, image_strides, threshold, k_const)

    return output