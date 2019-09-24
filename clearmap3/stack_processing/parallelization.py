
import os
import numpy as np
import uuid
from clearmap3.utils.timer import Timer
import clearmap3.IO as io
from clearmap3.IO.FileList import splitFileExpression
import logging

from clearmap3.utils.chunking import unique_slice
from clearmap3.image_filters.functions import filter_image
from clearmap3.analysis.label_properties import label_props

log = logging.getLogger(__name__)

#define the subroutine for the processing
def processSubStack(flow, source, overlap_indices, unique_indices, temp_dir):
    """ Helper to process stack in parallel

    Args:
        flow (tuple): images filters to run in sequential order.
            Entries should be a dict and will be passed to *clearmap3.image_filters.filter_image*.
            The input image to each filter will the be output of the pevious filter.
        source (str): path to image file to analyse.
        overlap_indices (tuple or list): list of indices as [start,stop] along each axis to analyse.
        unique_indices (tuple or list): list of indices as [start,stop] along each axis corresponding
            to the non-overlapping portion of the image being analyzed.
        temp_dir (str): temp dir to be used for processing.

    Returns:

    """
    timer = Timer()

    xRng, yRng, zRng = overlap_indices
    log.info(f'chunk ranges: x= {xRng}, y= {yRng}, z= {zRng}')

    #memMap routine
    if not os.path.exists(temp_dir):
        os.mkdir(temp_dir)

    mmapFile = os.path.join(temp_dir, str(uuid.uuid4())) + '.tif'
    log.info('Creating memory mapped substack at: {}'.format(mmapFile))

    img = io.copyData(source, mmapFile, x=xRng, y=yRng, z=zRng, returnMemmap=True)

    rawFile = os.path.join(temp_dir, str(uuid.uuid4())) + '.tif'
    log.info('Creating raw substack at: {}'.format(rawFile))
    raw = io.copyData(img.filename, rawFile, returnMemmap=True)

    # if a flow
    filtered_im = img
    for p in flow:
        params = dict(p)
        filter = params.pop('filter')

        if 'save' in params:
            save = params.pop('save')
        else:
            save = False

        filtered_im = filter_image(filter, filtered_im, **params)

        # save intermediate output
        if save:
            log.info(f'Saving output to {save}')
            h, ext, dfmt = splitFileExpression(save)

            for z in range(*zRng):
                fname = h + (dfmt % z) + ext
                if not os.path.isfile(fname):
                    io.empty(fname, io.dataSize(source)[:2], filtered_im.dtype)

            unique = filtered_im[unique_slice(overlap_indices, unique_indices)]
            io.writeData(save, unique, substack=unique_indices)

    # get label properties and return
    centers, cintensity, csize, labels = label_props(raw, filtered_im)
    cintensity = np.array(cintensity)
    csize = np.array(csize)
    centers = np.array(centers)

    seg = (centers, np.vstack((cintensity, csize)).transpose())

    os.remove(mmapFile)
    timer.log_elapsed(prefix='Processed chunk')
    return seg
