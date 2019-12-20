import numpy as np
import tifffile as tif
from multiprocessing import Pool
import cv2

from bq3d import io

from bq3d import config
from bq3d.image_filters import filter_manager
from bq3d.image_filters.filter import FilterBase

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


class ExtractSurface(FilterBase):
    """ Extracts only the surface layer of the largest object in an image.

    Attributes:
        input (array): Image to pass through filter.
        threshold (int or None): minimum threshold to binarize image. If None input must already be binary.
        offset (int): thickness of surface to extract in pixels.
        min_size (int): minimum size of objects in pixels to include as part of sample.
        save_mask (str or None): save mask to file.
        output (array): Filter result.
    """

    def __init__(self):
        self.threshold      = None
        self.offset_x       = 100
        self.offset_z       = 100
        self.min_size       = 10000

        self.save_mask = False
        self.processes = config.processes
        super().__init__(temp_dir=True)

    def _generate_output(self):

        mask_f    = self.temp_dir / 'mask.tif'
        erode_f = self.temp_dir / 'erode.tif'

        self.log.info('preparing temp files')
        tif.tifffile.memmap(mask_f, dtype=np.uint8, shape=self.input.T.shape, bigtiff=True)
        tif.tifffile.memmap(erode_f, dtype=np.uint8, shape=self.input.T.shape, bigtiff=True)

        mask     = io.readData(mask_f)
        erode_im = io.readData(erode_f)

        # extract brain mask
        z_idxs   =  list(range(mask.shape[-1]))
        z_chunks = [z_idxs[i::self.processes] for i in range(self.processes)]
        args = [(self.input.filename, mask.filename, z_idxs, self.threshold, self.min_size) for z_idxs in z_chunks]
        if self.processes == 1:
            _parallel_mask(*args)
        else:
            pool = Pool(self.processes)
            pool.map(_parallel_mask, args)
            pool.close()

        erode(mask.filename, erode_im.filename, self.offset_z, self.offset_x, processes = self.processes)

        # merge erosions and mask
        args = [(mask.filename, erode_im.filename, z_idxs) for z_idxs in z_chunks]
        if self.processes == 1:
            _parallel_merge_mask(*args)
        else:
            pool = Pool(self.processes)
            pool.map(_parallel_merge_mask, args)
            pool.close()

        if self.save_mask:
            self.log.info(f'saving mask to {self.save_mask}')
            io.writeData(self.save_mask, mask)

        # mask input
        # not working
        self.log.info(f'masking image')
        for z in range(self.input.shape[2]):
            im = self.input[:,:,z]
            im[mask[:,:,z] == 0] = 0
            self.input[:, :, z] = im

        return self.input

filter_manager.add_filter(ExtractSurface())

def erode(mask, sink, offset_z, offset_x, processes = 1):
    # offset surface in z

    mask = io.readData(mask)
    sink = io.readData(sink)

    z_idxs = list(range(mask.shape[-1]))
    z_chunks = [z_idxs[i::processes] for i in range(processes)]
    z_args = [(mask.filename, sink.filename, z_idxs, offset_x) for z_idxs in z_chunks]

    x_idxs = list(range(mask.shape[0]))
    x_chunks = [x_idxs[i::processes] for i in range(processes)]
    x_args = [(sink.filename, sink.filename, x_idxs, offset_z) for x_idxs in x_chunks]

    if processes == 1:
        _parallel_erode_z(*z_args)
        _parallel_erode_x(*x_args)
    else:
        pool = Pool(processes)
        pool.map(_parallel_erode_z, z_args)
        pool.map(_parallel_erode_x, x_args)
        pool.close()

    return sink


def _parallel_mask(args):
    source, sink, idxs, threshold, min_size = args

    source = io.readData(source)
    sink = io.readData(sink)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    for i in idxs:
        print(f'Mask slice {i}')
        # mask image
        if threshold:
            _, img = cv2.threshold(source[..., i], threshold, 255, cv2.THRESH_BINARY)
            # extract only brain mask
            nlabels, labels, stats, centroid = cv2.connectedComponentsWithStats(img.astype(np.uint8))
            sample_labels = np.where(stats[1:,..., 4] > min_size)[0] + 1  # get largest shapes
            img = np.isin(labels, sample_labels).astype(np.uint8)
        else:
            img = np.array(source[..., i]).copy()
        # remove holes in brain mask
        contours, hier = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours.sort(key=len)
        # keep biggest contours
        img = np.zeros(img.shape)
        for c in contours[-5:]:
            img = cv2.drawContours(img, [c], 0, 255, -1)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        sink[..., i] = img.astype(np.uint8)

def _parallel_erode_z(args):

    source, sink, idxs, offset = args

    source = io.readData(source)
    sink = io.readData(sink)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for i in idxs:
        print(f'Erode slice {i} along z')
        img = source[..., i].astype(np.uint8)
        sink[..., i] = cv2.erode(img, kernel, iterations=offset)

def _parallel_erode_x(args):

    source, sink, idxs, offset = args

    source = io.readData(source)
    sink = io.readData(sink)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for i in idxs:
        print(f'Erode slice {i} along x')
        img = source[i].astype(np.uint8)
        sink[i] = cv2.erode(img, kernel, iterations=offset)

def _parallel_merge_mask(args):
    mask, erosion, idxs = args

    bkg_mask  = io.readData(mask)
    erosion = io.readData(erosion)

    for z in idxs:
        print(f'Generating output for slice {z}')
        merge = np.full(bkg_mask.shape[:2], 255, dtype=np.uint8)
        merge[bkg_mask[:,:,z] == 0] = 0
        merge[erosion[:,:,z] != 0] = 0
        bkg_mask[:,:,z] = merge
