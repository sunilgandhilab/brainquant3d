import numpy as np
import cv2
import json
import os

import clearmap3.IO as io
from clearmap3.image_filters.functions import filter_image

def classify_rois(source, classifier, visualize = True, **kwargs):

    data = io.readData(source)
    prob = filter_image('PixelClassification', data, project = classifier)
    thresh = filter_image('Label', prob, **kwargs).astype(np.uint16)

    # generate ROIs
    roi_coords = []
    roi_dicts = []
    for label in np.unique(thresh):
        iso = (thresh == label).astype(np.uint8)
        # extract contours
        contour,_ = cv2.findContours(iso, 1,2)
        epsilon = 0.05*cv2.arcLength(contour[0],True)
        approx = cv2.approxPolyDP(contour[0],epsilon,True)

        coord_ls = np.flip(approx,axis=2).flatten().tolist() # have to flip for pacu to read coord correctly
        coord_str = str(coord_ls).replace (' ', '').replace('[','').replace(']','')
        dt = {'polygon': coord_str, 'roi_id': label.item()+1} # convert numpy dtype to python. cant have index 0.

        roi_dicts.append(dt)
        roi_coords.append(approx)

    # output
    file,_ = os.path.splitext(source)
    roi_sink = f'{file}-rois.json'
    with open(roi_sink, 'w') as outfile:
        json.dump(roi_dicts, outfile)

    if visualize:
        sink = np.zeros(iso.shape, dtype = iso.dtype)
        cv2.drawContours(sink,roi_coords,-1,255,1)
        io.writeData('/mnt/c/swap/cont.tif', sink)
        io.writeData('/mnt/c/swap/prob.tif', prob)
        io.writeData('/mnt/c/swap/thresh.tif', thresh)

    return roi_sink

source = '/Volumes/data/Ricardo/2P/analysis_tools/2P_classified_sets/moco_aligned_r022-1_001_005-r022-1_001_005_BINO-2019-09-24-11-01-10-mean-projection.tif'
classifier = '/Volumes/data/Ricardo/2P/analysis_tools/2P_classified_sets/2Pdata.ilp'

classify_rois(source, classifier, sigmas=(.2,.2), algorithm='identity-preserving hysteresis '
                                                             'thresholding',
              min_size=2, max_size=80, core_threshold=.6, final_threshold=.3)

print('Done')
