"""
Interface to ANTs for alignment and transformation of volumetric data

The ANTs full documentation can be found `here <>`
The ANTsPy documentation can be found `here <https://antspy.readthedocs.io/en/latest/>`

Summary
-------
    * ANTs finds the tranformation parameters to match two images
    * the fixed image is typically the image to be registered
    * the moving image is typically the reference image
    * Transforms can be applied to images or point coordinates and direction is assumed to be from fixed -> moving
    * ANTs allows for inverting transforms which can be usefull if better result are obtained with a fixed reference image.
    * point arrays are assumed to be in (x,y,z) coordinates consistent with (x,y,z) array represenation of images in ClearMap

Main routines are: :func:`alignData`, :func:`transformImage` and :func:`transformPoints`.
"""

import os
import shutil
import ants
import numpy as np

from clearmap3 import config
from clearmap3 import io
from clearmap3.utils.logger import log_parameters

import logging
log = logging.getLogger(__name__)

def alignData(fixedImage, movingImage, resultDirectory = None, type_of_transform = 'SyNRA', **kwargs):
    """Align images using elastix, estimates a transformation :math:`T:` fixed image :math:`\\rightarrow` moving image.

    Arguments:
        fixedImage (str): image source of the fixed image (typically the reference image)
        movingImage (str): image source of the moving image (typically the image to be registered)
        resultDirectory (str or None): result directory for transform parameters. None saves to clearmap3default temp file
        transform: (str): type of transform to apply as defined in 'ants.registration' type_of_transform
        **kwargs: additional arguments to pass to 'ants.registration'

    Returns:
        str: path to elastix result directory
    """

    log_parameters(fixedImage = fixedImage, movingImage = movingImage, resultDirectory = resultDirectory, type_of_transform = type_of_transform)

    # setup input
    mi = ants.from_numpy(io.readData(movingImage))
    fi = ants.from_numpy(io.readData(fixedImage))

    # setup output directory
    if not resultDirectory:
        tmp_folder = os.path.join(config.temp_dir, 'ANTs')
        resultDirectory = tmp_folder
    resultDirectory = resultDirectory + '/' if not resultDirectory.endswith('/') else resultDirectory #make sure ends with '/'
    os.makedirs(resultDirectory, exist_ok=True)

    # run
    result = ants.registration(fi, mi, type_of_transform = type_of_transform, outprefix = resultDirectory, verbose=True, **kwargs)

    # save output
    io.writeData(os.path.join(resultDirectory, 'result.tif'), result['warpedmovout'].numpy())

    # cleanup#
    if not resultDirectory:
        shutil.rmtree(tmp_folder)

    return resultDirectory

def transformImage(image, reference, transformDirectory, sink = None, invert = False, interpolation = 'bspline'):
    """Transform a raw data set to reference using the ANTs alignment results

    Arguments:
        source (str or array): image source to be transformed
        reference (str or array): fixed image from transform
        transformDirectory (str): Directory containing ANTS transform parameters
        sink (str or None): image sink to save transformed image to.
        interpolation (str): ANTS interpolator to use for generating image.

    Returns:
        array or str: file name of the transformed data. If sink is None, return array
    """

    log.info('transforming image with ' + transformDirectory)
    log.info ('invert: {}'.format(invert))

    # get image and tranform
    im = ants.from_numpy(io.readData(image))
    ref = ants.from_numpy(io.readData(reference))
    composite_trans = _compose_transforms(transformDirectory, invert = invert)
    # apply transforms
    res = composite_trans.apply_to_image(im, ref, interpolation = interpolation)
    # output
    if isinstance(sink, str):
        return io.writeData(sink, res.numpy())
    else:
        return res.numpy()


def transformPoints(points_source, transformDirectory, sink = None, invert = False):
    """Transform coordinates math:`x` via elastix estimated transformation to :math:`T(x)`
    Note the transformation is from the fixed image coorindates to the moving image coordiantes.

    Arguments:
        points_source (str or numpy.array): source of the points.

    Returns:
        array or str: array or file name of transformed points
    """

    log.info('transforming points with ' + transformDirectory)
    log.info ('invert: {}'.format(invert))

    pts = io.readPoints(points_source).tolist()
    composite_trans = _compose_transforms(transformDirectory, invert=invert)

    trans_pts = []
    for i in pts:
        trans_pts.append(composite_trans.apply_to_point(i))

    res= np.array(trans_pts)

    if isinstance(sink, str):
        return io.writeData(sink, res)
    else:
        return res

def _compose_transforms(transformDirectory, invert = False):
    """Strings together transofrm files in the correct order to apply a transform.
    """

    transforms = []
    if not invert:
        if '1Warp.nii.gz' in os.listdir(transformDirectory):
            SyN_file = os.path.join(transformDirectory, '1Warp.nii.gz')
            transforms.append(ants.transform_from_displacement_field(ants.image_read(SyN_file)))
        if '0GenericAffine.mat' in os.listdir(transformDirectory):
            affine_file = os.path.join(transformDirectory, '0GenericAffine.mat')
            transforms.append(ants.read_transform(affine_file))
    else:
        if '0GenericAffine.mat' in os.listdir(transformDirectory):
            affine_file = os.path.join(transformDirectory, '0GenericAffine.mat')
            transforms.append(ants.read_transform(affine_file).invert())
        if '1InverseWarp.nii.gz' in os.listdir(transformDirectory):
            inv_file = os.path.join(transformDirectory, '1InverseWarp.nii.gz')
            transforms.append(ants.transform_from_displacement_field(ants.image_read(inv_file)))

    return ants.compose_ants_transforms(transforms)
