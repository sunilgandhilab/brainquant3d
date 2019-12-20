"""
Interface to Elastix for alignment of volumetric data

The elastix documentation can be found `here <http://elastix.isi.uu.nl/>`_.

Summary
-------
    * Elastix finds the tranformation parameters to match two images
    * the fixed image is typically the image to be registered
    * the moving image is typically the reference image
    * Transforms can be applied to images or point coordinates and direction is assumed to be from fixed -> moving
    * ANTs allows for inverting transforms which can be usefull if better result are obtained with a fixed reference image.
    * point arrays are assumed to be in (x,y,z) coordinates consistent with (x,y,z) array represenation of images in brainquant3d


Main routines are: :func:`alignData`, :func:`transformImage` and :func:`transformPoints`.
"""

import os
import tempfile
import shutil
import numpy as np
import re

from bq3d import config
from bq3d import io

import logging

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"

log = logging.getLogger(__name__)

Initialized = False
"""bool: True if the elastixs binarys and paths are setup

Notes:
    - setup in :func:`initializeElastix`
"""

def initialize_Elastix():
    """Checks that all Elastix files are found """

    # check binaries
    if config.elastix_path is None:
        raise RuntimeError('Cannot find elastix path %s set Elastix path in brainquant3d.conf accordingly!' % bq3d.config.tranformix_binary)
    if config.elastix_binary is None:
        raise RuntimeError('Cannot find elastix binary %s, set Elastix path in brainquant3d.conf accordingly!' % bq3d.config.elastix_binary)
    if config.tranformix_binary is None:
        raise RuntimeError('Cannot find transformix binary %s set Elastix path in brainquant3d.conf accordingly!' % bq3d.config.tranformix_binary)

    # set elastix library path environment variable
    path = config.elastix_path

    if 'LD_LIBRARY_PATH' in os.environ:
        lp = os.environ['LD_LIBRARY_PATH']
        if not path in lp.split(':'):
            os.environ['LD_LIBRARY_PATH'] = lp + ':' + path
    else:
        os.environ['LD_LIBRARY_PATH'] = path

    log.verbose("Elastix sucessfully initialized from path: %s" % path)
    return path

##############################################################################
### Basic interface routines
##############################################################################

def getTransformParameterFiles(resultdir):
    """Finds and returns the transformation parameter file generated by elastix

    Notes:
        In case of multiple transformation parameter files the top level file is returned(highest number)

    Arguments:
        resultdir (str): path to directory of elastix results

    Returns:
        str: file name of the first transformation parameter file
    """

    files = os.listdir(resultdir)
    files = [x for x in files if re.match('TransformParameters.\d.txt', x)]
    files.sort()

    if not files:
        raise RuntimeError('Cannot find a valid transformation file in ' + resultdir)


    param_files = [os.path.join(resultdir, f) for f in files]

    return param_files


def setPathTransformParameterFiles(resultdir):
    """Replaces relative with abolsute path in the parameter files in the result directory

    Notes:
        When elastix is not run in the directory of the transformation files
        the aboslute path needs to be given in each transformation file
        to point to the subsequent transformation files. This is done via this
        routine

    Arguments:
        resultdir (str): path to directory of elastix results
    """

    log.info (resultdir)
    files = os.listdir(resultdir)
    #find elastix tranformation files
    files = [x for x in files if re.match('TransformParameters.\d.txt', x)]
    files.sort()

    if not files:
        raise RuntimeError('Cannot find a valid transformation file in ' + resultdir)

    rec = re.compile("\(InitialTransformParametersFileName \"(?P<parname>.*)\"\)")

    for f in files:
        fh, tmpfn = tempfile.mkstemp()
        ff = os.path.join(resultdir, f)

        with open(tmpfn, 'w') as newfile:
            with open(ff) as parfile:
                for line in parfile:
                    #print line
                    m = rec.match(line)
                    if m is not None:
                        pn = m.group('parname')
                        if pn != 'NoInitialTransform':
                            pathn, filen = os.path.split(pn)
                            filen = os.path.join(resultdir, filen)
                            newfile.write(line.replace(pn, filen))
                        else:
                            newfile.write(line)
                    else:
                        newfile.write(line)
        os.close(fh)
        os.remove(ff)
        # shutil "move" can produce permission errors when moving files between the windows sybsystem.
        # and the main Windows system. In this case, copy file contents to new location instead.
        try:
            shutil.move(tmpfn, ff)
        except PermissionError:
            with open(tmpfn) as tmpfn_file:
                with open(ff, 'w') as ff_file:
                    ff_file.writelines(tmpfn_file.readlines())
            os.remove(tmpfn)


def parseElastixOutputPoints(filename, indices = True):
    """Parses the output points from the output file of transformix

    Arguments:
        filename (str): file name of the transformix output file
        indices (bool): if True return pixel indices otherwise float coordinates

    Returns:
        points (array): the transformed coordinates
    """

    with open(filename) as f:
        lines = f.readlines()
        f.close()

    npoints = len(lines)

    if npoints == 0:
        return np.zeros((0,3))

    points = np.zeros((npoints, 3))
    k = 0
    for line in lines:
        ls = line.split()
        if indices:
            for i in range(0,3):
                points[k,i] = float(ls[i+22])
        else:
            for i in range(0,3):
                points[k,i] = float(ls[i+30])

        k += 1

    return points


def getTransformFileSizeAndSpacing(transformfile):
    """Parse the image size and spacing from a transformation parameter file

    Arguments:
        transformfile (str): File name of the transformix parameter file.

    Returns:
        (float, float): the image size and spacing
    """

    resi = re.compile("\(Size (?P<size>.*)\)")
    resp = re.compile("\(Spacing (?P<spacing>.*)\)")

    si = None
    sp = None
    with open(transformfile) as parfile:
        for line in parfile:
            #print line;
            m = resi.match(line)
            if m is not None:
                pn = m.group('size')
                si = pn.split()
                #print si

            m = resp.match(line)
            if m is not None:
                pn = m.group('spacing')
                sp = pn.split()
                #print sp

        parfile.close()

    si = [float(x) for x in si]
    sp = [float(x) for x in sp]

    return si, sp


def getResultDataFile(resultdir):
    """Returns the mhd result file in a result directory

    Arguments:
        resultdir (str): Path to elastix result directory.

    Returns:
        str: The mhd file in the result directory.

    """

    files = os.listdir(resultdir)
    files = [x for x in files if re.match('.*.mhd', x)]
    files.sort()

    if not files:
        raise RuntimeError('Cannot find a valid result data file in ' + resultdir)

    return os.path.join(resultdir, files[0])



def setTransformFileSizeAndSpacing(transformfile, size, spacing):
    """Replaces size and scale in the transformation parameter file

    Arguments:
        transformfile (str): transformation parameter file
        size (tuple): the new image size
        spacing (tuplr): the new image spacing
    """

    resi = re.compile("\(Size (?P<size>.*)\)")
    resp = re.compile("\(Spacing (?P<spacing>.*)\)")

    fh, tmpfn = tempfile.mkstemp()

    si = [int(x) for x in size]

    with open(transformfile) as parfile:
        with open(tmpfn, 'w') as newfile:
            for line in parfile:
                #print line

                m = resi.match(line)
                if m is not None:
                    newfile.write("Size {}".format(si))
                else:
                    m = resp.match(line)
                    if m is not None:
                        newfile.write("(Spacing %d %d %d)" % spacing)
                    else:
                        newfile.write(line)

            newfile.close()
            parfile.close()

            os.remove(transformfile)
            shutil.move(tmpfn, transformfile)


def rescaleSizeAndSpacing(size, spacing, scale):
    """Rescales the size and spacing

    Arguments:
        size (tuple): image size
        spacing (tuple): image spacing
        scale (tuple): the scale factor

    Returns:
        (tuple, tuple): new size and spacing
    """

    si = [int(x * scale) for x in size]
    sp = spacing / scale

    return si, sp


def alignData(fixedImage, movingImage, affineParameterFile, bSplineParameterFile = None, resultDirectory = None):
    """Align images using elastix, estimates a transformation :math:`T:` fixed image :math:`\\rightarrow` moving image.

    Arguments:
        fixedImage (str): image source of the fixed image (typically the reference image)
        movingImage (str): image source of the moving image (typically the image to be registered)
        affineParameterFile (str or None): elastix parameter file for the primary affine transformation
        bSplineParameterFile (str or None): elastix parameter file for the secondary non-linear transformation
        resultDirectory (str or None): elastic result directory

    Returns:
        str: path to elastix result directory
    """

    if resultDirectory is None:
        resultDirectory = tempfile.gettempdir()

    if not os.path.exists(resultDirectory):
        os.mkdir(resultDirectory)

    if bSplineParameterFile is None:
        cmd = config.elastix_binary + ' -threads 16 -m ' + movingImage + ' -f ' + fixedImage + ' -p ' + affineParameterFile + ' -out ' + resultDirectory
    elif affineParameterFile is None:
        cmd = config.elastix_binary + ' -threads 16 -m ' + movingImage + ' -f ' + fixedImage + ' -p ' + bSplineParameterFile + ' -out ' + resultDirectory
    else:
        cmd = config.elastix_binary + ' -threads 16 -m ' + movingImage + ' -f ' + fixedImage + ' -p ' + affineParameterFile + ' -p ' + bSplineParameterFile + ' -out ' + resultDirectory
        #$ELASTIX -threads 16 -m $MOVINGIMAGE -f $FIXEDIMAGE -fMask $FIXEDIMAGE_MASK -p  $AFFINEPARFILE -p $BSPLINEPARFILE -out $ELASTIX_OUTPUT_DIR

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('alignData: failed executing: ' + cmd)

    return resultDirectory


def transformImage(source, sink = [], transformParameterFile = None, transformDirectory = None, resultDirectory = None):
    """Transform a raw data set to reference using the elastix alignment results

    If the map determined by elastix is
    :math:`T \\mathrm{fixed} \\rightarrow \\mathrm{moving}`,
    transformix on data works as :math:`T^{-1}(\\mathrm{data})`.

    Arguments:
        source (str or array): image source to be transformed
        sink (str, [] or None): image sink to save transformed image to. if [] return the default name of the data file generated by transformix.
        transformParameterFile (str or None): parameter file for the primary transformation, if None, the file is determined from the transformDirectory.
        transformDirectory (str or None): result directory of elastix alignment, if None the transformParameterFile has to be given.
        resultDirectory (str or None): the directorty for the transformix results

    Returns:
        array or str: array or file name of the transformed data
    """

    if isinstance(source, np.ndarray):
        imgname = os.path.join(tempfile.gettempdir(), 'elastix_input.tif')
        io.writeData(source, imgname)
    elif isinstance(source, str):
        if io.dataFileNameToType(source) == "TIF":
            imgname = source
        else:
            imgname = os.path.join(tempfile.gettempdir(), 'elastix_input.tif')
            io.transformImage(source, imgname)#TODO: not sure if works
    else:
        raise RuntimeError('transformImage: source not a string or array')

    if resultDirectory is None:
        resultdirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
    else:
        resultdirname = resultDirectory

    if not os.path.exists(resultdirname):
        os.makedirs(resultdirname)

    if transformParameterFile is None:
        if transformDirectory is None:
            raise RuntimeError('neither alignment directory and transformation parameter file specified!')
        transformparameterdir = transformDirectory
        transformParameterFile[-1] = getTransformParameterFiles(transformparameterdir)
    else:
        transformparameterdir = os.path.split(transformParameterFile)
        transformparameterdir = transformparameterdir[0]

    #transform
    #make path in parameterfiles absolute
    setPathTransformParameterFiles(transformparameterdir)

    #transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txt
    cmd = config.tranformix_binary + ' -in ' + imgname + ' -out ' + resultdirname + ' -tp ' + transformParameterFile

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('transformImage: failed executing: ' + cmd)

    if not isinstance(source, str):
        os.remove(imgname)

    if not sink:
        return getResultDataFile(resultdirname)
    elif sink is None:
        resultfile = getResultDataFile(resultdirname)
        return io.readData(resultfile)
    elif isinstance(sink, str):
        resultfile = getResultDataFile(resultdirname)
        return io.convertData(resultfile, sink)
    else:
        raise RuntimeError('transformImage: sink not valid!')


def deformationField(sink = [], transformParameterFile = None, transformDirectory = None, resultDirectory = None):
    """Create the deformation field T(x) - x

    The map determined by elastix is
    :math:`T \\mathrm{fixed} \\rightarrow \\mathrm{moving}`

    Arguments:
        sink (str, [] or None): image sink to save the transformation field; if [] return the default name of the data file generated by transformix.
        transformParameterFile (str or None): parameter file for the primary transformation, if None, the file is determined from the transformDirectory.
        transformDirectory (str or None): result directory of elastix alignment, if None the transformParameterFile has to be given.
        resultDirectory (str or None): the directorty for the transformix results

    Returns:
        array or str: array or file name of the transformed data
    """
    if resultDirectory is None:
        resultdirname = os.path.join(tempfile.gettempdir(), 'elastix_output')
    else:
        resultdirname = resultDirectory

    if not os.path.exists(resultdirname):
        os.makedirs(resultdirname)

    if transformParameterFile is None:
        if transformDirectory is None:
            raise RuntimeError('neither alignment directory and transformation parameter file specified!')
        transformparameterdir = transformDirectory
        transformParameterFile = getTransformParameterFiles(transformparameterdir)[-1]
    else:
        transformparameterdir = os.path.split(transformParameterFile)
        transformparameterdir = transformparameterdir[0]

    #transform
    #make path in parameterfiles absolute
    setPathTransformParameterFiles(transformparameterdir)

    #transformix -in inputImage.ext -out outputDirectory -tp TransformParameters.txt
    cmd = config.tranformix_binary + ' -def all -out ' + resultdirname + ' -tp ' + transformParameterFile

    res = os.system(cmd)

    if res != 0:
        raise RuntimeError('deformationField: failed executing: ' + cmd)

    if not sink:
        return getResultDataFile(resultdirname)
    elif sink is None:
        resultfile = getResultDataFile(resultdirname)
        data = io.readData(resultfile)
        if resultDirectory is None:
            shutil.rmtree(resultdirname)
        return data
    elif isinstance(sink, str):
        resultfile = getResultDataFile(resultdirname)
        data = io.convertData(resultfile, sink)
        if resultDirectory is None:
            shutil.rmtree(resultdirname)
        return data
    else:
        raise RuntimeError('deformationField: sink not valid!')


def deformationDistance(deformationField, sink = None, scale = None):
    """Compute the distance field from a deformation vector field

    Arguments:
        deformationField (str or array): source of the deformation field determined by :func:`deformationField`
        sink (str or None): image sink to save the deformation field to
        scale (tuple or None): scale factor for each dimension, if None = (1,1,1)

    Returns:
        array or str: array or file name of the transformed data
    """

    deformationField = io.readData(deformationField)

    df = np.square(deformationField)
    if not scale is None:
        for i in range(3):
            df[:,:,:,i] = df[:,:,:,i] * (scale[i] * scale[i])

    return io.writeData(sink, np.sqrt(np.sum(df, axis = 3)))


def writePoints(filename, points, indices = True):
    """Write points as elastix/transformix point file. with format:
    <index, point>
    <number of points>
    point1 x point1 y [point1 z]
    point2 x point2 y [point2 z]

    Arguments:
        filename (str): file name of the elastix point file.
        points (array or str): source of the points.
        indices (bool): write as pixel indices or physical coordiantes

    Returns:
        str : file name of the elastix point file
    """
    log.debug('writing points to text file for Elastix input')
    points = io.readPoints(points)

    with open(filename, 'w+') as pointfile:
        if indices:
            pointfile.write('index\n')
        else:
            pointfile.write('point\n')
        pointfile.write('{}\n'.format(points.shape[0]))
        np.savetxt(pointfile, points, delimiter = ' ', newline = '\n', fmt = '%.5e')
        pointfile.close()

    return filename


def transformPoints(source, sink=None, transformParameter=None, indices=True,
                    resultDirectory=None, tmpFile=None):
    """Transform coordinates math:`x` via elastix estimated transformation to :math:`T(x)`
    Note the transformation is from the fixed image coorindates to the moving image coordiantes.

    Arguments:
        source (str): source of the points
        sink (str or None): sink for transformed points
        transformParameterFile (str or None): parameter file for the primary transformation, if None, the file is determined from the transformDirectory.
        transformDirectory (str or None): result directory of elastix alignment, if None the transformParameterFile has to be given.
        indices (bool): if True use points as pixel coordinates otherwise spatial coordinates.
        resultDirectory (str or None): elastic result directory
        tmpFile (str or None): file name for the elastix point file.

    Returns:
        array or str: array or file name of transformed points
    """

    if tmpFile == None:
        tmpFile = os.path.join(tempfile.tempdir, 'elastix_input.txt');

    # write text file
    if isinstance(source, str):

        # check if we have elastix signature
        with open(source) as f:
            line = f.readline();
            f.close();

            if line[:5] == 'point' or line[:5] != 'index':
                txtfile = source;
            else:
                points = io.readPoints(source);
                # points = points[:,[1,0,2]];
                txtfile = tmpFile;
                writePoints(txtfile, points);

    elif isinstance(source, np.ndarray):
        txtfile = tmpFile;
        # points = source[:,[1,0,2]];
        writePoints(txtfile, source);

    else:
        raise RuntimeError('transformPoints: source not string or array!');

    if resultDirectory == None:
        outdirname = os.path.join(tempfile.tempdir, 'elastix_output');
    else:
        outdirname = resultDirectory;
    if not os.path.exists(outdirname):
        os.makedirs(outdirname);


    if os.path.isdir(transformParameter):
        transformparameterfile = getTransformParameterFiles(transformParameter)[-1];
        transformparameterdir = transformParameter
    elif os.path.isfile(transformParameter):
        transformparameterdir = os.path.split(transformParameter)[0];
        transformparameterfile = transformParameter;
    else:
        raise RuntimeError('could not find parameters ' + transformParameter)

    # transform
    # make path in parameterfiles absolute
    setPathTransformParameterFiles(transformparameterdir);

    # run transformix
    cmd = config.tranformix_binary + ' -def ' + txtfile + ' -out ' + outdirname + ' -tp ' + transformparameterfile;
    res = os.system(cmd);

    if res != 0:
        raise RuntimeError('failed executing ' + cmd);

    # read data / file
    if sink == []:
        return io.path.join(outdirname, 'outputpoints.txt')

    else:
        # read coordinates
        transpoints = parseElastixOutputPoints(os.path.join(outdirname, 'outputpoints.txt'), indices=indices);

        # correct x,y,z to y,x,z
        # transpoints = transpoints[:,[1,0,2]];

        # cleanup
        for f in os.listdir(outdirname):
            os.remove(os.path.join(outdirname, f));
        os.rmdir(outdirname)

        return io.writePoints(sink, transpoints);

initialize_Elastix()
