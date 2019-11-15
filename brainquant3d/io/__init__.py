# -*- coding: utf-8 -*-
"""This sub-package provides routines to read and write data

Two types of data files are discriminated:
    * `Image data`_
    * `Point data`_

The image data are stacks from microscopes obtained by volume imaging, or the results of analysis representing
the visualization of the detected objects for instance.

The point data are lists of cell coordinates or measured intensities for instance. 

Image data
---------- 

Images are represented internally as numpy arrays. clearmap3assumes images
in arrays are arranged as [x,y], [x,y,z] or [x,y,z,c] where x,y,z correspond to 
the x,y,z coordinates as when viewed in an image viewer such as ImageJ. 
The c coordinate is a possible color channel.

.. note:: Many image libraries read images as [y,x,z] or [y,x] arrays!

The clearmap3toolbox supports a range of (volumetric) image formats:

=============== ========================================================== ============================
Format          Descrition                                                 Module
=============== ========================================================== ============================
TIF             tif images and stacks                                      :mod:`~clearmap3.IO.TIF`
RAW / MHD       raw image files with optional mhd header file              :mod:`~clearmap3.IO.RAW`
NRRD            nearly raw raster data files                               :mod:`~clearmap3.IO.NRRD`
IMS             imaris image file                                          :mod:`~clearmap3.IO.Imaris`
reg exp         folder, file list or file pattern of a stack of 2d images  :mod:`~clearmap3.IO.FileList`
=============== ========================================================== ============================

.. note:: clearmap3can read the image data from a Bitplane’s Imaris, but can’t export image data as an Imaris file.

The image format is inferred automatically from the file name extension.

For example to read image data use :func:`~clearmap3.IO.IO.readData`:

    >>> import os
    >>> from clearmap3 import io
    >>> import clearmap3.    >>> filename = os.path.join(clearmap3.config.ClearMap_path,'Test/Data/Tif/test.tif')
    >>> data = io.readData(filename)
    >>> print data.shape
    (20, 50, 10)

To write image data use :func:`~clearmap3.IO.IO.writeData`:

    >>> import os, numpy
    >>> from clearmap3 import io
    >>> import clearmap3.    >>> filename = os.path.join(clearmap3.config.ClearMap_path,'Test/Data/Tif/test.tif')
    >>> data = numpy.random.rand(20,50,10)
    >>> data[5:15, 20:45, 2:9] = 0
    >>> data = 20 * data
    >>> data = data.astype('int32')
    >>> res = io.writeData(filename, data)
    >>> print io.dataSize(res)
    (20, 50, 10)
    
Generally, the IO module is designed to work with image sources which can be
either files or already loaded numpy arrays. This is important to enable flexible
parallel processing, without rewriting the data analysis routines. 

For example:

    >>> import numpy
    >>> from clearmap3 import io
    >>> data = numpy.random.rand(20,50,10)
    >>> res = io.writeData(None, data)
    >>> print res.shape
    (20, 50, 10)

Range parameter can be passed in order to only load sub sets of image data,
useful when the images are very large. For example to load a sub-image:

    >>> import os, numpy
    >>> from clearmap3 import io
    >>> import clearmap3.    >>> filename = os.path.join(clearmap3.config.ClearMap_path,'Test/Data/Tif/test.tif')
    >>> res = io.readData(filename, data, x = (0,3), y = (4,6), z = (1,4))
    >>> print res.shape
    (3, 2, 3)


Point data
----------

clearmap3also supports several data formats for storing arrays of points, such
as cell center coordinates or intensities.

Points are assumed to be an array of coordinates where the first array index
is the point number and the second the spatial dimension, i.e. [i,d]
The spatial dimension can be extended with additional dimensions 
for intensity ,easires or other properties.

Points can also be given as tuples (coordinate arrray, property array).


clearmap3supports the following files formats for point like data:

========= ========================================================== =======================
Format    Description                                                Module
========= ========================================================== =======================
CSV       comma separated values in text file                        :mod:`~clearmap3.IO.CSV`
NPY       numpy binary file                                          :mod:`~clearmap3.IO.NPY`
VTK       vtk point data file                                        :mod:`~clearmap3.IO.VTK`
========= ========================================================== =======================

.. note:: clearmap3can write points data to a pre-existing Bitplane’s Imaris file, but can’t import the points from them.


The point file format is inferred automatically from the file name extension.

For example to read point data use :func:`~clearmap3.IO.IO.readPoints`:

    >>> import os
    >>> from clearmap3 import io
    >>> import clearmap3.    >>> filename = os.path.join(clearmap3.config.ClearMap_path, stack_processing')
    >>> points = io.readPoints(filename)
    >>> print points.shape
    (5, 3)

and to write it use :func:`~clearmap3.IO.IO.writePoints`:

    >>> import os, numpy
    >>> from clearmap3 import io
    >>> import clearmap3.    >>> filename = os.path.join(clearmap3.config.ClearMap_path, stack_stack_processing >>> points = numpy.random.rand(5,3)
    >>> io.writePoints(filename, points)


Summary
-------
    - All routines accessing data or data properties accept file name strings or numpy arrays or None
    - Numerical arrays represent data and point coordinates as [x,y,z] or [x,y] 
""" 
#:copyright: Copyright 2015 by Christoph Kirst, The Rockefeller University, New York City
#:license: GNU, see LICENSE.txt for details.

from clearmap3.io.io import *


