# -*- coding: utf-8 -*-
"""
Interface to write points to VTK files

Notes:
    - points are assumed to be in [x,y,z] coordinates as standard in ClearMap
    - reading of points not supported at the moment!

"""

import logging

from bq3d import io
import numpy

log = logging.getLogger(__name__)

def writePoints(filename, points, labelImage = None):
    """Write point data to vtk file
    
    Arguments:
        filename (str): file name
        points (array): point data
        labelImage (str, array or None): optional label image to determine point label
    
    Returns:
        str: file name
    """

    x = points[:,0]
    y = points[:,1]
    z = points[:,2]
    nPoint = x.size
    
    pointLabels = numpy.ones(nPoint)
    if not labelImage is None:
        if isinstance(labelImage, str):
            labelImage = io.readData(labelImage)

        dsize = labelImage.shape
        for i in range(nPoint):
            if 0 <= x[i] < dsize[0] and 0 <= y[i] < dsize[1] and 0 <= z[i] < dsize[2]:
                 pointLabels[i] = labelImage[x[i], y[i], z[i]]

    #write VTK file
    vtkFile = file(filename, 'w')
    vtkFile.write('# vtk DataFile Version 2.0\n')
    vtkFile.write('Unstructured Grid Example\n')
    vtkFile.write('ASCII\n')
    vtkFile.write('DATASET UNSTRUCTURED_GRID\n')
    vtkFile.write("POINTS " + str(nPoint) + " float\n")
    for iPoint in range(nPoint):
        vtkFile.write(str(x[iPoint]).format('%05.20f') + " " +  str(y[iPoint]).format('%05.20f') + " " + str(z[iPoint]).format('%05.20f') + "\n")

    vtkFile.write("CELLS " + str(nPoint) + " " + str(nPoint * 2) + "\n")

    for iPoint in range(nPoint):
        vtkFile.write("1 " + str(iPoint) + "\n")
    vtkFile.write("CELL_TYPES " + str(nPoint) + "\n")
    for iPoint in range(0, nPoint):
        vtkFile.write("1 \n")
    vtkFile.write("POINT_DATA " + str(nPoint) + "\n")
    vtkFile.write('SCALARS scalars float 1\n')
    vtkFile.write("LOOKUP_TABLE default\n")
    for iLabel in pointLabels:
        vtkFile.write(str(int(iLabel)) + " ")
    vtkFile.write("\n")
    vtkFile.close()

    return filename

