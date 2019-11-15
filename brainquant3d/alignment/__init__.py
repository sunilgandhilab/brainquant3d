# -*- coding: utf-8 -*-
"""
This sub-package provides alignment tools used to register datasets to a reference atlas as well
as resample images.

Main routine for resampling is:
:func:`~clearmap3.alignment.resampling.resampleData`

For image registration there are 2 main routines, one using Elastix (requires install of Elastix
binaries. Path added to elastix folder must be included in clearmap3 config.):

"S. Klein, M. Staring, K. Murphy, M.A. Viergever and J.P.W. Pluim,
elastix: a toolbox for intensity based medical image registration,
IEEE Transactions on Medical Imaging, vol. 29, no. 1, pp. 196 - 205, January 2010."

and another using ANTs:
"https://github.com/ANTsX/ANTs"

Which module you chose to use will depend on the dataset as one may work better than the other
under different circumstances.

Main routines for elastix registration are: 
:func:`clearmap3.alignment.elastix.alignData`,
:func:`clearmap3.alignment.elastix.transformImage`
:func:`clearmap3.alignment.elastix.transformPoints`.

Main routines for ants registration are:
:func:`clearmap3.alignment.ants.alignData`,
:func:`clearmap3.alignment.ants.transformImage`
:func:`clearmap3.alignment.ants.transformPoints`.
"""

from clearmap3._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"