# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.

import os
import sys
import numpy
import pip
import pkgutil
import shutil
import tarfile
import urllib

from pathlib import Path
from setuptools import setup
from setuptools import find_packages
from setuptools.command.install import install as _install
from distutils.extension import Extension

from clearmap3._version import __version__

if sys.platform not in ['linux']:
    import tarfile
    raise EnvironmentError(f'platform {sys.platform} not supported.')

USE_CYTHON = 'auto'

if USE_CYTHON:
    try:
        from Cython.Distutils import build_ext
        from Cython.Build import cythonize
    except ImportError:
        if USE_CYTHON == 'auto':
            USE_CYTHON = False
        else:
            raise

cmdclass = {}
ext_modules = []

if USE_CYTHON:
    ext_modules += cythonize([
        Extension("clearmap3.analysis._voxelization",
                  sources=["clearmap3/analysis/_voxelization.pyx"],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("clearmap3.image_filters.filters._background_subtraction",
                  sources=["clearmap3/image_filters/filters/_background_subtraction.pyx"]
                  ),
        Extension("clearmap3.image_filters.filters.helpers.array_manipulations",
                  sources=["clearmap3/image_filters/filters/helpers/array_manipulations.pyx"]
                  ),
        Extension("clearmap3.image_filters.filters.label._label",
                  sources=["clearmap3/image_filters/filters/label/_label.pyx"],
                  language="c++",
                  include_dirs=[numpy.get_include(), "include"],
                  extra_link_args=[os.path.join("clearmap3/.lib", f) for f in os.listdir("clearmap3/.lib")],
                  ),
        Extension("clearmap3.image_filters.filters.label._threshold",
                  sources=["clearmap3/image_filters/filters/label/_threshold.pyx"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("clearmap3.image_filters.filters.label.filter",
                  sources=["clearmap3/image_filters/filters/label/filter.pyx"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("clearmap3.image_filters.filters.label.overlap",
                  sources=["clearmap3/image_filters/filters/label/overlap.pyx"],
                  include_dirs=[numpy.get_include()],
                  ),
        Extension("clearmap3.image_filters.filters.label.watershed._watershed",
                  sources=["clearmap3/image_filters/filters/label/watershed/_watershed.pyx"],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("clearmap3.image_filters.filters.label.util.nonzero_coords",
                  sources=["clearmap3/image_filters/filters/label/util/nonzero_coords.pyx"],
                  language='c++',
                  include_dirs=[numpy.get_include()]
                  )
    ])
    cmdclass['build_ext'] = build_ext
else:
    ext_modules += [
        Extension("clearmap3.analysis._voxelization",
                  sources=["clearmap3/analysis/_voxelization.c"],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("clearmap3.image_filters.filters._background_subtraction",
                  sources=["clearmap3/image_filters/filters/_background_subtraction.c"]
                  ),
        Extension("clearmap3.image_filters.filters.helpers.array_manipulations",
                  sources=["clearmap3/image_filters/filters/helpers/array_manipulations.c"]
                  ),
        Extension("clearmap3.image_filters.filters.label._label",
                  sources=["clearmap3/image_filters/filters/label/_label.cpp"],
                  language="c++",
                  include_dirs=[numpy.get_include(), "include"],
                  extra_link_args=[os.path.join("clearmap3/.lib", f) for f in os.listdir("clearmap3/.lib")],
                  ),
        Extension("clearmap3.image_filters.filters.label._threshold",
                  sources=["clearmap3/image_filters/filters/label/_threshold.cpp"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("clearmap3.image_filters.filters.label.filter",
                  sources=["clearmap3/image_filters/filters/label/filter.cpp"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("clearmap3.image_filters.filters.label.overlap",
                  sources=["clearmap3/image_filters/filters/label/overlap.c"],
                  include_dirs=[numpy.get_include()],
                  ),
        Extension("clearmap3.image_filters.filters.label.watershed._watershed",
                  sources=["clearmap3/image_filters/filters/label/watershed/_watershed.c"],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("clearmap3.image_filters.filters.label.util.nonzero_coords",
                  sources=["clearmap3/image_filters/filters/label/util/nonzero_coords.cpp"],
                  language='c++',
                  include_dirs=[numpy.get_include()]
                  )
    ]


class install(_install):
    def run(self):
        _install.run(self)

        # get external programs required by package to install directory. they are in folders
        dest = Path(pkgutil.get_loader('clearmap3').path).parent / '.external'

        url = 'https://github.com/SuperElastix/elastix/releases/download/4.9.0/elastix-4.9.0-linux.tar.bz2'
        tmp = Path(url).name
        sink = dest / 'elastix-4.9.0-linux'
        with urllib.request.urlopen(url) as response, open(tmp, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            tar = tarfile.open(tmp, "r:bz2")
            tar.extractall(sink)
            tar.close()

        url = 'http://files.ilastik.org/ilastik-1.3.2post1-Linux.tar.bz2'
        tmp = Path(url).name
        sink = dest / 'ilastik-1.3.2post1-Linux'
        with urllib.request.urlopen(url) as response, open(tmp, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            tar = tarfile.open(tmp, "r:bz2")
            tar.extractall(sink)
            tar.close()

        # install antspy
        if sys.platform == 'linux':
            pip.main(['install',
                "https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl"])
        if sys.platform == 'darwin':
            pip.main(['install',
                "https://github.com/ANTsX/ANTsPy/releases/download/Weekly/antspy-0.1.4-cp36-cp36m-macosx_10_7_x86_64.whl"])


cmdclass['install'] = install

setup(
    name=               'clearmap3',
    version=            __version__,
    description=        'Tools for tera-voxel image analysis.',
    author=             'Ricardo Azevedo, Jack Zeitoun',
    author_email=       'ricardo-re-azevedo@gmail.com, jack.zeitoun@outlook.com',
    maintainer=         'Ricardo Azevedo',
    maintainer_email=   'ricardo-re-azevedo@gmail.com',
    url=                'https://github.com/ricardo-re-azevedo/clearmap3',
    license=            'BY-NC-SA 4.0',
    cmdclass=           cmdclass,
    packages=           find_packages(),
    install_requires=[
        'numpy',
        'pyyaml',
        'scipy',
        'opencv-python',
        'tifffile',
        'scikit-image',
        'pandas',
        'h5py',
        'vtk',
        'anytree',
        'webcolors',  # required for antspy
        'plotly'  # required for antspy
    ],
    include_package_data=True,
    ext_modules=ext_modules,
    keywords='tera voxel teravoxel image analysis biology'
)
