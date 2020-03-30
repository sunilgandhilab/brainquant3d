# Set this to True to enable building extensions using Cython.
# Set it to False to build extensions from the C file (that was previously created using Cython).
# Set it to 'auto' to build with Cython if available, otherwise
# from the C file.

import os
import sys
import numpy
import shutil
import tarfile
import urllib.request as request
import ssl

from glob import glob
from pathlib import Path
from setuptools import find_packages
from setuptools.command.install import install as _install
from distutils.core import setup
from distutils.extension import Extension

from bq3d._version import __version__

if sys.platform not in ['linux', 'darwin']:
    raise EnvironmentError(f'Platform {sys.platform} not supported.')

if sys.platform == 'linux':
    opencv_libs = '.lib-linux'
    elastix_URL = 'elastix-5.0.0-linux.tar.bz2'
    ilastik_URL = 'ilastik-1.3.3-Linux-noGurobi.tar.bz2'
elif sys.platform == 'darwin':
    opencv_libs = '.lib-osx'
    elastix_URL = 'elastix-5.0.0-mac.tar.gz'
    ilastik_URL = 'ilastik-1.3.3post2-OSX-noGurobi.tar.bz2'

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
        Extension("bq3d.analysis._voxelization",
                  sources=["bq3d/analysis/_voxelization.pyx"],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("bq3d.image_filters.filters._background_subtraction",
                  sources=["bq3d/image_filters/filters/_background_subtraction.pyx"]
                  ),
        Extension("bq3d.image_filters.filters.helpers.array_manipulations",
                  sources=["bq3d/image_filters/filters/helpers/array_manipulations.pyx"]
                  ),
        Extension("bq3d.image_filters.filters.label._connect",
                  sources=["bq3d/image_filters/filters/label/_connect.pyx"],
                  language="c++",
                  include_dirs=[numpy.get_include(), "include"],
                  extra_link_args=[os.path.join(f'bq3d/{opencv_libs}', f) for f in os.listdir(
                      f'bq3d/{opencv_libs}')],
                  runtime_library_dirs=[f'$ORIGIN/../../../{opencv_libs}']
                  ),
        Extension("bq3d.image_filters.filters.label._threshold",
                  sources=["bq3d/image_filters/filters/label/_threshold.pyx"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("bq3d.image_filters.filters.label._filter",
                  sources=["bq3d/image_filters/filters/label/_filter.pyx"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("bq3d.image_filters.filters.label._overlap",
                  sources=["bq3d/image_filters/filters/label/_overlap.pyx"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("bq3d.image_filters.filters.label.watershed._watershed",
                  sources=["bq3d/image_filters/filters/label/watershed/_watershed.pyx"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("bq3d.image_filters.filters.helpers._nonzero_coords",
                  sources=["bq3d/image_filters/filters/helpers/_nonzero_coords.pyx"],
                  include_dirs=[numpy.get_include()],
                  language='c++'
                  ),
        Extension("bq3d.image_filters.filters._standardize",
                  sources=["bq3d/image_filters/filters/_standardize.pyx"],
                  include_dirs=[numpy.get_include()],
                  language='c++'
                  )
    ])
    cmdclass['build_ext'] = build_ext
else:
    ext_modules += [
        Extension("bq3d.analysis._voxelization",
                  sources=["bq3d/analysis/_voxelization.c"],
                  include_dirs=[numpy.get_include()]
                  ),
        Extension("bq3d.image_filters.filters._background_subtraction",
                  sources=["bq3d/image_filters/filters/_background_subtraction.c"]
                  ),
        Extension("bq3d.image_filters.filters.helpers.array_manipulations",
                  sources=["bq3d/image_filters/filters/helpers/array_manipulations.c"]
                  ),
        Extension("bq3d.image_filters.filters.label._connect",
                  sources=["bq3d/image_filters/filters/label/_connect.cpp"],
                  language="c++",
                  include_dirs=[numpy.get_include(), "include"],
                  extra_link_args=[os.path.join(f'bq3d/{opencv_libs}', f) for f in os.listdir(
                      f'bq3d/{opencv_libs}')],
                  runtime_library_dirs=[f'$ORIGIN/../../../{opencv_libs}']
                  ),
        Extension("bq3d.image_filters.filters.label._threshold",
                  sources=["bq3d/image_filters/filters/label/_threshold.cpp"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("bq3d.image_filters.filters.label._filter",
                  sources=["bq3d/image_filters/filters/label/_filter.cpp"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("bq3d.image_filters.filters.label._overlap",
                  sources=["bq3d/image_filters/filters/label/_overlap.cpp"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("bq3d.image_filters.filters.label.watershed._watershed",
                  sources=["bq3d/image_filters/filters/label/watershed/_watershed.cpp"],
                  include_dirs=[numpy.get_include()],
                  language="c++"
                  ),
        Extension("bq3d.image_filters.filters.helpers._nonzero_coords",
                  sources=["bq3d/image_filters/filters/helpers/_nonzero_coords.cpp"],
                  include_dirs=[numpy.get_include()],
                  language='c++'
                  ),
        Extension("bq3d.image_filters.filters._standardize",
                  sources=["bq3d/image_filters/filters/_standardize.cpp"],
                  include_dirs=[numpy.get_include()],
                  language='c++'
                  )
    ]


class install(_install):
    def run(self):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        #  download external programs required by package to install directory.
        cwd = os.getcwd()
        matches = glob(os.path.join(cwd, 'build/lib.*'))
        build_dir = cwd if len(matches) == 0 else matches[0]
        dest = Path(build_dir) / 'bq3d/.external'

        print('Installing elastix')
        url = 'https://glams.bio.uci.edu/' + elastix_URL
        tmp = Path(url).name
        sink = dest / 'elastix-5.0.0'
        with request.urlopen(url, context=ctx) as response, open(tmp, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            try:
                tar = tarfile.open(tmp, "r:bz2") # Linux
            except:
                tar = tarfile.open(tmp, "r:gz") # MacOS
            tar.extractall(sink)
            tar.close()

        print('Installing ilastik')
        url = 'https://glams.bio.uci.edu/' + ilastik_URL
        tmp = Path(url).name

        sink = dest / 'ilastik-1.3.3'
        with request.urlopen(url, context=ctx) as response, open(tmp, 'wb') as out_file:
            shutil.copyfileobj(response, out_file)
            tar = tarfile.open(tmp, "r:bz2")
            tar.extractall(sink)
            tar.close()

        # Install ANTsPy
        if sys.platform == 'linux':
            try:
                print('Installing ants')
                subprocess.check_call([sys.executable, 'pip', '-m', 'install', '--user', "https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl"])
            except:
                try:
                    print('Attempting alternative ants install')
                    subprocess.check_call([sys.executable, 'pip', '-m', 'install', '--user', "https://github.com/ANTsX/ANTsPy/releases/download/v0.2.0/antspyx-0.2.0-cp37-cp37m-linux_x86_64.whl"])
                except:
                    print('Unable to install ants')
        if sys.platform == 'darwin':
            try:
                print('Installing ants')
                subprocess.check_call([sys.executable, 'pip', '-m', 'install', '--user', "https://github.com/ANTsX/ANTsPy/releases/download/Weekly/antspy-0.1.4-cp36-cp36m-macosx_10_7_x86_64.whl"])
            except:
                try:
                    print('Attempting alternative ants install')
                    subprocess.check_call([sys.executable, 'pip', '-m', 'install', '--user', "https://github.com/ANTsX/ANTsPy/releases/download/v0.1.8/antspyx-0.1.8-cp37-cp37m-macosx_10_14_x86_64.whl"])
                except:
                    print('Unable to install ants')

        _install.run(self)

cmdclass['install'] = install

setup(
    name=               'brainquant3d',
    version=            __version__,
    description=        'Tools for tera-voxel image analysis.',
    author=             'Ricardo Azevedo, Jack Zeitoun',
    author_email=       'ricardo-re-azevedo@gmail.com, jack.zeitoun@outlook.com',
    maintainer=         'Ricardo Azevedo',
    maintainer_email=   'ricardo-re-azevedo@gmail.com',
    url=                'https://github.com/ricardo-re-azevedo/brainquant3d',
    license=            'BY-NC-SA 4.0',
    cmdclass=           cmdclass,
    packages=           find_packages(),
    install_requires=[
        'numpy',
        'pyyaml',
        'scipy',
        'opencv-python',
        'tifffile==2019.7.26', # Breaks in newer version
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
