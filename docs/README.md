# Quickstart Guide

## Requirements

- Python 3.6+

## Installation

Download and install the main package

```
$ python3 -m pip install --user git+https://github.com/sunilgandhilab/brainquant3d-dev.git
```

Download **Warping.zip**, which contains the Allen Brain Atlas datasets:

Option 1 - Download from Google Drive:

```
https://drive.google.com/drive/u/2/folders/1JjUxW3k2foIB2LBOa5LrtnCTofHToVho
```

Option 2 - Download using the following command:
```
$ wget https://glams.bio.uci.edu/Warping.zip
```

Unzip and move the downloaded **Warping** directory to a location you will remember (e.g. SSD drive).

```
$ unzip Warping.zip
$ mv Warping /mnt/ssd/
```

Now you need to set up the configuration file. This file will tell brainquant3d where to place temp data and where to find the atlas data. The configuration template file, **default.conf**, can be found in the package install location. On Linux, this should be:

```
/home/<user>/.local/lib/python3.x/site-packages/bq3d
```

Copy this file and change the name

```
$ cd /home/<user>/.local/lib/python3.x/site-packages/bq3d
$ cp default.conf brainquant3d.conf
```

Open the new file, **brainquant3d.conf**, and edit the following paths:

**Temp_path**: This is the location where temp data will be stored. It should be on a high-performance SSD. Preferably an NVMe. If the path does not exist, it will be created.

**Rigid_default**, **Affine_default**, **BSpline_default**, **Labeled_default**, **Annotations_default**: These are the paths to the Allen Brain Atlas datasets downloaded earlier.

**Processing_cores**, **Thread_ram_max_Gb**: These parameters specify how many processing cores to use and how much RAM to user per core. For example, if your system has 10 cores and 256 GB of RAM, you would set "Processing_cores" to 10 and "Thread_ram_max_Gb" to a value such as 22. This would use 10 x 22 = 220 GB of total RAM. It's a good idea to leave a little available RAM so as to not overload the system.

An example configuration is provided below:

```yaml
user:
    default:
        # Paths
        Ilastik_path:        !pkg_path '.external/ilastik-1.3.3-Linux/ilastik-1.3.3-Linux-noGurobi'
        Elastix_path:        !pkg_path '.external/elastix-5.0.0-linux'
        Temp_path:           ‘/mnt/ssd/brainquant3d-tmp’

        Rigid_default:       '/mnt/ssd/Warping/ParRigid.txt'
        Affine_default:      '/mnt/ssd/Warping/ParAffine.txt'
        BSpline_default:     '/mnt/ssd/Warping/ParBSpline.txt'
        Labeled_default:     '/mnt/ssd/Warping/ARA2/annotation_25_right.tif'
        Annotations_default: '/mnt/ssd/Wrping/ARA2_annotation_info_collapse.csv'
        Console_level:       'verbose'
        Processing_cores:    10
        Thread_ram_max_Gb:   22
```

Now open a python shell and try to import brainquant3d:

```
Python 3.7.4 (default, Jul  9 2019, 18:13:23)
>>> import bq3d
```

If the import succeeds with no warnings, you are ready to go. If you see any warning messages, your configuration file paths are incorrect and will need to be fixed before proceeding.

## Tutorial

BrainQuant3D is a toolkit for image processing. It provides numerous resources designed to aid users in processing large-scale microscopy data. For this tutorial, we will provide an example of how to use BrainQuant3D to set up a pipeline that will segment target cells and generate a plot of cell densities by brain region. The input data was acquired on a Zeiss Z.1 microscope and has been stitched and converted to TIFF format. The original size was 768 x 10500 x 5320 (Z x Y x X), but has been downsampled to a manageable size: 692 x 1400 x 709.

<p align="center">
  <img src="https://github.com/sunilgandhilab/brainquant3d/blob/master/common/sample.jpg"/>
</p>

In BrainQuant3D, a pipeline is created by editing two files. The first file is the parameter file, **parameter.py**. The user will use this file to specify which filters to use, which order to run them in, and the various parameters for each filter. The second file is the process file, **process.py**. The process file is used to specify which analysis routines to run (e.g. cell detection, brain-to-atlas warping).

First, create the folder that will serve as the working directory and a subfolder to hold the raw data:
```
$ cd /mnt/ssd
$ mkdir bq3d-tutorial
$ mkdir bq3d-tutorial/data
```

Next, download **tutorial.zip**:

Option 1 - Download from Google Drive:
```
https://drive.google.com/drive/u/2/folders/1JjUxW3k2foIB2LBOa5LrtnCTofHToVho
```

Option 2 - Download using the following command:
```
$ wget https://glams.bio.uci.edu/tutorial.zip
```

Unzip **tutorial.zip**:
```
$ unzip tutorial.zip
```

Inside, you should see two directories: **C01** and **C02**. These contain channels 1 and 2 of the data.
Move **C01** and **C02** into the **data** folder of the working directory:
```
$ mv tutorial/* /mnt/ssd/bq3d-tutorial/data/
```

Download the **parameter.py** and **process.py** template files and place them in the newly created working directory (**bq3d-tutorial**). They can be found in the **common** directory in the root of this repository.

Now we will edit the **parameter.py** file.

The first field to edit is the **BaseDirectory**. This is the path to where the analysis results will be stored:

```python
BaseDirectory = "/mnt/ssd/bq3d-tutorial/analysis"
```

The next field is the **DataDirectory**. This is the path to where the raw data is stored:

```python
DataDirectory = "/mnt/ssd/bq3d-tutorial/data"
```

Now we will edit the **SignalFile** field. This is the path to the data containing the signal channel. Typically, data will be split into a single file for each plane with a naming scheme similar to *lightsheet_data_Z0001_C01.tif*, *lightsheet_data_Z0002_C01.tif*, and so on. In order for BrainQuant3D to know which part of the filename is variable, we must specify full the path using a regular expression.

```python
SignalFile = os.path.join(DataDirectory, "C01/lightsheet_data_Z\d{3}_C01.tif")
```

The **\d{3}** means the filename will contain a number with 3 digits. If there are more or less digits in the filename, simply change the value in the curly brackets to match.

The next field is the **AutoFluoFile**, which specifies the path to the autofluorescence channel. This channel is used for image registration to the Allen Brain Atlas. It will use the same format as the “SignalFile” field.

```python
AutoFluoFile = os.path.join(DataDirectory, "C02/lightsheet_data_Z\d{3}_C02.tif")
```

Next we will specify the voxel dimensions. The value for this field will be a tuple containing the Z sampling distance between slices followed by the Y and then X pixel dimensions. Units are in microns. For the purpose of this tutorial, the data was downsampled to a resolution of 9um in each dimension.

```python
OriginalResolution = (9, 9, 9)
```

Now, we may need to flip the data across one or more axes so that it matches the orientation of the atlas. To this, we will provide another tuple that will contain 3 values. These 3 values represent each dimensions of the data and are in the order (Z, Y, X). If the image does not need to be adjusted, the value will be `(1, 2, 3)`. This indicates that the dimensions are in the correct order and no inverting is necessary. If we need to invert 1 or more axes, simply change the value to a negative. For example, if the image needed to be inverted across the Y axis, the value would be `(1, -2, 3)`. If we needed to invert both the Y and X axes, the value would be `(1, -2, -3)`. We can also transpose axes by changing the order of the tuple. By inputting `(1, 3, 2)`, the Y and X axes would be transposed. For this tutorial, we will only be inverting the Z axis and X axis.

```python
FinalOrientation = (-1, 2, -3)
```

The next field is the Atlas resolution. For this tutorial, the Atlas is at a resolution of 20x20x20 um.

```python
AtlasResolution = (25, 25, 25)
```

The following field specifies the resolution to downsample the data for misalignment correction between the signal and autofluorescence channels. 12 microns is a safe value for all axes.

```python
CorrectionResolution = (12, 12, 12)
```

The next three fields will tell BrainQuant3D which atlas datasets to use. These fields should point to the location that we set up for the Atlases during the installation.

```python
PathReg = "/mnt/ssd/Warping/ARA2"
AtlasFile = os.path.join(PathReg, "average_template_25_right.tif")
AnnotationFile = os.path.join(PathReg, "annotation_25_right.tif")
```

Next, we specify where to save the absolute cell coordinates after cell detection is complete.

```python
sink = os.path.join(BaseDirectory, 'cells.json')
```

The following line will specify where to save the cell coordinates that have been transformed into Atlas space.

```python
transformedCellsFile = os.path.join(BaseDirectory, "cells_transformed.json")
```

The next field is where the actual pipeline is built. This field is a list of Python dictionaries where each dictionary represents a filter and the corresponding parameters. When you run the pipeline, BrainQuant3D will pass the data through each of these filters in the order they are specified. This tutorial will employ a basic pipeline that first segments all cells using a machine-learning classifier that has been trained to work with this dataset (see Ilastik) and then labels all cells by assigning a unique integer value to each cell.

```python
flow = [
    {
        'filter'             : 'RollingBackgroundSubtract',
        'size'               : 5,
        'save'               : os.path.join(BaseDirectory, 'bkgrdsub/Z\d{3}.tif'),
    },
    {
        'filter'             : 'Label',
        'mode'               : 1,
        'min_size'           : 3,
        'max_size'           : 50,
        'min_size2'          : 3,
        'max_size2'          : 50,
        'high_threshold'     : 450,
        'low_threshold'      : 450,
        "save"               : os.path.join(BaseDirectory, 'labels/Z\d{3}.tif'),
    }
]
```

The rest of the file contains parameters that are specific to the data downsampling and registration operations. In general, these should not be modified.

The parameter file is now ready. Now we need to edit the process file. Edit the following line so that it points to the parameter file.

```python
exec(open("/mnt/ssd/bq3d-tutorial/parameter.py").read())
```

For this tutorial, we will not be making any other changes to the process file. In general, the process file will not need to be modified unless you want to exclude certain routines (e.g. warping/registration) from the analysis. In that case, simply comment out the block of code you want to exclude. See below for a brief description of each block in the process file.

1.	Cells detection is performed first. This is the block of code that will execute the filter pipeline we created and assigned to the “flow” field. The output of this block will be the absolute coordinates for each detected cell.
2.	After cell detection, the signal and autofluorescence channels will be downsampled to a more manageable size. The channels will then be registered to each other to correct for any misalignment during acquisition. Finally, the autofluorescence channel will be registered to the Atlas.
3.	Using the transformation vectors generated from the previous registration step, the cell coordinates will be transformed into Atlas space.
4.	A heatmap containing all transformed cell coordinates will be written onto the Atlas so that cells can be localized to brain regions.
5.	A CSV file will be generated that contains cell density properties for each brain region.

At this point, you are ready to run BrainQuant3D. Make sure the **process.py** and **parameter.py** files are in the **BaseDirectory**. Now you simply run the **process.py** script.

```
$ python3 /mnt/ssd/bq3d-tutorial/process.py
```

If everything was done correctly, you should begin seeing a log print to the screen. The runtime for this tutorial should be approximately 1-2 hours, though this will depend on your computing infrastrucure. For full-sized datasets, the runtime is quite variable. For a typical dataset with 2-300GB per channel, the runtime is somewhere between 12 - 24 hours.

When complete, the **BaseDirectory** should contain the following new files and directories:

```
ants_auto_to_atlas/
ants_signal_to_auto/
autofluo_resampled_12.tif
autofluo_resampled_25.tif
cells_atlas.csv
cells_atlas.tif
cells_corr.tif
cells_ds.tif
cells.json
cells_transformed.json
labels/
probs/
signal_resampled_12.tif
signal_resampled_25.tif
```

This completes the tutorial. If you have any questions or need assistance, reach out to **jzeitoun@uci.edu**.
