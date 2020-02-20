# -*- coding: utf-8 -*-
"""
script to set up the parameters for the image processing pipeline
"""
import os

######################### Data parameters

#Directory to save all the results, usually containing the data for one sample
BaseDirectory = '/mnt/e/jack-temp/brainquant3d-tutorial/analysis'
DataDirectory = '/mnt/e/jack-temp/brainquant3d-tutorial/data'

#Data File and Reference channel File, usually as a sequence of files from the microscope
SignalFile = os.path.join(DataDirectory, 'C01_9/lightsheet_data_Z\d{3,4}_C01.tif')
AutofluoFile = os.path.join(DataDirectory, 'C02_9/lightsheet_data_Z\d{3,4}_C02.tif')

#Resolution of the Raw Data (in um / pixel)
OriginalResolution = (9, 9, 9);

#Orientation: 1,2,3 means the same orientation as the reference and atlas files.
#Flip axis with - sign (eg. (-1,2,3) flips z). 3D Rotate by swapping numbers. (eg. (2,1,3) swaps x and y)
FinalOrientation = (-1, 2, -3);

#Resolution of the Atlas (in um/ pixel)
AtlasResolution = (25, 25, 25);
#Resolution to downsample to for correction between channels (in um/ pixel)
CorrectionResolution =  (12, 12, 12);

#Path to registration parameters and atlases
PathReg        = '/mnt/e/warping/ARA2';
AtlasFile      = os.path.join(PathReg, 'average_template_25_right.tif');
AnnotationFile = os.path.join(PathReg, 'annotation_25_right.tif');

#Files to save cell info too.
# Coordinates and properties will be saved to this file
OutputProperties = ['centroid', 'sum', 'area']
sink = os.path.join(BaseDirectory, 'cells.json')

# File to save coordinates too after transforming to atlas
transformedCellsFile = os.path.join(BaseDirectory, 'cells_transformed.json')

######################### Cell Detection Parameters
# flow is a series of dictionaries containing which filters to run

flow = (
    {
        'filter'             : 'RollingBackgroundSubtract',
        'size'          	 : 5,
        "save"               : os.path.join(BaseDirectory, 'bkgrdsub/Z\d{4}.tif'),
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
        "save"               : os.path.join(BaseDirectory, 'labels/Z\d{4}.tif'),
    }
)

CellDetectionParams = {
    # Specify the cropped range for the cell detection. If None will not crop.
    # This doesn't affect the resampling and registration operations
    'x' : None,
    'y' : None,
    'z' : None,

    # chunking args.
    'processes'    : 1,           # number of physical cores to use
    'min_sizes'    : (30,30,30),  # min substack size along each axis in pixels
    'overlap'      : 10,          # amount of overlap in pixels between substacks
    'aspect_ratio' : (1,10,10),   # ratio bewtween axes to maintain in substacks

    # general args. These do not normally need to be modified.
    'source'            : SignalFile,
    'flow'              : flow,
    'sink'              : sink,
    'output_properties' : OutputProperties,
    'log_level'         : 'verbose', # how much info to log in console
}

######################### Registration and Resampling Parameters
# These generally do not need to be modified

CorrectionResamplingParamSignal = {
    'interpolation'     : 'area',
    'source'            : SignalFile,
    'sink'              : os.path.join(BaseDirectory, 'signal_resampled_12.tif'),
    'resolutionSource'  : OriginalResolution,
    'resolutionSink'    : CorrectionResolution,
    'orientation'       : FinalOrientation,
}

CorrectionResamplingParamAuto = {
    **CorrectionResamplingParamSignal,
    'source'            : AutofluoFile,
    'sink'              : os.path.join(BaseDirectory, 'autofluo_resampled_12.tif'),
}

RegistrationResamplingParamSignal = {
    'interpolation'     : 'area',
    'source'            : SignalFile,
    'sink'              : os.path.join(BaseDirectory, 'signal_resampled_25.tif'),
    'resolutionSource'  : OriginalResolution,
    'resolutionSink'    : AtlasResolution,
    'orientation'       : FinalOrientation,
}

RegistrationResamplingParamAuto = {
    **RegistrationResamplingParamSignal,
    'source'            : AutofluoFile,
    'sink'              : os.path.join(BaseDirectory, 'autofluo_resampled_25.tif'),
}

CorrectionAlignmentParam = {
    #moving and reference images
    "fixedImage"   : CorrectionResamplingParamAuto["sink"],
    "movingImage"  : CorrectionResamplingParamSignal["sink"],

    #ants parameter files for alignment. see ants docs for definitions
    'type_of_transform' : 'Affine',
    'reg_iterations'    : (320,320,160,0),
    'aff_sampling'      : 256,

    #directory of the alignment result
    "resultDirectory" :  os.path.join(BaseDirectory, 'ants_signal_to_auto')
    }

RegistrationAlignmentParam = {
    #moving and reference images
    "fixedImage"   : AtlasFile,
    "movingImage"  : RegistrationResamplingParamAuto["sink"],

    #ants parameter files for alignment. see ants docs for definitions
    'type_of_transform' : 'SyNRA',
    'reg_iterations'    : (320,320,160,0),
    'aff_sampling'      : 256,
    'syn_sampling'      : 256,

    #directory of the alignment result
    "resultDirectory" :  os.path.join(BaseDirectory, 'ants_auto_to_atlas')
    }

#################### Heat map generation
# These generally do not need to be modified

voxelizeParameter = {
    "method" : 'Spherical', # Spherical,'Rectangular, Gaussian'
    "size" : (3, 3, 3) # Define size of each voxelized point in pixels
};

######################### Detected Cells Transformation Parameters
# transform coordinates from original positions to the atlas
# These generally do not need to be modified

CorrectionResamplingPointsParam = {
    'dataSizeSource'    : SignalFile,
    'resolutionSource'  : OriginalResolution,
    'resolutionSink'    : CorrectionResolution,
    'orientation'       : FinalOrientation,
}

CorrectionResamplingPointsInverseParam = {
    'dataSizeSource'    : CorrectionResamplingParamAuto['sink'],
    'dataSizeSink'      : SignalFile,
    'resolutionSource'  : CorrectionResolution,
    'resolutionSink'    : OriginalResolution,
    'orientation'       : FinalOrientation
}

RegistrationResamplingPointParam = {
    **CorrectionResamplingPointsParam,
    'resolutionSink'    : AtlasResolution,
    'dataSizeSink'      : RegistrationResamplingParamAuto['sink']
}
