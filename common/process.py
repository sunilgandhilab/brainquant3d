# -*- coding: utf-8 -*-
"""
script to run the processing pipeline
"""

# Load the parameters:
exec(open("/mnt/e/jack-temp/brainquant3d-tutorial/parameter.py").read())

################
## Cell Detection:
################
from bq3d.stack_processing.cell_detection import detect_cells
detect_cells(**CellDetectionParams);

################
## Resampling:
################
# Downsampling for the correction of stage movements during the acquisition between channels:
from bq3d.alignment.resampling import resampleData;
resampleData(**CorrectionResamplingParamSignal);
resampleData(**CorrectionResamplingParamAuto);
# Downsampling for alignment to the Atlas:
resampleData(**RegistrationResamplingParamSignal);
resampleData(**RegistrationResamplingParamAuto);

################
## Alignment:
################
from bq3d.alignment.ants import alignData
#correction between channels:
alignData(**CorrectionAlignmentParam)
#alignment to the Atlas:
alignData(**RegistrationAlignmentParam)

################
## Transform Point Coordinates:
################
from bq3d import io
from bq3d.alignment.resampling import resamplePoints
from bq3d.alignment.ants import transformPoints
from bq3d.analysis.voxelization import voxelize
from bq3d.stack_processing.cell_detection import jsonify_points

points = io.readPoints(sink)
# Downsample to chromatic correction size
points = resamplePoints(sink, **CorrectionResamplingPointsParam);
vox = voxelize(points, CorrectionResamplingParamSignal['sink'] , sink = os.path.join(BaseDirectory, 'cells_cd1.tif'), **voxelizeParameter);
# Apply correction transform
points = transformPoints(points, CorrectionAlignmentParam["resultDirectory"], invert = True);
vox = voxelize(points, CorrectionResamplingParamSignal['sink'] , sink = os.path.join(BaseDirectory, 'cells_corr.tif'), **voxelizeParameter);
# Upsample back to original size
points = resamplePoints(points, **CorrectionResamplingPointsInverseParam);
#vox = voxelize(points, SignalFile , sink = os.path.join(BaseDirectory, 'cells_full.tif'), **voxelizeParameter);
# Downsample to atlas resolution
points = resamplePoints(points, **RegistrationResamplingPointParam);
vox = voxelize(points, RegistrationResamplingParamAuto['sink'], sink = os.path.join(BaseDirectory, 'cells_ds.tif'), **voxelizeParameter);
# Warp to atlas
points = transformPoints(points, RegistrationAlignmentParam["resultDirectory"], invert = True);
# Write out heatmap and transformed points
vox = voxelize(points, AtlasFile, sink = os.path.join(BaseDirectory, 'cells_atlas.tif'), **voxelizeParameter);
points = points.T.tolist()
points_with_props = {**io.readData(sink), 'z': points[0], 'y': points[1], 'x': points[2]}
io.writePoints(transformedCellsFile, points_with_props);

# Regional analysis
from bq3d.analysis.region_atlas import Atlas
print('generating atlas')
a = Atlas(collapse=True)
a.add_point_groups(transformedCellsFile)
print('generating dataframe')
a.get_region_info_dataframe('id', ['nPoints', 'points_density', 'volume', 'name', 'parent_id'], sink=os.path.join(BaseDirectory, 'cells_atlas.csv'))
