# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
from functools import cached_property
from typing import Union
import logging
import csv
from pathlib import Path

from anytree import AnyNode, RenderTree, PostOrderIter, PreOrderIter
from anytree.exporter import JsonExporter
import pandas as pd

import bq3d
from bq3d import io
from bq3d.utils.timer import Timer
from bq3d.analysis import label_properties

from bq3d._version import __version__

__author__ = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__ = "Copyright 2019, Gandhi Lab"
__license__ = 'BY-NC-SA 4.0'
__version__ = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__ = 'ricardo-re-azevedo@gmail.com'
__status__ = "Development"

log = logging.getLogger(__name__)


class Region(AnyNode):
    """ container defining an anatomical region and its related information.
    Built off anytree framework and compatable will all anytree functions.
    Regions can have a heigharchy.
    Required fields for proper function are region['name'], region['id'], region['parent_id']
    Extra attributes are added as kwargs at initializatation or with self.update.
    Arguments:
        name (str): name of the region. analagous to anytree.Node.name
        collapse (bool): if true, propties belonging to this region will include voxels from
        child regions.
        parent (Node): parent node in anytree heigharchy.
    Attributes:
        name (str): name of the region. analagous to anytree.Node.name
        id (int): unique id for corresponding brain region.
        parent (Node): parent node in anytree heigharchy.
        parent_id (Node): parent node id in anytree heigharchy.
        rgb_id (tuple): unique rgb id for corresponding brain region.
        acronym (str): acronym of corresponding brain region.
        parent_acronym (str): acronym of parent region.
        collapse (bool): if true, properties returned by this region invlude all child regions.
        volume (int): volume of region in voxels
        voxels (dict): voxels making up the region and number of data points within that voxel
        as {(x,y,z): [Voxel()]}.
    """

    def __init__(self, name=None, id=None, parent=None, collapse=False, **kwargs):
        self.name = name
        self.id = id
        self.parent = parent

        # additional properties belonging to the region passed thorugh kwargs
        self.properties = kwargs

        self.collapse = collapse

        self.label_props: label_properties.RegionProperties = None
        self.samples = []
        self.params = []
        self._data = None

        # indicates if will return an iterable for each sample.
        self.samplewise_params = ['n_points']

        self._reset_data()

    def __hash__(self):
        return hash(self.id)

    def __eq__(self, other):
        return self.id == other.id

    @property
    def depth(self):
        # depth of 0 would be root
        return len(self.path) - 1

    @cached_property
    def volume(self):
        if self.label_props:
            return self.label_props.area()  # will include children
        else:
            return 0

    @property
    def parent_name(self):
        try:
            return self.parent.name
        except AttributeError:
            return None

    @property
    def n_points(self):

        if self.collapse:
            iter = PreOrderIter(self)
        else:
            iter = [self]

        n_points = pd.Series(data=0, index=self.samples, dtype=int)
        for reg in iter:
            if len(reg._data.index) != 0:
                n_points += reg._data['sample'].value_counts()

        return n_points

    def has_property(self, prop: str) -> bool:
        if prop in self.properties:
            return True
        elif prop in dir(self):  # unlike hasattr with not call @properties
            return True

        return False

    def set_label_props(self, props: label_properties.RegionProperties):
        self.label_props = props

    def add_points(self, props: dict = None):
        """ adds multipe point info. props must contain, in addition to points info, 'sample': the
        sample the point belongs to, and 'voxel', the voxel the point resides in.
        Arguments:
            coord (tuple): coordinate of point
            sample (str): sample to add point to
        """
        if not isinstance(props, pd.DataFrame):
            d = pd.DataFrame(props)
        else:
            d = props
        d['sample'] = d['sample'].astype(self._data['sample'].dtype)  # needs same categorical dtype
        self._data = self._data.append(d, ignore_index=True)

    def add_sample(self, name: str):
        """ adds a new sample name to the region
        Arguments:
            name (str): name of sample to add
        """
        self.samples.append(name)
        self._data['sample'] = self._data['sample'].cat.add_categories(name)

    def compile_params(self, region_params: list, points_params: list,
                       method='sum', voxelwise=False, iter=None):
        """ Compiles parameters on the given region.
        Args:
            region_params (list): list of parameters for the region to compile
            points_params (list): list of parameters for each point to compile
            method (str): pd.agg method to joing points into voxel or regional aggregate. If not
            voxelwise: (bool): aggregate points by voxel instead of region
            include_empty (bool): if voxelwise is true will also return voxels with no points
        Returns:
            pd.Dataframe: requested parameters
        """
        if self.collapse:
            iter = PreOrderIter(self)
        else:
            iter = [self]

        points_params = set(points_params)
        region_params = set(region_params)

        if voxelwise:
            adtl_cols = {'z', 'y', 'x', 'sample'}
            available_special_params = {'n_points', 'voxel'}
        else:
            adtl_cols = {'sample'}
            available_special_params = set()

        special_params = available_special_params & (points_params | region_params)
        nonsp_region_params = list(region_params - special_params - adtl_cols)
        requested_pts_params = list(points_params - special_params - adtl_cols)
        adtl_cols = list(adtl_cols)
        special_params = list(special_params)

        all_cols_raw = adtl_cols + list(points_params) + nonsp_region_params + special_params
        all_cols = sorted(set(all_cols_raw), key=all_cols_raw.index)

        # compile regionwise params
        region_info = self.get_region_props(nonsp_region_params)

        # compile pointwise params
        all_points_dfs = [reg._data for reg in iter]
        all_pts = pd.concat(all_points_dfs)
        all_pts_filt = all_pts.drop([c for c in all_pts.columns if c not in all_cols], axis=1)

        if voxelwise:

            if len(all_pts_filt != 0):
                all_pts_filt[['z', 'y', 'x']] = all_pts_filt[['z', 'y', 'x']].astype(int)
                # too slow if dont convert
                all_pts_filt['sample'] = all_pts_filt['sample'].astype(str)
                reduced = all_pts_filt.groupby(['sample', 'z', 'y', 'x'])[requested_pts_params].agg(
                    method).reset_index()

                if 'n_points' in special_params:
                    n_pts = all_pts_filt.groupby(['sample', 'z', 'y', 'x'])[
                        requested_pts_params[0]].count().reset_index()
                    reduced['n_points'] = n_pts[requested_pts_params[0]]
                if 'voxel' in special_params:
                    reduced['voxel'] = all_pts_filt.groupby(['sample', 'z', 'y', 'x'])[
                        'voxel'].first().values

            else:
                reduced = pd.DataFrame([], columns=all_cols)

            if 'n_points' in special_params:
                n_pts = all_pts_filt.groupby(['sample', 'z', 'y', 'x'])['sum'].count().reset_index()
                reduced['n_points'] = n_pts['sum']

            reduced[nonsp_region_params] = region_info.iloc[0][nonsp_region_params]
            return reduced

        else:
            reduced = all_pts_filt.groupby('sample').agg(method)
            if reduced.empty:
                return region_info
            return reduced.join(region_info, sort=False).reset_index()

    def get_region_props(self, props: list):
        """ return parameters about the region in a DataFrame
        Args:
            props (list): parameters to return
        """
        data = pd.DataFrame(index=self.samples, columns=props)
        for p in props:
            if hasattr(self, p):
                data[p] = getattr(self, p)
            else:
                data[p] = self.properties[p]
        return data

    def apply(self, value, func, *args):
        """ applies a funcion to each point in each voxel in each region. used to generate
        new params. each point is treated as a pandas series.
        Args:
            value (str): name of param to save to.
            func (lambda function): lambda function to apply to each point.
        """
        if len(self._data.index) != 0:
            self._data[value] = self._data.apply(func, axis=1)

    def filter(self, value, func, *args):
        """ filters point in each voxel. Keeps points where the given function returns true.
        The field provided in `val` will be the value passed to the function
        Args:
            value (str): name of param to filter by.
            func (function): function to apply to each point.
        """

        self._data.filter(value, func)

    def clear(self):
        """ removes points info"""

        self.samples = []
        self._reset_data()

    def update(self, **kwargs):
        """ updates all attribues in kwargs
        Arguments:
            kwargs (dict): attributes to update as, attribute: value
        """

        for key, value in list(kwargs.items()):
            setattr(self, key, value)

    def print_tree(self):
        """ prints tree info using `anytree.RenderTree` """

        print((RenderTree(self)))

    def _reset_data(self):
        self._data = pd.DataFrame({'voxel': pd.Series([], dtype='object'),
                                   'sample': pd.Series([], dtype='category')
                                   })


class Atlas(object):
    """ Container defining an anatomical atlas and all of its related info.
    Stores regions under self.tree_root using anytree framework and compatable will all anytree
    functions. At initialization the Atlas will populate self.tree_root with a heigharchy of
    regions with info from the labeled image and annotation file..
    All modification done to the region heigharchy should be done through this object.
    Arguments:
        label_image (str): labeled image with each label correspondign to a brain region. passed to
        self.image. annotation_file (str): csv file containing annotated info. Required fields
        'name', 'id', 'parent_id'. 'id' should correspond to labels in label_image.
        passed to self.annotation_file.collapse (bool): if True parents will inherit info from
        their children. passed to self.COLLAPSE.
    Attributes:
        image (str): labeled image file
        annotation_file (str): file containing annotation info
        regions_by_id (dict): docts if regions as id: Region pairs {id (int): (region)}.
        quick way to access regions by id
        COLLAPSE (Bool): if True parents will inherit info from their children. not to be modified
        after init
        sample_props (dict): properties that belong to a sample but no particular region. stored
        as dataframe with index as samples and columns as props.
     """

    def __init__(self, label_image: Union[str, Path],
                 annotation_file: Union[str, Path] = bq3d.config.annotations_default_file,
                 collapse=True):

        timer = Timer()

        # all attributes not listed here will be generated by the populate regions routine below
        self.image_file = Path(label_image)
        self.image = io.readData(label_image)
        self.annotation_file = Path(annotation_file)
        self.regions_by_id = {}  # dict of int(region_id): region
        self.tree_root = None
        self.backround = Region(id=None, name='background')
        self.COLLAPSE = collapse
        self.sample_props = pd.DataFrame()

        # populate regions
        atlas_ext = io.fileExtension(self.annotation_file)
        if atlas_ext == 'csv':
            self.populate_regions_from_csv(file=annotation_file)
        else:
            ValueError(f'cannot generate regions from from type {atlas_ext}')

        self._populate_region_props(self.image_file)
        timer.log_elapsed(prefix='Atlas initialization')

    def __iter__(self):
        return PostOrderIter(self.tree_root)

    def populate_regions_from_csv(self, file):
        """ creates region from csv and populates them with attributes in the csv
        Csv should have attributes in columns. header should be the name of the attribute to
        generate. Required attributes are `name`, `id`, `rgb_id`, and `parent_id`.
        Arguments:
            file (str): csv file containing region information
        """
        with open(file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            region_info = []
            for line in reader:
                # reformat parameter names to fit Region() naming scheme
                line['name'] = line['name'].replace(" ", "_")
                line['id'] = int(line['id'])
                line['parent_id'] = int(line['parent_id']) if line['parent_id'] != '' else None
                region_info.append(line)

            self._populate_regions_from_list(region_info)

    def _populate_regions_from_list(self, regions):
        """ creates regions and their attributes given region information as a list of dicts
        where each dict contains a regions attributes
        If a region has region['name'] of 'root' it will be considered the root node.
        If a region has region['name'] of 'background' it will be considered a child of the
        root node. Required fields are region['name'], region['id'], region['parent_id']
        Arguments:
            regions (list): list of dicts where each entry is seperate region with attributes as
            key: value pairs
        """
        for i, region_info in enumerate(regions):
            if region_info['name'] == 'root':
                self.tree_root = Region(collapse=self.COLLAPSE, **region_info)
                self.regions_by_id[region_info['id']] = self.tree_root
            elif region_info['name'] == 'background':
                self.backround = Region(**region_info)
                self.regions_by_id[region_info['id']] = self.backround
            else:
                parent = self.get_region_by_id(region_info['parent_id'])
                if parent is None:
                    continue
                else:
                    region_info['parent'] = parent
                    region = Region(collapse=self.COLLAPSE, **region_info)
                    self.regions_by_id[region_info['id']] = region

            regions.remove(region_info)

        if regions != []:
            self._populate_regions_from_list(regions)

    def _populate_region_props(self, labels):
        """ adds coordinate info to regions under region.voxels. Will also generate self.volume.
        Arguments:
            labels (array or str): labeled image with values corresponding to region id
        """
        #       image = np.array(io.readData(labels, returnMemmap=False), dtype=np.uint32)  # to facilitate
        # image = io.readData(labels, returnMemmap=True).astype(np.uint32)
        # pooling and speed up
        image = io.readData(labels, returnMemmap=True).astype(int)
        log.verbose(f'calculating region info from {labels}')
        properties = {r.label: r for r in label_properties.region_props(image)}

        for region in PostOrderIter(self.tree_root):

            # regions not in image wont have props
            props = [properties[region.id]] if region.id in properties else []

            tree_props = props + [r.label_props for r in PostOrderIter(region) if r.label_props]
            if tree_props:
                merged = label_properties.join_regions(tree_props)
                region.set_label_props(merged)

            log.debug(f'retrieving coordinates and densities for region {region.id}')

    def add_points(self, name, source):
        """ add point counts to each region with voxel info/ points falling in the background
        will be ignored.
        Points will be added to self.voxels as a list at each coordinate making up the region.
        Point density by sample will added to self.points_density
        Arguments:
            name (str): name to give sample
            source: (str, list): array, file, or list of files with point
            coordinates in format {x: [], y: [],z: [], <prop>: []),...}. non-coordinate properties
            will be stored within each voxels record.
        """
        timer = Timer()

        if name in self.tree_root.samples:
            raise ValueError(f'cannot add duplicate sample {name}')

        self.sample_props = self.sample_props.append(pd.Series(name=name))

        # add data
        data = io.readData(source)
        coord_keys = ('z', 'y', 'x')
        props_keys = tuple(k for k in data)

        if not all([p in data for p in coord_keys]):
            raise ValueError('source must have keys "x", "Y", and "z"')

        # add samples
        for region in PostOrderIter(self.tree_root):
            region.add_sample(name)

        # create dict for each point and sample by region
        regionwise_points = {}
        dropped_count = 0
        for i in range(len(data['x'])):
            vox = tuple(int(data[k][i]) for k in coord_keys)
            reg = self.get_region_by_voxel(vox)

            if reg:
                if not reg in regionwise_points:
                    regionwise_points[reg] = []

                props = {k: data[k][i] for k in props_keys}
                regionwise_points[reg].append({'voxel': vox, 'sample': name, **props})
            else:
                dropped_count += 1

        # add points to regions
        for reg, props in regionwise_points.items():
            reg.add_points(props)

        log.verbose(f'Points outside sample: {dropped_count}/{len(data["x"])}')
        timer.log_elapsed(prefix='Added points sample')

    def add_region_info_from_dataframe(self, df, columns=None):

        if columns:
            df = df[columns]

        # set default values
        defaults = df.iloc[0].to_dict()
        for region in PostOrderIter(self.tree_root):
            region.update(**defaults)

        # add info from df
        for index, row in df.iterrows():
            region = self.get_region_by_id(index)
            row_dict = row.to_dict()
            region.update(**row_dict)

    def add_voxelized_points(self, voxels, name, method='divide'):
        """ add voxelized point data. Rather that entering points lists, this is for data where the
        number of points have already been counted for each voxel
        Arguments:
            voxels (str, pd.DataFrame): csv file or dataframe where first column or index is the
            voxel coordinate as '(z,y,x)' with header 'voxel'.
            Must also contain a column 'n_points' for how many points in that voxel. All other
            columns will be split into several points internaly.
            based on the value of 'n_points'
            name (str): method to split points with
            method: we need to store data as individual points. This will decide how to calculate
            the pointwise value of the other columns.
        """
        timer = Timer()

        if isinstance(voxels, pd.DataFrame):
            df = voxels
        else:
            df = pd.read_csv(voxels)

        df['id'] = 0
        df['id'] = df.apply(lambda row: self.get_region_by_voxel(row.name).id, axis=1)

        param_cols = [c for c in df if c not in ['n_points', 'id']]
        if method == 'divide':
            df[param_cols] = df[param_cols].div(df['n_points'], axis=0)
        elif method == 'multiply':
            df[param_cols] = df[param_cols].mul(df['n_points'], axis=0)
        else:
            raise ValueError('method not recognized')

        df = df.reset_index()
        df = df.rename(columns={'index': 'voxel'})

        all_n_pts = df['n_points'].unique()
        duplicate_cts = all_n_pts[all_n_pts > 1]

        for ct in duplicate_cts:
            dup_rows = df[df['n_points'] == ct]
            duplicated = pd.concat([dup_rows for _ in range(ct - 1)])
            df = pd.concat([df, duplicated])

        # add sample
        for region in PostOrderIter(self.tree_root):
            region.add_sample(name)
        df['sample'] = name

        for reg_id, reg_df in df.groupby('id'):
            reg = self.get_region_by_id(reg_id)
            reg_df = reg_df.drop(columns=['n_points', 'id'])
            reg.add_points(reg_df)

        timer.log_elapsed(prefix='Added voxelized points')

    def _params_by_type(self, params, sample=[], region=[], points=[]):

        sample_params = set(sample)
        region_params = set(region)
        points_params = set(points)
        for c in params:
            if c in self.sample_props.columns:
                sample_params.add(c)
            elif self.tree_root.has_property(c):
                region_params.add(c)
            else:
                points_params.add(c)

        return list(sample_params), list(region_params), list(points_params)

    def get_region_info_dataframe(self, index, columns, sink=None,
                                  drop_empty=True, iterator=PostOrderIter):
        """ format region attributes into a dataframe. attributes belonging to vpoints such as
        Arguments:
            index (str): attribute used for row labels.
            columns (list): attributes to be included in columns.
            sink (str): file to save dataframe too.
            drop_empty (bool): if True, will ignore regions with no collapsedvolume.
            iterator (Object): `anynode` iterator to use when populating dataframe. Will determine order of entries.
        Returns:
            pandas.DataFrame: if sink, will return filename of sink.
        """
        timer = Timer()

        sample_ps, region_ps, points_ps = self._params_by_type(columns, region=[index, 'sample'])
        # get column and row attributes by region and merge into list or lists
        region_dfs = []
        for region in iterator(self.tree_root):

            if drop_empty and region.volume == 0:
                continue

            if region.id == 567:
                print('')

            values = region.compile_params(region_ps, points_ps)
            region_dfs.append(values)

        data_df = pd.concat(region_dfs)

        # get sample params
        if sample_ps:
            sample_df = self.sample_props[sample_ps]
            data_df = pd.merge(data_df, sample_df, left_on='sample', right_index=True)

        timer.log_elapsed(prefix='Generated dataframe for regions')

        data_df.set_index(index, inplace=True)
        if sink:
            io.writeData(sink, data_df)
        return data_df

    def get_voxel_info_dataframe(self, columns, sink=None, iterator=PreOrderIter):
        """ format voxel info with corresponding region attributes into a dataframe
        If an attribute in column is type list or tuple each member will get its own column in
           the dataframe
        Rows will always be by voxel.
        Arguments:
            columns (str or list): attributes to be included in columns. 'n_points' will return
           #        point counts by voxel.
            ignore_empty (bool): only return voxels containing points
            sink (str): file to save dataframe too.
            iterator (Object): `anynode` iterator to use when populating dataframe. Will
       #        determine order of entries.
        Returns:
            pandasDataFrame: if sink, will return filename of sink.
        """
        timer = Timer()

        sample_ps, region_ps, points_ps = self._params_by_type(columns, region=['sample'])

        # get column and row attributes by region and merge into list or lists
        region_dfs = []
        for region in iterator(self.tree_root):
            if len(region._data) == 0:
                continue
            region.collapse = False
            values = region.compile_params(region_ps, points_ps, voxelwise=True)
            region.collapse = self.COLLAPSE
            region_dfs.append(values)

        data_df = pd.concat(region_dfs)
        data_df.set_index('sample', inplace=True)

        # get sample params
        if sample_ps:
            sample_df = self.sample_props[sample_ps]
            data_df = pd.merge(data_df, sample_df, left_on='sample', right_index=True)

        timer.log_elapsed(prefix='Generated dataframe for voxels')

        if sink:
            io.writeData(sink, data_df)
        return data_df

    def get_points_info_dataframe(self, columns, sink=None, iterator=PreOrderIter):
        """ format points info with corresponding region attributes into a dataframe. The sample
        the point belongs to will be included in an columns labled 'sample' If an attribute in
        column is type list or tuple each member will get its own column in
        the dataframe.
        Arguments:
            columns (str or list): attributes to be included in columns.
            sink (str): file to save dataframe too.
            iterator (Object): `anynode` iterator to use when populating dataframe. Will
            determine order of entries.
        Returns:
            pandasDataFrame: if sink, will return filename of sink.
        """

        timer = Timer()
        sample_ps, region_ps, points_ps = self._params_by_type(columns, points=['sample'])

        # get column and row attributes by region and merge into list or lists
        pts_dfs = []
        for region in iterator(self.tree_root):
            if len(region._data) == 0:
                continue

            points_info = region._data[points_ps]
            region_info = region.get_region_props(region_ps)
            points_info[region_info.columns] = region_info.iloc[0]

            pts_dfs.append(points_info)

        data_df = pd.concat(pts_dfs)
        data_df.set_index('sample', inplace=True)

        # get sample params
        if sample_ps:
            sample_df = self.sample_props[sample_ps]
            data_df = pd.merge(data_df, sample_df, left_on='sample', right_index=True).reset_index()

        timer.log_elapsed(prefix='Generated dataframe for points')

        if sink:
            io.writeData(sink, data_df)
        return data_df

    def get_region_by_id(self, region_id):
        """ returns Region object given id
        Arguments:
            region_id (int): id of region
        Returns:
            Region: region with corresponding id
        """
        try:
            return self.regions_by_id[region_id]
        except Exception:
            return None

    def get_region_by_voxel(self, voxel: tuple):
        """ returns Region object given voxel coordinate
        Arguments:
            voxel (tuple): (x,y,z) coordinate corresponding to tuple
        Returns:
            Region: region with corresponding id
        """
        if any(v < 0 for v in voxel):
            return self.backround
        try:
            rid = self.image[voxel]
        except IndexError:
            return self.backround
        return self.get_region_by_id(rid)

    def add_sample_props(self, props: dict, join_on: str = 'name'):
        """ Adds sample properties to the atlas.
        Args:
            props (dict): parameters that belong to a sample but no particular region. stored
            join_on (str): key within props to join on. must match the name of the points table.
        as {'sample': {'param': value}}
        """
        props_df = pd.DataFrame(props)
        props_df.set_index(join_on, inplace=True)
        self.sample_props = pd.concat([self.sample_props, props_df], axis=1)

    def apply(self, value, func, *args):
        """ applies a function to each point in each voxel in each region. used to generate
        new params. each point is treated as a pandas series.
        Args:
            value (str): name of param to save to.
            func (function): function to apply to each point.
        """
        timer = Timer()
        for region in PostOrderIter(self.tree_root):
            region.apply(value, func, *args)
        timer.log_elapsed(prefix='apply complete')

    def filter(self, value, func):
        """ filters point in each voxel in each region. Keeps points where the given function
        returns true. The field provided in `val` will be the value passed to the function.
        Args:
            value (str): name of param to filter by.
            func (function): function to apply to each point.
        """
        timer = Timer()
        for region in PostOrderIter(self.tree_root):
            region.filter(value, func)
        timer.log_elapsed(prefix='filter complete')

    def filter_regions(self, filt):
        """ filters out regions based on their attributes
        Arguments:
            filt (function or lambda): function to apply to each node. Should return True to keep node
        """
        for region in PostOrderIter(self.tree_root):
            if filt(region) is not True:
                siblings = list(region.parent.children)
                siblings.pop(siblings.index(region))
                region.parent.children = siblings

    def clear(self):
        """ removes voxel info from all regions"""
        self.sample_props = pd.DataFrame([])
        for region in PostOrderIter(self.tree_root):
            region.clear()

    def del_attribute(self, attr):
        """ deletes an attribute from all regions
        Arguments:
            attr (str): attribute to delete
        """

        delattr(self.backround, attr)
        for i in PostOrderIter(self.tree_root):
            delattr(i, attr)

    def print_tree(self, tag='id'):
        """ prints tree to console
        Arguments:
            tag (str): attribute to display for each node.
        """

        for pre, _, node in RenderTree(self.tree_root):
            attr = getattr(node, tag)
            print(f"{pre}{attr}")

    def to_json(self, sink=None, **kwargs):
        """ writes region tree info to json
        Arguments:
            sink (str or None): file to save to. if None, will return json object.
            kwargs: addtional arguments to pass to anytree.exporter.jsonexporter.JsonExporter and json.dumps.
        """

        exporter = JsonExporter(indent=2, **kwargs)
        if sink:
            with open(sink, 'w') as outfile:
                exporter.write(self.tree_root, outfile)
            return sink
        else:
            data = exporter.export(self.tree_root)
            return data
