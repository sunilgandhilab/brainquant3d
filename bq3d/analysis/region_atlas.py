# -*- coding: utf-8 -*-
import logging
import csv
import pandas as pd
import numpy as np
from ast import literal_eval
from bq3d.analysis.label_properties import region_props
from anytree import AnyNode, RenderTree, PostOrderIter, PreOrderIter
from anytree.exporter import JsonExporter

import bq3d
from bq3d import io
from bq3d.utils.timer import Timer

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"

log = logging.getLogger(__name__)

class Region(AnyNode):
    """ container defining an anatomical region and its related information.

    Built off anytree framework and compatable will all anytree functions.
    Regions can have a heigharchy.
    Required fields for proper function are region['name'], region['id'], region['parent_id']
    Extra attributes are added as kwargs at initializatation or with self.update.

    Arguments:
        name (str): name of the region. analagous to anytree.Node.name
        parent (Node): parent node in anytree heigharchy.

    Attributes:
        name (str): name of the region. analagous to anytree.Node.name
        id (int): unique id for corresponding brain region.
        parent (Node): parent node in anytree heigharchy.
        rgb_id (tuple): unique rgb id for corresponding brain region.
        acronym (str): acronym of corresponding brain region.
        parent_acronym (str): acronym of parent region.
        volume (int): volume of region in voxels
        voxels (dict): voxels making up the region and number of data points within that voxel as {(x,y,z): [points]}. Each position in [points] should be from a different data points group.
        voxels_collapsed (dict): same format as :attr:voxels but returns voxels for region and subregions. This is a @property.
        voxels_nonzero (dict): same format as :attr:voxels but only returns voxels that have points. This is a @property.
        voxels_collapsed_nonzero (dict): same format as :attr:voxels but returns  only voxels that have points in region and subregions. This is a @property.
        nPoints (list): total number of points in region. Each position in list is from a different data points group.
        points_density (list): density of points in region. Each position in list is from a different data points group.
    """

    def __init__(self, parent = None, **kwargs):
        self.name            = None
        self.id              = None
        self.parent          = parent
        self.parent_id       = None
        self.rgb_id          = None
        self.acronym         = None
        self.parent_acronym  = None
        self.volume          = 0
        self.voxels          = {}

        self.nPoints         = []
        self.points_density  = []
        # all kwargs aside from the required ones above
        # get placed in an attribute named after the key
        for key, value in list(kwargs.items()):
            setattr(self, key, value)
    
    @property            
    def voxels_nonzero(self):
        return {k:v for k,v in self.voxels.items() if sum(v) != 0}
    
    @property
    def voxels_collapsed(self):
        res = {}
        for reg in PreOrderIter(self):
            res.update(reg.voxels)
        return res
        
    @property
    def voxels_collapsed_nonzero(self):
        res = {}
        for reg in PreOrderIter(self):
            res.update(reg.voxels_nonzero)
        return res

    def add_volume(self, volume):
        """ adds volume info

        Arguments:
            volume (int): volume of brain region
        """
        self.volume += int(volume)

    def add_voxel(self, voxel):
        """ adds voxel info

        Arguments:
            voxel (tuple): voxel coordinate in (x,y,z) to add to region
        """

        #self.volume += 1
        self.voxels[voxel] = []

    def add_point(self, coord, group):
        """ adds point info

        Arguments:
            coord (tuple): coordinate of point
            group (int): position of group to add to
        """

        self.voxels[coord][group] += 1
        self.nPoints[group] += 1

    def add_voxel_groups(self, coord, groups, index = None):
        """ adds point info

        Arguments:
            coord (tuple): voxel coordinate of points
            groups (list): list of point counts by group
            index (slice): indicies to append from. If none, append to end
        """
        if index is None:
            idx = len(self.nPoints)
            index = slice(idx, None)
        self.voxels[coord][index] = groups
        self.nPoints[index] = list(np.add(self.nPoints[index], groups))

    def add_empty_voxel_groups(self, ngroups = 1, index = None):
        """ appends points groups with value 0

        Arguments:
            ngroups (int): number of entries to add
            index (nslice): indicies to append from. If none, append to end
        """
        if index is None:
            idx = len(self.nPoints)
            index = slice(idx, None)
        if self.voxels:
            for key in self.voxels:
                self.voxels[key][index] = [0] * ngroups

        self.nPoints[index] = [0] * ngroups

    def collapse_volume(self):
        """ adds regions volume to parent """

        if self.parent:
            self.parent.volume += self.volume

    def collapse_voxels(self):
        """ add voxel info to parent. root will never inherit voxels """

        if self.parent:
            if not self.parent.name == 'root':
                self.parent.voxels = {**self.parent.voxels, **self.voxels}


    def collapse_nPoints(self, index = None):
        """ add nPoints info to parent

        Arguments:
            index (slice): indicies to collapse. if none, will collapse all entries
        """

        if index is None:
            index = slice(None)
        if self.parent:
            if not self.nPoints:
                self.nPoints[index] = [0] * len(self.parent.nPoints[index])
            self.parent.nPoints[index] = list(np.add(self.parent.nPoints[index],self.nPoints[index]))

    def clear(self):
        """ removes voxel info"""
        self.nPoints = []
        self.points_density = []
        self.voxels = dict.fromkeys(self.voxels, [])

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

    def add_attr_across_tree(self, attribute, value):
        """ adds an attribute to all Regions

        Arguments:
            attribute (str): name of attribute to add
            value: value to give to attribute
        """

        for region in PostOrderIter(self):
            setattr(region, attribute, value)


class Atlas(object):
    """ Container defining an anatomical atlas and all of its related info.

    Stores regions under self.tree_root using anytree framework and compatable will all anytree functions.
    At initialization the Atlas will populate self.tree_root with a heigharchy of regions with info from the labeled image and annotation file..
    All modification done to the region heigharchy should be done through this object.

    Arguments:
        label_image (str): labeled image with each label correspondign to a brain region. passed to self.image.
        annotation_file (str): csv file containing annotated info. Required fields 'name', 'id', 'parent_id'. 'id' should correspond to labels in label_image. passed to self.annotation_file.
        collapse (bool): if True parents will inherit info from their children. passed to self.COLLAPSE.

    Attributes:
        image (str): labeled image file
        annotation_file (str): file containing annotation info
        regions_by_id (dict): docts if regions as id: Region pairs {id (int): (region)}. quick way to access regions by id
        COLLAPSE (Bool): if True parents will inherit info from their children. not to be modified after init
     """

    def __init__(self, label_image = bq3d.config.labeled_image,
                 annotation_file = bq3d.config.annotations_default_file,
                 collapse = True):


        # all attributes not listed here will be generated by the populate regions routine below
        self.image                    = label_image
        self.annotation_file          = annotation_file
        self.regions_by_id            = {} # dict of int(region_id): region
        self.regions_by_coord         = {}
        self.tree_root                = None
        self.backround                = None
        self.COLLAPSE                 = collapse

        # populate regions
        timer = Timer()
        atlas_ext = io.fileExtension(self.annotation_file)
        if atlas_ext == 'csv':
            self.populate_regions_from_csv(file = annotation_file)
        else:
            ValueError(f'cannot generate regions from from type {atlas_ext}')

        self._populate_region_coordiantes(self.image)
        timer.log_elapsed(prefix = 'Atlas initialization')

    def populate_regions_from_csv(self, file):
        """ creates region from csv and populates them with attributes in the csv

        Csv should have attributes in columns. header should be the name of the attribute to generate.
        Required attributes are `name`, `id`, `rgb_id`, and `parent_id`.

        Arguments:
            file (str): csv file containing region information
        """

        with open(file, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            region_info = []
            for line in reader:
                # reformat parameter names to fit Region() naming scheme
                line['name']        = line['name'].replace(" ", "_")
                line['id']          = int(line['id'])
                line['rgb_id']      = str((int(line.pop('red')), int(line.pop('green')), int(line.pop('blue'))))
                if line['parent_id'] != '':
                    line['parent_id'] = int(line.pop('parent_id'))
                else:
                    line['parent_id'] = None
                region_info.append(line)

            self._populate_regions_from_list(region_info)

    def _populate_regions_from_list(self, regions):
        """ creates regions and their attributes given region information as a list of dicts where each dict contains a regions attributes

        If a region has region['name'] of 'root' it will be considered the root node.
        If a region has region['name'] of 'background' it will be considered a child of the root node.
        Required fields are region['name'], region['id'], region['parent_id']

        Arguments:
            regions (list): list of dicts where each entry is seperate region with attributes as key: value pairs
        """
        for i,region_info in enumerate(regions):
            if region_info['name'] == 'root':
                self.tree_root = Region(**region_info)
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
                    region = Region(**region_info)
                    self.regions_by_id[region_info['id']] = region

            regions.remove(region_info)

        if regions != []:
            self._populate_regions_from_list(regions)

    def _populate_region_coordiantes(self, labels):
        """ adds coordinate info to regions under region.voxels. Will also generate self.volume.

        Arguments:
            labels (array or str): labeled image with values corresponding to region id
        """
        image = io.readData(labels, returnMemmap = False).astype(int) # to facilitate pooling and speed up
        log.verbose(f'calculating region info from {labels}')
        properties = region_props(image)

        # add voxels
        for i,prop in enumerate(properties):
            log.verbose(f'retrieving coordinates and densities for region {i}')
            region = self.get_region_by_id(prop.label)
            region.volume = int(prop.area())
            for c in prop.coords():
                c = tuple(c)
                region.add_voxel(c)
                self.regions_by_coord[c] = region

        # label value 0 is ignored by region_props
        self.get_region_by_id(0).add_volume(int(np.sum(image == 0)))

        if self.COLLAPSE:
            for region in PostOrderIter(self.tree_root):
                region.collapse_volume()

    def add_point_groups(self, points_sources):
        """ add point counts to each region with voxel info

        Points will be added to self.voxels as a list at each coordinate making up the region.
        Point density by group will added to self.points_density

        Arguments:
            points_sources (array, str, list): array, file, or list of files with point coordinates in [(x,y,z),...]
        """
        timer = Timer()
        if not isinstance(points_sources, list):
            points_sources = [points_sources]

        start_group = len(self.tree_root.nPoints)

        n = len(points_sources)
        for region in PostOrderIter(self.tree_root):
            region.add_empty_voxel_groups(ngroups = n)
        self.backround.add_empty_voxel_groups(ngroups = n)

        for group, points in enumerate(points_sources):
            group_index = start_group + group
            coords = io.readPoints(points).astype(int)

            for i in coords:
                i = tuple(i)
                try:
                    self.get_region_by_voxel(i).add_point(i, group_index)
                except: # if coord is out of the image it will be included in backgroud
                    self.get_region_by_id(0).nPoints[group_index] += 1

        if self.COLLAPSE:
            for region in PostOrderIter(self.tree_root):
                region.collapse_nPoints(index = slice(start_group, None))

        # calculate densities
        self.calc_point_densities()
        timer.log_elapsed(prefix='Added points group')

    def add_region_info_from_dataframe(self, df, columns = None):

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


    def add_voxelized_points_from_csv(self, voxels_file):
        """ add voxelized point data. Rather that entering points lists, this is for data where the number of points have already been counted for each voxel

        Arguments:
            voxels_file (str): csv file where first column is the voxel coordinate as '(x,y,z)' with header 'voxel'.
            each subsequent column contains point counts within that voxel with each column as a seperate data group.
        """
        timer = Timer()

        df = pd.read_csv(voxels_file, index_col='voxel')
        df = df.loc[(df != 0).any(axis=1)]  # ignore voxels with only 0 values

        total_vox = df.shape[0]
        first_group_index = len(self.tree_root.nPoints)
        s = slice(first_group_index, None)

        for region in PostOrderIter(self.tree_root):
            region.add_empty_voxel_groups(ngroups = df.shape[1], index = s)

        i = 0
        for idx,counts in df.iterrows():
            if i % 10000 == 0:
                log.verbose(f'adding voxel {i} of {total_vox}')
            idx = literal_eval(idx)
            counts = list(counts)
            self.get_region_by_voxel(idx).add_voxel_groups(idx, counts, index = s)
            i += 1

        if self.COLLAPSE:
            for region in PostOrderIter(self.tree_root):
                region.collapse_nPoints() #TODO: will not properly collapse because no index passed

        self.calc_point_densities()
        timer.log_elapsed(prefix='Added voxelized points')

    def calc_point_densities(self):
        """ calculate density of points by region and save them in Region.points_density"""
        log.verbose('calculating points density')
        groups = list(range(len(self.tree_root.nPoints)))
        for region in PostOrderIter(self.tree_root):
            region.points_density = []
            for gp in groups:
                if region.volume == 0:
                    region.points_density.append(0)
                else:
                    region.points_density.append(region.nPoints[gp] / region.volume)

    def get_region_info_dataframe(self, index, columns, sink = None, iterator = PreOrderIter):
        """ format region attributes into a dataframe

        If an attribute in column is type list or tuple each member will get its own column in the dataframe

        Arguments:
            index (str): attribute used for row labels.
            columns (str or list): attributes to be included in columns.
            sink (str): file to save dataframe too.
            iterator (Object): `anynode` iterator to use when populating dataframe. Will determine order of entries.
        Returns:
            pandas.DataFrame: if sink, will return filename of sink.
        """
        timer = Timer()
        if not isinstance(columns, list):
            columns = [columns]

        row_labels = []
        data = []
        # get column and row attributes by region and merge into list or lists
        for region in iterator(self.tree_root):
            row_labels.append(getattr(region, index, None))
            region_attrs = []
            for att in columns:
                value = getattr(region, att, None)

                if isinstance(value, (list, tuple)):
                    region_attrs.extend(value)
                else:
                    region_attrs.append(value)
            data.append(region_attrs)

        # create column labels. If attribue has a lenght, duplicate column headers will be added.
        col_labels = []
        for col in columns:
            att = getattr(self.tree_root, col, None)
            if isinstance(att, (list, tuple)):
                col_labels.extend([f'{col}.{i}' for i in range(len(att))])
            else:
                col_labels.append(col)

        data_df = pd.DataFrame(data=data, index=row_labels, columns=col_labels)
        data_df.index.name = index
        timer.log_elapsed(prefix='Generated dataframe for regions')

        if sink:
            return io.writeData(sink, data_df)
        else:
            return data_df

    def get_voxel_info_dataframe(self, columns, ignore_empty = False, sink=None, iterator=PreOrderIter):
        """ format voxel info with corresponding region attributes into a dataframe

        If an attribute in column is type list or tuple each member will get its own column in the dataframe
        Rows will always be by voxel.

        Arguments:
            columns (str or list): attributes to be included in columns. 'nPoints' will return point counts by voxel.
            ignore_empty (bool): only return voxels containing points
            sink (str): file to save dataframe too.
            iterator (Object): `anynode` iterator to use when populating dataframe. Will determine order of entries.
        Returns:
            pandasDataFrame: if sink, will return filename of sink.
        """
        timer = Timer()
        if not isinstance(columns, list):
            columns = [columns]

        row_labels = []
        data = []
        # get column and row attributes by region and merge into list or list s
        for region in iterator(self.tree_root):
            print(f'Adding region: {region.id}')
            for vox, points in list(region.voxels.items()):
                
                if ignore_empty:
                    if sum(points) == 0:
                        continue
                
                row_labels.append(vox)
                region_attrs = [vox[0],vox[1],vox[2]]
                for att in columns:
                    if att == 'nPoints':
                        value = points
                    else:
                        value = getattr(region, att, None)

                    if isinstance(value, (list, tuple)):
                        region_attrs.extend(value)
                    else:
                        region_attrs.append(value)
                data.append(region_attrs)

        # create column labels. If attribue has a lenght, duplicate column headers will be added.
        col_labels = ['x','y','z']
        for col in columns:
            att = getattr(self.tree_root, col, None) # voxel points list will be same lenght as nPoints
            if isinstance(att, (list, tuple)):
                col = [f'{col}{i}' for i in range(len(att))]
                col_labels.extend(col)
            else:
                col_labels.extend([col])

        data_df = pd.DataFrame(data=data, index=row_labels, columns=col_labels)
        data_df.index.name = 'voxel'
        timer.log_elapsed(prefix='Generated dataframe for voxels')

        if sink:
            return io.writeData(sink, data_df)
        else:
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
        except:
            return None

    def get_region_by_voxel(self, voxel):
        """ returns Region object given voxel coordinate

        Arguments:
            voxel (tuple): (x,y,z) coordinate corresponding to tuple
        Returns:
            Region: region with corresponding id
        """
        try:
            return self.regions_by_coord[voxel]
        except:
            return None

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

    def print_tree(self, tag = 'id'):
        """ prints tree to console

        Arguments:
            tag (str): attribute to display for each node.
        """

        for pre, _, node in RenderTree(self.tree_root):
            attr = getattr(node, tag)
            print(f"{pre}{attr}")

    def to_json(self, sink= None, **kwargs):
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
