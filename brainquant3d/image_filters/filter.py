# -*- coding: utf-8 -*-
import shutil
from abc import ABC, abstractmethod

from clearmap3.utils.files import unique_temp_dir
from clearmap3.utils.timer import Timer
import logging

class FilterManager(object):
    """Interface for managing image filters. Holds an index of all available filters
    and is used to get filter instances.

    Attributes:
        _filters (dict): Store of filters as 'filter_name': <class>.
    """

    def __init__(self):
        self._filters = {} # 'filter_name': <class>

    def __repr__(self):
        return f'<image_filters.FilterManager with filters: {self._filters.keys()}>'

    def get_filter(self, im_filter):
        """Returns new filter instance given the name of a filter

        Arguments:
            im_filter (str): Name of filter to return. name should equal filter class name.
        Returns:
            filter (filter object): Returns new instance of filter.
        """

        if im_filter in self._filters:
            return self._filters[im_filter]() # return instance of class
        else:
            raise ValueError(f'{im_filter} filter not found in FilterManager')

    def add_filter(self, im_filter):
        """Adds a filter to the FilterManager. This function should be used to register functions.
        Registers filters to self._filters.

        Arguments:
            im_filter (filter object): Filter to add.
        """

        if not isinstance(im_filter, FilterBase):
            raise ValueError('add_filter requires a FilterBase object')

        filter_name = im_filter.__class__.__name__

        base_class = im_filter.__class__.__bases__[0].__name__
        if base_class == 'FilterBase': # make sur correct base class
            if not filter_name in self._filters.keys():
                self._filters[filter_name] = im_filter.__class__
            else:
                raise ValueError(f'{filter_name} already registered')
        else:
            raise ValueError(f'Cannot add filter with base class {base_class}')

class FilterBase(ABC):
    """Base class to be used as a template for all image filters. Additional input can be added during __init__.
    _generate_output is what does the calculation. It should be writted for each filter.

    Attributes:
        input (array or memmap): Required input. Image to pass through filter.
        output (array): Filter result.
        cleanup (bool): delete temp data folder.
        temp_dir (bool): temp dir for processing. Set to True for function to create own path.
        name (str): name of filter class.
        log (logger): logger instance to be used for filter
    """

    def __init__(self, temp_dir = False):

        self.input    = None
        self.output   = None
        self.cleanup  = False
        self.subStack = None

        self.temp_dir = temp_dir
        self.name     = self.__class__.__name__
        self.log      = logging.getLogger(__name__)

        super().__init__()

    def set_inputs(self, kwargs):
        """Parses inputs into attributes.

        Arguments:
            kwargs (dict): Input argumentas as key value pairs. where key matches the attribute name.
        """

        for attr,value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
            else:
                self.log.debug(f'{attr} not defined in {__name__}')
        self._validate_input()

    @abstractmethod
    def _generate_output(self):
        # Function that does the image filtering. Must return an array. `run` will set the return to `self.output`
        pass

    def _validate_input(self):
        """Check if inputs are of valid type.

        Notes:
            Consider abstract method
            """
        pass

    def run(self):
        """Calculates output and saves it to self.output.

        Returns:
            (array): Filtered image.
        """

        timer = Timer()
        if self.temp_dir != False:
            self.set_temp_dir()
        self.log_parameters()

        try:
            self.output = self._generate_output()
        except Exception as err:
            if self.temp_dir:
                self.del_temp_dir()
            raise err

        if self.temp_dir and self.cleanup:
            self.del_temp_dir()
        timer.log_elapsed()

        return self.output

    def log_parameters(self):
        """prints filter attrubutes to log. Will not print 'output', 'input', 'log' for conciseness.
        """

        for key,value in self.__dict__.items():
            if key not in ['output', 'input', 'log']:
                self.log.verbose(f'{self.name}| {key}: {value}')

    def set_temp_dir(self, root=None):
        """ Creates unique temportary directory
        """
        if self.temp_dir == True:
            if root:
                self.temp_dir = unique_temp_dir(self.name, path=root)
            else:
                self.temp_dir = unique_temp_dir(self.name)
            self.temp_dir.mkdir()
            self.log.debug(f'Set temp path {self.temp_dir}')
        return

    def del_temp_dir(self):
        """ Deletes unique temportary directory
        """
        shutil.rmtree(self.temp_dir.as_posix(), ignore_errors=True)
