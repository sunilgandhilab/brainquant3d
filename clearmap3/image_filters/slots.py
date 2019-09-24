# -*- coding: utf-8 -*-
import logging
import numpy as np

class Slot(object):

    def __init__(self, value=None, vtype=None, vdims=None):
        if value:
            self._validate_value(value)
        if not isinstance(vtype, list):
            vtype = [vtype]
        if not isinstance(vdims, list):
            vdims = [vdims]

        self.value       = value
        self.valid_types = vtype
        self.valid_dims  = vdims
        self.type        = None
        self.dims        = None
        self.log         = logging.getLogger(__name__)

    def _validate_value(self, value, dims: str = None):

        vtype = None
        for valid_type in self.valid_types:
            if isinstance(value, valid_type):
                vtype = valid_type

        if not vtype:
            raise ValueError(f'Invalid input type. Must be {self.valid_types}')

        dims_is_valid = False
        if isinstance(vtype, (np.array, np.memmap)):
            if not dims or isinstance(dims, int):
                dims = len(value.shape)

            if isinstance(dims, int):
                for valid_dim in self.valid_dims:
                    if isinstance(valid_dim, int):
                        if dims == valid_dim:
                            dims_is_valid = True
                            break
                    elif isinstance(self.valid_dims, str):
                        if dims == len(self.dims):
                            dims_is_valid = True
                            break

            if isinstance(dims, str):
                for valid_dim in self.valid_dims:
                    if isinstance(valid_dim, int):
                        if len(dims) == valid_dim:
                            dims_is_valid = True
                            break
                    elif isinstance(self.valid_dims, str):
                        if dims == valid_dim:
                            dims_is_valid = True
                            break

        if not dims_is_valid:
            raise ValueError(f'Invalid input dimensionality. Must be {self.valid_dims}')

        self.type = type
        self.dims = dims
        return value

    def set(self, value, dims: str = None):
        self.value = self._validate_value(self, value, dims=dims)

    def params(self):
        return {
            'description': self.__doc__,
            'valid_types': self.valid_types,
            'valid_dims': self._valid_dims
        }

class InputSlot(Slot):
    def __init__(self, value=None, vtype=None, vdims=None):
        super().__init__(value=value, vtype=vtype, vdims=vdims)


class ParamSlot(Slot):
    def __init__(self, value=None, vtype=None, vdims=None):
        super().__init__(value=value, vtype=vtype, vdims=vdims)


class OutputSlot(Slot):
    def __init__(self, value=None, vtype=None, vdims=None):
        super().__init__(value=value, vtype=vtype, vdims=vdims)
