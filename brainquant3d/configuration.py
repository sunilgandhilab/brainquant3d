# -*- coding: utf-8 -*-
"""
Module to setup configuration parameters. Parameters are generally loaded from a yaml file.
A default file is provided in the package root.
Additional users can be added under 'user' to provide user/system specific settings.

    e.g.
    mypc:
        # Paths
        Ilastik_path:        '/home/somewhere/ilastik-1.3.2post1-Linux'
        Thread_ram_max_Gb:   200000

The package will first attempt to load user specific settings. If they are not present,
or invalid, the defaults will be used. the user name should match the host name.
"""

import os
import socket
import yaml
from pathlib import Path

from clearmap3.utils.logger import setup_logging

import logging
log = logging.getLogger(__name__)

from clearmap3._version import __version__
__author__     = 'Ricardo Azevedo'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"

class Config(object):
    """Object to hold global settings and configure package environment"""

    def __init__(self):
        path = _get_ClearMapPath()
        self.ClearMap_path = path
        # setup Clearmap_path yal constructor
        yaml.add_constructor('!pkg_path', _prepend_ClearMap_path)

        # get config file
        conf_file = os.path.join(path, 'ClearMap.conf')
        if os.path.isfile(conf_file):
            with open(os.path.join(path, 'ClearMap.conf'), 'rt') as f:
                conf = yaml.load(f.read(), Loader=yaml.Loader)
        else:
            with open(os.path.join(path, 'default.conf'), 'rt') as f:
                conf = yaml.load(f.read(), Loader=yaml.Loader)

        # get user specific config
        hostname = socket.gethostname()
        if hostname in conf['user']:
            user_conf = conf['user'][hostname]
        else:
            user_conf = conf['user']['default']
        default_conf = conf['user']['default']

        #setup logger
        self.console_level = _choose_valid_value(user_conf, default_conf, 'Console_level')
        setup_logging(log_path = os.path.join(self.ClearMap_path, 'logs/'), console_level=self.console_level)

        # set params
        self.processes                = _choose_valid_value(user_conf, default_conf, 'Processing_cores')
        self.thread_ram_max           = _choose_valid_value(user_conf, default_conf, 'Thread_ram_max_Gb')

        # set file paths
        self.annotations_default_file = _choose_valid_value(user_conf, default_conf, 'Annotations_default', path=True)
        self.labeled_image            = _choose_valid_value(user_conf, default_conf, 'Labeled_default', path=True)
        self.temp_dir                 = _choose_valid_value(user_conf, default_conf, 'Temp_path', path = True, create = True)

        # Elaxtix settings
        self.elastix_path             = _choose_valid_value(user_conf, default_conf, 'Elastix_path', path = True)
        if self.elastix_path:
            self.elastix_binary       = _check_exists(os.path.join(self.elastix_path, 'bin/elastix'))
            self.tranformix_binary    = _check_exists(os.path.join(self.elastix_path, 'bin/transformix'))
        else:
            self.elastix_binary       = None
            self.tranformix_binary    = None

        self.Rigid_default_file       = _choose_valid_value(user_conf, default_conf, 'Rigid_default', path = True)
        self.Affine_default_file      = _choose_valid_value(user_conf, default_conf, 'Affine_default', path = True)
        self.BSpline_default_file     = _choose_valid_value(user_conf, default_conf, 'BSpline_default', path = True)

        # Ilaxtik settings
        self.ilastik_path             = _choose_valid_value(user_conf, default_conf, 'Ilastik_path', path = True)
        if self.ilastik_path:
            self.ilastik_binary       = _check_exists(os.path.join(self.ilastik_path, 'run_ilastik.sh'))
        else:
            self.ilastik_binary       = None


def _choose_valid_value(user: dict, default: dict, param: str, path: bool = False, create: bool = False):
    """
    returns user specific parameter if it exists in the user config; otherwise, return the default.
    Paths are treated the same but path will only be returned if the path exists.
    If 'create' path wil be created if it does not exist.

    Args:
        user (dict): user config to get parameter value from.
        default (dict): config to get default value from.
        param (str): parameter in the configs to test.
        path (bool): Treat parameter as a file path.
        create (bool): if param is a path that does not exist, create it.

    Returns:

    """

    if param in user:
        if path:
            path = _check_exists(user[param], create = create)
            log.debug('User defined path: {} set to {}'.format(path, user[param]))
            return path
        else:
            log.debug('User defined config: {} set to {}'.format(param, user[param]))
            return user[param]

    elif param in default:
        if path:
            path = _check_exists(default[param], create = create)
            log.debug('Default path: {} set to {}'.format(path, default[param]))
            return path
        else:
            log.debug('Default config: {} set to {}'.format(param, default[param]))
            return default[param]

    else:
        raise RuntimeError(param +' not defined in ClearMap.conf')


def _check_exists(path, create = True):
    """Checks if path exists and creates it if create is True."""

    if os.path.exists(path):
        return path

    # if 'create' will create folder if not found
    elif create:
        try:
            os.makedirs(path)
            log.debug('Using path: ' + path)
            return path
        except:
            log.warning('Could not create path at {}'.format(path))
            return None

    else:
        log.warning('Path {} does not exist'.format(path))
        return None

def _prepend_ClearMap_path(loader, node):
    """joins Config.ClearMap_path to a path in response to the !ClearMap_path constructor """

    path = os.path.join(_get_ClearMapPath(), node.value)
    return path


def _get_ClearMapPath():
    """Returns root path to the ClearMap software

    Returns:
        str: root path to ClearMap
    """
    fn = os.path.split(__file__)
    fn = Path(os.path.abspath(fn[0]))
    return fn
