import re
import uuid
from bq3d import config
from pathlib import Path

from bq3d._version import __version__
__author__     = 'Ricardo Azevedo, Jack Zeitoun'
__copyright__  = "Copyright 2019, Gandhi Lab"
__license__    = 'BY-NC-SA 4.0'
__version__    = __version__
__maintainer__ = 'Ricardo Azevedo'
__email__      = 'ricardo-re-azevedo@gmail.com'
__status__     = "Development"


def unique_temp_dir(folder, path = config.temp_dir):
    """ Creates a unique temp.
    """
    counter = 0
    while True:
        counter += 1
        tpath = Path(path) / f'{folder}{uuid.uuid4()}'
        if not tpath.exists():
            return tpath


def sort(list:list):
    """ Sorts the given iterable in a predictable way.

    Arguments:
        list (list): list to sort

    Returns:

    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(list, key=alphanum_key)
