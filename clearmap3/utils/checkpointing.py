import os
import tempfile
from collections import defaultdict
import re
import functools
import pickle

import logging
log = logging.getLogger(__name__)

def writeCheckpoint(file, dt):
    """Writes data to pickle.
        Used for checkpointing.

        Arguments:
           file (str): directory to save JSON to.
           dt: variables to save to JSON
    """

    #use tmp if file not defines
    if not file or file == '':
        new_file, file = tempfile.mkstemp(suffix = '.json')
    #create destination folder if not exist
    path = os.path.dirname(file)
    if path:
        os.makedirs(os.path.dirname(file), exist_ok=True)

    log.info('Writing state data to pickle: ' + file)
    with open(file, 'wb') as handle:
        pickle.dump(dt, handle, protocol=pickle.HIGHEST_PROTOCOL)

    log.debug('Done writing state data')
    return file
    # previously used json
    #def serialize(o):
    #    if o == 'results':
    #        print (o)
    #    if isinstance(o, dict):
    #        return {k: serialize(v) for k,v in o.items()}
    #    elif isinstance(o, list):
    #        return [serialize(i) for i in o]
    #    elif isinstance(o, tuple):
    #        return tuple(serialize(i) for i in o)
    #    elif isinstance(o, np.ndarray):
    #        return o.tolist()
    #    else:
    #        return o

    ##use tmp if file not defines
    #if not file or file == '':
    #    new_file, file = tempfile.mkstemp(suffix = '.json')

    ##create destination folder if not exist
    #os.makedirs(os.path.dirname(file), exist_ok=True)

    #print ('Writing state data to json: ' + file)
    #with open(file, 'w') as df:
    #    json.dump(serialize(dt), df, sort_keys=True, indent=4, default=repr)
    #print ('Done writing state data')
    #return file

def renameCheckpoint(file, newFile):
    """renames JSON file.
        Used for checkpointing.

    Arguments:
        file (str): input file
        newFile: outputfile
    """

    if os.path.isfile(file):
        os.rename(file, newFile)
        log.debug('Renaming state data: ' + file + ' to ' + newFile)
    else:
        log.debug('No file containing state data found')

def compareCheckpoint(file1, file2):
    """check for differences between two checkpoint JSON files.
        Used for checkpointing.

        Arguments:
           file1 (str): firstinput file
           file2 (str): second input file
           verbose (Bool): print information on comparisons

        Returns:
           dict or None: a dict of tuples in form 'entry with differences' : (value from d1, value from d2). IF None, checkpoints are the same.
    """
    if os.path.isfile(file1) & os.path.isfile(file2):
        #load in pickle
        with open(file1, 'rb') as handle:
            db1 = pickle.load(handle)
        with open(file2, 'rb') as handle:
            db2 = pickle.load(handle)
        difference = compareDict(db1, db2)

        if 'argdata' in list(difference.keys()): # argdata will always be diff because of memaddresses
            del difference['argdata']
        result = discardDict(difference, '(?<=<function )(.+)(?= at)') # erase entries that are just different memory adresses for the same functions.

        log.debug('Difference between checkpoints: ' + str(result))

    else:
        raise RuntimeError('checkpoint for comparison does not exist or is in a different directory')

    return result

##############################################################################
# Dict tools
##############################################################################


def compareDict(d1, d2, path=[], difference = {}, verbose = False):
    """check for differences between two dicts.
        limitation- new entries in d2 will not be picked up.

        Arguments:
           d1 (str): dict to compare to
           d2 (str): dict being compared
           path (list): list of nested dicts taken to reach key.
           difference (dict): a dict of tuples in form 'entry with differences' : (value from d1, value from d2)
           verbose (Bool): print information on comparisons

        Returns:
           dict: a dict of tuples in form 'entry with differences' : (value from d1, value from d2)
    """

    for k in list(d1.keys()):
        if not k in d2:
             log.debug(k + " as key not in d2")
        else:
            if type(d1[k]) is dict: # enter nested dicts
                pth = path + [k]
                compareDict(d1[k],d2[k], path = pth, difference = difference)
            else:
                if d1[k] != d2[k]:
                    if not path:
                        difference[k] = (d1[k],d2[k])
                    else:
                        tempD = nested_dd()
                        functools.reduce(lambda d,key: d[key],path,tempD).update({k:(d1[k],d2[k])}) # probably more efficient way of doing this

                        tempD = nested_dd_to_dict(tempD) # convert back to regular dict to merge
                        difference = merge_dicts(difference, tempD)

    return difference

def discardDict(dt, valueFilter):
    """ discards key: value pairs if value is a list or tuple containing two strings that have the same regex output after applying 'valueFilter'

        Arguments:
           dt (dict): dict of key: values pairs. where value is a list or tuple containing two values. Can have nested dicts.
           valueFilter (str): regex string.
        Returns:
           dict: filtered dict
    """
    if isinstance(dt, dict):
        nested = {k:discardDict(dt[k],valueFilter) for k in list(dt.keys()) if discardDict(dt[k], valueFilter) is not None}
        return nested

    elif isinstance(dt, tuple):
        list_vals = ()
        for i in dt:
            if isinstance(i, str) and '<function' in i: # remove regex
                val = re.findall(valueFilter, i)[0]
            else:
                val = i
            list_vals = list_vals + (val,)

        if list_vals[0] != list_vals[1]:
            return dt


def merge_dicts(a, b, path=None):
    """ Recursive dict merge. Unlike dict.update(), instead of
        updating only top-level keys, this merges within nested dicts.

        Arguments:
           a (dict): dict to compare to
           b (dict): dict being compared
           path (list): list of nested dicts taken to reach key.
        Returns:
           dict: merged dict
    """

    if path is None: path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge_dicts(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass
            else:
                raise Exception('Issue at %s' % '.'.join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a

def nested_dd():
    # Creates a depault dic so that keys can be added without being first defined.

    # Used by compareDict.
    return defaultdict(nested_dd)

def nested_dd_to_dict(d):
    # converts dict created with nested_dd to dict.
    if isinstance(d, defaultdict):
        d = {k: nested_dd_to_dict(v) for k, v in list(d.items())}
    return d