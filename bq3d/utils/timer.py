"""
tools for timing functions
"""


import time
import logging
import inspect
log = logging.getLogger(__name__)

class Timer(object):
    """Class to stop time and print results in formatted way
    
    Attributes:
        time (float): the time since the timer was started
    """
    def __init__(self):
        self.start()

    def start(self):
        """Start the timer"""
        self.time = time.time()

    def reset(self):
        """Reset the timer"""
        self.time = time.time()

    def elapsed(self, prefix = None, asstring = True):
        """Calculate elapsed time and return as formated string
        
        Arguments:
            prefix (str or None): prefix to the string
            asstring (bool): return as string or float
        
        Returns:
            str or float: elapsed time
        """
        
        t = time.time()

        if asstring:
            t = self.format_elapsed(t - self.time)
            if prefix:
                return prefix + "| elapsed time: " + t
            else:
                return "Elapsed time: " + t
        else:
            return t - self.time

    def log_elapsed(self, prefix = True):
        """Print elapsed time as formated string
        
        Arguments:
            prefix (str or None): prefix to the string
        """
        if prefix:
            if isinstance(prefix, str):
                head = prefix
            else:
                head = inspect.stack()[1][3] # gets function that called log_parameters
        else:
            head = ''

        log.info(self.elapsed(prefix = head))

    def format_elapsed(self, t):
        """Format time to string
        
        Arguments:
            t (float): time in seconds prefix
        
        Returns:
            str: time as hours:minutes:seconds
        """
        m, s = divmod(t, 60)
        h, m = divmod(m, 60)

        return "%d:%02d:%02d" % (h, m, s)

    def get_time(self, format= '%Y%m%d%H%M%S'):
        """return time as formatted to string

        Arguments:
            format (str): formatted time as string

        Returns:
            str: time
        """
        return time.strftime(format)