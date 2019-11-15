import os
import logging
import logging.config
import yaml
import inspect

##############################################################################
# Tools to setup logger behavior
##############################################################################

def setup_logging(log_path = '', console_level= logging.INFO):
    """Setup logging configuration
    Default log path will save to current working directory
    """

    os.makedirs(log_path, exist_ok= True)
    add_verbose_level()

    #load logger config
    conf = get_logger_config()

    # set log file locations
    for i in conf['handlers']:
        if 'filename' in list(conf['handlers'][i].keys()):
            fn = conf['handlers'][i]['filename']
            conf['handlers'][i]['filename'] = os.path.join(log_path, fn)

    #disable run logger on startup
    if 'run_file_handler' in conf['root']['handlers']:
        conf['root']['handlers'].remove('run_file_handler')

    #config logger
    if conf:
        logging.config.dictConfig(conf)
        set_console_level(console_level)
    else:
        logging.basicConfig(level=console_level)

    log = logging.getLogger(__name__)
    log.debug('Logger initialized')

def enable_run_file_handler(file):
    """sets path run log file"""

    # Set up text logger and add it to logging instance
    conf = get_logger_config()
    # enable logger
    conf['root']['handlers'].append('run_file_handler')
    #set log output location
    try:
        conf['handlers']['run_file_handler']['filename'] = file
        logging.config.dictConfig(conf)
        log = logging.getLogger(__name__)
        log.debug('run log saving to ' + file)

    except:
        log = logging.getLogger(__name__)
        log.warning('Unable to set up run log or file not defined')

def get_logger_config(file = None):
    """Get config from file and return as dict"""

    if not file:
        # load default config file
        fn = os.path.split(__file__)
        path = os.path.abspath(fn[0])
        confFile = os.path.join(path, 'logger_default.conf')
    else:
        confFile = file

    if os.path.exists(confFile):
        with open(confFile, 'rt') as f:
            conf = yaml.safe_load(f.read())
        return conf
    else:
        return None


def add_verbose_level(LEVEL = 15):
    """adds a verbose level that can be invoked with logger.verbose"""

    logging.addLevelName(LEVEL, "VERBOSE")
    # function gets added to logging object
    def verbose(self, message, *args, **kws):
        self._log(LEVEL, message, args, **kws)

    logging.Logger.verbose = verbose

    return LEVEL

def set_console_level(level):
    """changes level of console handler"""

    if isinstance(level, str):
        if level == 'info':
            level = 20
        if level == 'verbose':
            level = 15
        if level == 'debug':
            level = 10
    elif isinstance(level, int):
        level = level
    log = logging.getLogger() # gets root logger
    log.handlers[0].setLevel(level) #console must be first handle
    log.debug('Logger level set to {}'.format(level))

    return level


##############################################################################
# Formatted logging output
##############################################################################

def log_parameters(head=True, **args):
    """Writes parameter settings in a formatted way

    Arguments:
        head (Bool): prefix of each line. If True will use the function that called it
        **args: the parameter values as key=value arguments

    Returns:
        str or None: a formated string with parameter info
    """
    if head:
        if isinstance(head, str):
            prefix = head
        else:
            prefix = inspect.stack()[1][3] # gets function that called log_parameters
    else:
        prefix = ''

    keys = list(args.keys())
    vals = list(args.values())
    parsize = max([len(x) for x in keys])

    s = [prefix + '| ' + keys[i].ljust(parsize) + ': ' + str(vals[i]) for i in range(len(keys))]
    log = logging.getLogger(__name__)
    for i in s:
        log.info(i)
