# -*- coding: utf-8 -*-

from clearmap3 import io
from clearmap3.image_filters import filter_manager

def filter_image(filter, input, output = None, **kwargs):
    """ Passes an image through the specified image filter.

    Arguments:
        filter (str): filter to use. string should match filter class name.
        source (str or array): Image to filter
        kwargs (dict): additional arguments to pass to the filter as 'argument': value. These will be parsed into
            the filters attributes. key string must match a filters attribute name.
    Returns:
        (array): filtered image
    """

    input = io.readData(input)
    kwargs['input'] = input
    kwargs['output'] = input
    im_filter = set_filter(filter, kwargs)
    return im_filter.run()

def set_filter(filter, kwargs):
    """ Instantiates an image filter and passed to it the specified input arguments.

    Arguments:
        filter (str): filter to use. string should match filter class name.
        kwargs (dict): additional arguments to pass to the filter as 'argument': value. These will be parsed into
            the filters attributes. key string must match a filters attribute name.
    Returns:
        (Filter object): filter instance with inputs set.
    """
    im_filter = filter_manager.get_filter(filter)
    im_filter.set_inputs(kwargs)
    return im_filter