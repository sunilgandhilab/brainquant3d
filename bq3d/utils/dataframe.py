# -*- coding: utf-8 -*-
import pandas as pd
import logging
log = logging.getLogger(__name__)


def cleanup_df(df):
    """downcasts numerical types and converts objects to categories"""
    init_mem = mem_usage(df)
    for c in df:
        if str(df[c].dtype).startswith('int'):
            df[c] = pd.to_numeric(df[c], downcast = 'unsigned')
        if str(df[c].dtype).startswith('float'):
            df[c] = pd.to_numeric(df[c], downcast = 'float')
        if str(df[c].dtype).startswith('object'):
            df[c] = df[c].astype('category')
    fin_mem = mem_usage(df)
    print(f'Reduced size from {init_mem} to {fin_mem}')
    return df

def mem_usage(df):
    """prints memory usage"""
    if isinstance(df,pd.DataFrame):
        usage_b = df.memory_usage(deep=True).sum()
        usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
        return '{:03.2f} MB'.format(usage_mb)
    else:
        return None