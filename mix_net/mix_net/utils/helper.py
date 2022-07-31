import json
import cProfile
import io
import pstats

import numpy as np


def get_param_dict(config_parser):
    """Write a config parser as a dict of dicts.

    Args:
        config_parser ([type]): [description]

    Returns:
        [type]: [description]
    """
    param_dict = {}
    for sec in config_parser.sections():
        for key, value in config_parser.items(sec):
            param_dict[sec] = json.loads(config_parser.get(sec, key))

    return param_dict


def fill_with_nans(array, desired_length):

    nan_mask = np.empty(desired_length) * np.nan
    nan_mask[: len(array)] = array

    return nan_mask


def profile(func):
    def inner(*args, **kwargs):
        pr = cProfile.Profile()
        pr.enable()
        retval = func(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = "cumulative"
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner
