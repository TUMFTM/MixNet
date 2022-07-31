"""Helper functions to setup Objects class."""
import os
import sys
import configparser
from shutil import copyfile
import datetime

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from utils.logging_helper import (
    DataLogging,
    MessageLogging,
)
from utils.helper import get_param_dict


def stamp2time(seconds: int, nanoseconds: int) -> int:
    return seconds * 1000000000 + nanoseconds


def setup_logger(path_dict):
    """Setup main, detection and tracking logger."""
    # Set up Prediction Logger
    if not os.path.exists(path_dict["abs_log_path"]):
        os.makedirs(path_dict["abs_log_path"])

    # copy version.txt file to log folder:
    version_path = os.path.join(path_dict["src_path"], "version.txt")
    if os.path.exists(version_path):
        logdir = os.path.join(path_dict["abs_log_path"], "version.txt")
        copyfile(version_path, logdir)

    return (
        MessageLogging(path_dict["main_log_path"]),
        DataLogging(path_dict["data_log_path"]),
    )


def get_params(path_dict):
    """Get all params."""
    # read prediction parameter files
    main_param = configparser.ConfigParser()
    if not main_param.read(path_dict["main_param_path"]):
        raise ValueError(
            "Broken path to main_params: {}".format(path_dict["main_param_path"])
        )

    # get prediction params as dict
    param_dict = get_param_dict(main_param)
    param_dict["track_path"] = os.path.join(
        path_dict["map_path"], param_dict["FILE_NAMES"]["map_file"]
    )

    return param_dict


def get_raceline_csv(
    params: dict,
    raceline: str,
    track_key: str,
) -> str:
    """Get csv-file name for chosen raceline.

    Possible raceline: default, inner, outer, center (LVMS only)

    Args:
        tracking_params (dict): Contains tracking parameter
        raceline (str): String specifying raceline
        track_key (str): Abbreviation for track ('LVMS', 'IMS', 'LO')

    Returns:
        str: csv-file name for chosen raceline.
    """
    track_file_str = "track_file_" + track_key
    if raceline == "default":
        return params[track_file_str]
    elif raceline == "center":
        return params[track_file_str + "_center"]
    elif raceline == "inner":
        return params[track_file_str + "_inner"]
    elif raceline == "outer":
        return params[track_file_str + "_outer"]

    return params[track_file_str]


def create_path_dict():
    """Create a dict of all required paths.

    Args:
        node_path (str): absolute path to module

    Returns:
        dict: dict with paths
    """

    node_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Global variables
    abs_log_path = os.path.join(
        node_path,
        "logs",
        datetime.date.today().strftime("%Y_%m_%d"),
        datetime.datetime.now().strftime("%H_%M_%S"),
    )

    path_dict = {
        "node_path": node_path,
        "src_path": os.path.abspath(os.path.join("src", "mod_prediction")),
        "main_param_path": os.path.join(
            node_path,
            "config",
            "main_params.ini",
        ),
        "abs_log_path": abs_log_path,
        "main_log_path": os.path.join(
            abs_log_path,
            "prediction_main.csv",
        ),
        "data_log_path": os.path.join(
            abs_log_path,
            "prediction_data.csv",
        ),
        "mix_net_path": os.path.join(node_path, "data", "inference_model", "mix_net"),
        "indy_net_path": os.path.join(node_path, "data", "inference_model", "indy_net"),
        "map_path": os.path.join(node_path, "data", "map"),
    }

    return path_dict
