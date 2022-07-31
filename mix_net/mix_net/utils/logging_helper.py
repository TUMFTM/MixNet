import logging
import json
import numpy as np
import time
import os
from pathlib import Path
from tqdm import tqdm

formatter = logging.Formatter("%(asctime)s %(message)s")


def setup_logger(name, log_file, level=logging.INFO):
    """To setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


class DataLogging:
    """
    Logging class that handles the setup and data-flow in order to write a log for the graph-planner in an iterative
    manner.
    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Alexander Heilmeier
    :Created on:
        23.01.2019
    """

    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(self, log_path: str) -> None:
        """"""
        # Create directories
        Path(os.path.dirname(log_path)).mkdir(parents=True, exist_ok=True)
        # write header to logging file
        self.__log_path = log_path
        with open(self.__log_path, "w+") as fh:
            header = (
                "time;obj_dict;hist_input;boundaries_input;pred_dict;" "calc_time_avg"
            )
            fh.write(header)

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def log_pred_data(
        self,
        time: float,
        obj_dict: dict,
        hist_input: dict,
        boundaries_input: dict,
        pred_dict: dict,
        calc_time_avg: float,
        log_params=None,
    ) -> None:
        """
        Write one line to the log file.
        :param time:             current time stamp (float time)
        """

        if log_params is not None:
            hist_input_local = {}
            boundaries_input_local = {}
            pred_dict_local = {}

            for ID in pred_dict.keys():
                if log_params["history"]:
                    hist_input_local[ID] = hist_input[ID]
                else:
                    hist_input_local[ID] = None

                if log_params["boundaries"]:
                    boundaries_input_local[ID] = boundaries_input[ID]
                else:
                    boundaries_input_local[ID] = None

                # num_covs = (
                #     log_params["num_covs"]
                #     if log_params["num_covs"] < pred_dict[ID]["cov"].shape[0]
                #     else pred_dict[ID]["cov"].shape[0]
                # )

                pred_dict_local[ID] = {
                    "vehicle_id": pred_dict[ID]["vehicle_id"],
                    "prediction_type": pred_dict[ID]["prediction_type"],
                    "t_abs_perception": pred_dict[ID]["t_abs_perception"],
                    "t": pred_dict[ID]["t"] if log_params["time_array"] else None,
                    "x": pred_dict[ID]["x"],
                    "y": pred_dict[ID]["y"],
                    # "cov": pred_dict[ID]["cov"][:num_covs, :, :],
                    "heading": pred_dict[ID]["heading"]
                    if log_params["heading"]
                    else None,
                }

            with open(self.__log_path, "a") as fh:
                fh.write(
                    "\n"
                    + str(time)
                    + ";"
                    + json.dumps(obj_dict, default=default)
                    + ";"
                    + json.dumps(hist_input_local, default=default)
                    + ";"
                    + json.dumps(boundaries_input_local, default=default)
                    + ";"
                    + json.dumps(pred_dict_local, default=default)
                    + ";"
                    + json.dumps(calc_time_avg, default=default)
                )
        else:
            with open(self.__log_path, "a") as fh:
                fh.write(
                    "\n"
                    + str(time)
                    + ";"
                    + json.dumps(obj_dict, default=default)
                    + ";"
                    + json.dumps(hist_input, default=default)
                    + ";"
                    + json.dumps(boundaries_input, default=default)
                    + ";"
                    + json.dumps(pred_dict, default=default)
                    + ";"
                    + json.dumps(calc_time_avg, default=default)
                )

    def log_tracking_data(
        self,
        time: float,
        detection_dict: dict,
        log_ego_state: list,
        match_dict: list,
        filter_log: dict,
        object_dict: dict,
        old_object_dict: dict,
        data_based_input: dict,
        pred_dict: dict,
        calc_time_avg: float,
    ) -> None:
        """
        Write one line to the log file.
        :param time:             current time stamp (float time)
        """
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time)
                + ";"
                + json.dumps(detection_dict, default=default)
                + ";"
                + json.dumps(log_ego_state, default=default)
                + ";"
                + json.dumps(match_dict, default=default)
                + ";"
                + json.dumps(filter_log, default=default)
                + ";"
                + json.dumps(object_dict, default=default)
                + ";"
                + json.dumps(old_object_dict, default=default)
                + ";"
                + json.dumps(data_based_input, default=default)
                + ";"
                + json.dumps(pred_dict, default=default)
                + ";"
                + json.dumps(calc_time_avg, default=default)
            )


class MessageLogging:
    """
    Logging class that handles the setup and data-flow in order to write a log for the graph-planner in an iterative
    manner.
    :Authors:
        * Tim Stahl <tim.stahl@tum.de>
        * Alexander Heilmeier
    :Created on:
        23.01.2019
    """

    # ----------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ----------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------
    def __init__(self, log_path: str) -> None:
        """"""
        # write header to logging file
        self.__log_path = log_path
        with open(self.__log_path, "w+") as fh:
            header = "time;type;message"
            fh.write(header)

    # ----------------------------------------------------------------------------------------------------------
    # CLASS METHODS --------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------

    def log_message(self, time: float, msg_type: str, message: str) -> None:
        """
        Write one line to the log file.
        :param time:             current time stamp (float time)
        """
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time)
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )

    def warning(self, message: str) -> None:
        msg_type = "WARNING"
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time.time())
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )

    def info(self, message: str) -> None:
        msg_type = "INFO"
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time.time())
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )

    def debug(self, message: str) -> None:
        msg_type = "DEBUG"
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time.time())
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )

    def error(self, message: str) -> None:
        msg_type = "error"
        with open(self.__log_path, "a") as fh:
            fh.write(
                "\n"
                + str(time.time())
                + ";"
                + json.dumps(msg_type, default=default)
                + ";"
                + json.dumps(message, default=default)
            )


def default(obj):
    # handle numpy arrays when converting to json
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError("Not serializable (type: " + str(type(obj)) + ")")


def read_all_data(file_path_in, keys=None, zip_horz=False):

    with open(file_path_in) as f:
        total_lines = sum(1 for _ in f)

    total_lines = max(1, total_lines)

    all_data = None

    assert (
        total_lines > 1
    ), "Invalid logs: No tracking files, most likely short simulation time"

    # extract a certain line number (based on time_stamp)

    with open(file_path_in) as file:
        # get to top of file (1st line)
        file.seek(0)
        # get header (":-1" in order to remove tailing newline character)
        header = file.readline()[:-1]
        # extract line
        line = ""
        for j in tqdm(range(total_lines - 1)):
            line = file.readline()

            if zip_horz:
                if all_data is None:
                    all_data = []
                    all_data = [header.split(";"), [None] * (total_lines - 1)]

                all_data[1][j] = tuple(json.loads(ll) for ll in line.split(";"))
            else:
                # parse the data objects we want to retrieve from that line
                data = dict(zip(header.split(";"), line.split(";")))
                if all_data is None:
                    if keys is None:
                        keys = data.keys()
                    all_data = {key: [0.0] * (total_lines - 1) for key in keys}
                for key in keys:
                    all_data[key][j] = json.loads(data[key])

    return all_data


def read_info_data(info_file_path):
    """Extracts the infos about the observations with each ID and stores them in
    a dictionary, which can be queried by the IDs.
    """

    info_dict = {}
    with open(info_file_path) as f:
        for line in f:
            if "Prediction-ID" in line:
                i0 = line.index("Prediction-ID") + len("Prediction-ID") + 1
                i1 = line.index(":", i0)
                ID = line[i0:i1]
                info_dict[ID] = []

                if "static" in line:
                    info_dict[ID].append("static")

                elif "physics-prediction" in line:
                    info_dict[ID].append("physics-based")

                    i0 = line.index("reason:") + len("reason:") + 1
                    i1 = line.index(",", i0)
                    info_dict[ID].append(line[i0:i1])

                elif "data-prediction" in line:
                    info_dict[ID].append("data-based")

                    if "mixers" in line:
                        i0 = line.index("mixers")
                        info_dict[ID].append("\n")
                        info_dict[ID].append(line[i0:])

                elif "data-physics-override-prediction" in line:
                    info_dict[ID].append("data-physics-override")

                    if "mixers" in line:
                        i0 = line.index("mixers")
                        info_dict[ID].append("\n")
                        info_dict[ID].append(line[i0:])

                elif "rail-prediction" in line:
                    info_dict[ID].append("rail-based")

                    i0 = line.index("reason:") + len("reason:") + 1
                    i1 = line.index(",", i0)
                    info_dict[ID].append(line[i0:i1])

                elif "potential-field" in line:
                    info_dict[ID].append("potential-field")

                    i0 = (
                        line.index("potential-field prediction")
                        + len("potential-field prediction")
                        + 2
                    )
                    i1 = line.index("id:")
                    info_dict[ID].append(line[i0:i1])

                elif "Invalid" in line:
                    info_dict[ID].append("invalid")

            elif "Collision" in line:
                i0 = line.index("IDs") + len("IDs") + 1
                i1 = line.index("(", i0) - 1
                ID1 = line[i0:i1]

                i0 = line.index("and") + len("and") + 1
                i1 = line.index("(", i0) - 1
                ID2 = line[i0:i1]

                i0 = line.index("timestep") + len("timestep") + 1
                ts = line[i0:-2]

                info_dict[ID1].append("collision with ID " + ID2 + " at " + ts)
                info_dict[ID2].append("collision with ID " + ID1 + " at " + ts)

            elif "not adjusted" in line:
                pass

            elif "adjusted" in line:
                if "ID" in line:
                    i0 = line.index("ID") + len("ID") + 1
                    i1 = line.index("adjusted", i0) - 1
                    ID = line[i0:i1]
                    if "right" in line:
                        direction = "right"
                    else:
                        direction = "left"

                    i0 = line.index("distance of") + len("distance of") + 1
                    dist = line[i0:-2]

                    info_dict[ID].append("adjusted to the " + direction + " by " + dist)

    return info_dict


def recover_trajectories(obj_data):
    """Recovers the trajectories from the log. It takes the list of obj_dicts, which
    were recovered from the data log file and creates x, y, t arrays for each vehicle
    based in them.

    args:
        obj_data: (list of dicts), the list of obj_dicts that was recovered from the log.
    returns:
        trajectories: (dict of dicts), The keys of the main dict are the vehicle IDs
            and the value of each vehicle ID is an other dict, which contains the x, y and time
            data.
    """

    trajectories = {}

    for object_dict in obj_data:

        if not bool(object_dict):
            continue
        for objID, xy_pos, t_abs in object_dict.values():
            # division by 1e9 is needed, because mod_object sends nanosecs:
            t = float(t_abs) / 1e9

            if objID not in trajectories.keys():
                trajectories[objID] = {"t_list": [], "x_list": [], "y_list": []}
                trajectories[objID]["t_list"].append(t)
                trajectories[objID]["x_list"].append(xy_pos[0])
                trajectories[objID]["y_list"].append(xy_pos[1])

            elif t != trajectories[objID]["t_list"][-1]:
                trajectories[objID]["t_list"].append(t)
                trajectories[objID]["x_list"].append(xy_pos[0])
                trajectories[objID]["y_list"].append(xy_pos[1])

    # numpify:
    for ID, val in trajectories.items():
        trajectories[ID]["t_list"] = np.array(val["t_list"])
        trajectories[ID]["x_list"] = np.array(val["x_list"])
        trajectories[ID]["y_list"] = np.array(val["y_list"])

    return trajectories


def recover_params(info_file):
    """recovers the parameters that were used. If the log file is old and
    hence no params are logged in it, some dummy params are created, so that
    the rest of the visualization will not complain.

    args:
        info_file: (string), the main log file.

    returns:
        dict, that contains the params that were used.
    """

    with open(info_file) as f:

        params = {}
        master_key = None
        started = False

        for line in f:
            if "=====" in line:
                if not started:
                    started = True
                else:
                    break
            elif started:
                if ":" in line:
                    i0 = line.index(" - ") + len(" - ")
                    i1 = line.index(":", i0)
                    key = line[i0:i1]

                    i0 = i1 + 2
                    i1 = line.index('"', i0)
                    val = line[i0:i1]

                    if master_key is not None:
                        params[master_key][key] = val
                    else:
                        print("Failed to recover params from log.\n")
                        return {}
                else:
                    i0 = line.index("INFO") + len("INFO") + 3
                    i1 = line.index('"', i0)
                    master_key = line[i0:i1]
                    params[master_key] = {}
    if "MODEL_PARAMS" not in list(params.keys()):
        # creating dummy params:
        params["MODEL_PARAMS"] = {}
        params["MODEL_PARAMS"]["sampling_frequency"] = "10"
        params["MODEL_PARAMS"]["data_min_obs_length"] = "1.0"
        params["MODEL_PARAMS"]["view"] = "400"
        params["MODEL_PARAMS"]["dist"] = "20"

        print("Could not recover params, hence dummy params are provided.")
    else:
        if "sampling_frequency" not in (list(params["MODEL_PARAMS"].keys())):
            params["MODEL_PARAMS"]["sampling_frequency"] = "10"
            print("using dummy param for MODEL_PARAMS/sampling_frequency")

        if "data_min_obs_length" not in (list(params["MODEL_PARAMS"].keys())):
            params["MODEL_PARAMS"]["data_min_obs_length"] = "1.0"
            print("using dummy param for MODEL_PARAMS/data_min_obs_length")

        if "view" not in (list(params["MODEL_PARAMS"].keys())):
            params["MODEL_PARAMS"]["view"] = "400"
            print("using dummy param for MODEL_PARAMS/view")

        if "dist" not in (list(params["MODEL_PARAMS"].keys())):
            params["MODEL_PARAMS"]["dist"] = "20"
            print("using dummy param for MODEL_PARAMS/dist")

    if "OBJ_HANDLING_PARAMS" not in list(params.keys()):
        params["OBJ_HANDLING_PARAMS"] = {}
        params["OBJ_HANDLING_PARAMS"]["max_obs_length"] = "30"
    else:
        if "max_obs_length" not in (list(params["OBJ_HANDLING_PARAMS"].keys())):
            params["OBJ_HANDLING_PARAMS"]["max_obs_length"] = "30"
            print("using dummy param for MODEL_PARAMS/max_obs_length")

    rpl_key = []
    for param_key in params.keys():
        if [param_key] == list(params[param_key].keys()):
            rpl_key.append(param_key)

    for param_key in rpl_key:
        params[param_key] = params[param_key][param_key]

    for param_key in params.keys():
        if "path" in param_key:
            repo_path = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            )
            params[param_key] = params[param_key].replace(
                "/dev_ws/src/mod_prediction", repo_path
            )
    return params


def get_data_from_line(file_path_in: str, line_num: int, log_type: str = "prediction"):
    line_num = max(1, line_num)
    # extract a certain line number (based on time_stamp)
    with open(file_path_in) as file:
        # get to top of file (1st line)
        file.seek(0)
        # get header (":-1" in order to remove tailing newline character)
        header = file.readline()[:-1]
        # extract line
        line = ""
        for _ in range(line_num):
            line = file.readline()

        # parse the data objects we want to retrieve from that line
        data = dict(zip(header.split(";"), line.split(";")))

        if log_type == "tracking":
            detection_dict = json.loads(data["detection_dict"])
            log_ego_state = json.loads(data["log_ego_state"])
            match_dict = json.loads(data["match_dict"])
            filter_log = json.loads(data["filter_log"])
            object_dict = json.loads(data["object_dict"])
            old_object_dict = json.loads(data["old_object_dict"])
            pred_input_dict = json.loads(data["pred_input_dict"])
            pred_dict = json.loads(data["pred_dict"])
            calc_time_avg = json.loads(data["calc_time_avg"])
            return (
                detection_dict,
                log_ego_state,
                match_dict,
                filter_log,
                object_dict,
                old_object_dict,
                pred_input_dict,
                pred_dict,
                calc_time_avg,
            )
        else:
            obj_dict = json.loads(data["obj_dict"])
            hist = json.loads(data["hist_input"])
            boundaries = json.loads(data["boundaries_input"])
            pred_dict = json.loads(data["pred_dict"])
            calc_time_avg = json.loads(data["calc_time_avg"])
            return obj_dict, hist, boundaries, pred_dict, calc_time_avg


def get_number_of_lines(file_path_in: str):
    with open(file_path_in) as file:
        row_count = sum(1 for row in file)

    return row_count


def log_param_dict(param_dict, logger):
    logger.info("=" * 40)
    for sec, dic in param_dict.items():
        logger.info(sec)
        if isinstance(dic, dict):
            for key, val in dic.items():
                logger.info(" - {}: {}".format(key, val))
        else:
            logger.info(" - {}: {}".format(sec, dic))
    logger.info("=" * 40)
