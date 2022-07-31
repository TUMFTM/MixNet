"""helper functions to create dataset."""
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split


def load_indy_net_data(path, use_every_nth=1.0):
    """Loads all the data from the provided path.

    !!!
    It loads everything under a given directory except for files which's name
    contains "info"
    !!!

    args:
        path: (str), A single datafile or a directory under which the data can be found.
            If its a dir, every subdir will be searched through for datafiles. If its a dir,
            it can only contain the data files or an "info.txt" file which is going to be ignored.
        use_every_nth: (float, default=1.0), only every n^th datapoint is going to be used, if provided
            and is bigger then 1.0.

    returns:
        data: (dict), all of the data with the keys:
            "id": (list) the IDs
            "hist": (np.array of shape=(num_data, hist_len, 2)) the history trajectories.
            "fut": (np.array of shape=(num_data, fut_len, 2)) the groundtruth future trajectories.
            "left_bd": (np.array of shape=(num_data, bound_len, 2)) left track boundary snippet.
            "right_bd": (np.array of shape=(num_data, bound_len, 2)) right track boundary snippet.
    """
    data = {}

    if not os.path.isabs(path):
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path)

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                ff = os.path.join(root, file)

                if "info" in file:
                    continue

                with open(ff, "rb") as fp:
                    temp_data = pickle.load(fp)
                    for key, items in temp_data.items():
                        if key not in data.keys():
                            data[key] = []
                        data[key] += items

    else:  # single file input
        with open(path, "rb") as fp:
            data = pickle.load(fp)

    data["id"] = list(range(len(data["hist"])))

    # numpify the data except for "id":
    for key in ["hist", "fut", "left_bd", "right_bd"]:
        data[key] = np.array(data[key])

    # taking only every n^th datapoint:
    if use_every_nth > 1.0:
        N = int(len(data["id"]) / use_every_nth)

        # using random generator with seed for reproducible results:
        rng = np.random.default_rng(0)
        idxs = rng.choice(data["id"], size=(N,), replace=False)

        for key in ["hist", "fut", "left_bd", "right_bd"]:
            data[key] = data[key][idxs]

        data["id"] = list(range(data["hist"].shape[0]))

    return data


def split_indy_net_data(data, train_size=0.8, val_size=0.1, test_size=0.1):
    """Splits the given data into train, validation and test sets.

    args:
        data: (dict), all of the data with the keys:
            "id": (list) the IDs
            "hist": (np.array of shape=(num_data, hist_len, 2)) the history trajectories.
            "fut": (np.array of shape=(num_data, fut_len, 2)) the groundtruth future trajectories.
            "left_bd": (np.array of shape=(num_data, bound_len, 2)) left track boundary snippet.
            "right_bd": (np.array of shape=(num_data, bound_len, 2)) right track boundary snippet.
        train_size: (float ]0.0, 1.0[), the ratio of the train data split
        val_size: (float ]0.0, 1.0[), the ratio of the validation data split
        test_size: (float ]0.0, 1.0[), the ratio of the test data split

    returns:
        train_data: (dict) the splitted training set with the same keys as data
        val_data: (dict) the splitted training set with the same keys as data
        test_data: (dict) the splitted training set with the same keys as data
    """

    # train - (val + test) split:
    (
        train_id,
        val_id,
        train_hist,
        val_hist,
        train_fut,
        val_fut,
        train_left_bd,
        val_left_bd,
        train_right_bd,
        val_right_bd,
    ) = train_test_split(
        data["id"],
        data["hist"],
        data["fut"],
        data["left_bd"],
        data["right_bd"],
        train_size=train_size,
    )

    # val - test split:
    (
        val_id,
        test_id,
        val_hist,
        test_hist,
        val_fut,
        test_fut,
        val_left_bd,
        test_left_bd,
        val_right_bd,
        test_right_bd,
    ) = train_test_split(
        val_id,
        val_hist,
        val_fut,
        val_left_bd,
        val_right_bd,
        train_size=(val_size / (val_size + test_size)),
    )

    # constructing the dicts:
    train_data = {
        "id": train_id,
        "hist": train_hist,
        "fut": train_fut,
        "left_bd": train_left_bd,
        "right_bd": train_right_bd,
    }

    val_data = {
        "id": val_id,
        "hist": val_hist,
        "fut": val_fut,
        "left_bd": val_left_bd,
        "right_bd": val_right_bd,
    }

    test_data = {
        "id": test_id,
        "hist": test_hist,
        "fut": test_fut,
        "left_bd": test_left_bd,
        "right_bd": test_right_bd,
    }

    return train_data, val_data, test_data


def load_mix_net_data(path, use_every_nth=1.0):
    """Loads all the data from the provided path.

    args:
        path: (str), A single datafile or a directory under which the data can be found.
            If its a dir, every subdir will be searched through for datafiles. If its a dir,
            it can only contain the data files or an "info.txt" file which is going to be ignored.
        use_every_nth: (float, default=1.0), only every n^th datapoint is going to be used, if provided
        and is bigger then 1.0.


    returns:
        data: (dict), all of the data with the keys:
            "hist": (list of 2D lists) the history trajectories.
            "fut": (list of 2D lists) the groundtruth future trajectories.
            "fut_inds": (list of lists) the indices of the nearest centerline points
                corresponding to the ground truth prediction.
            "left_bd": (list of 2D lists) left track boundary snippet.
            "right_bd": (list of 2D lists) right track boundary snippet.
    """
    data = {}

    if not os.path.isabs(path):
        os.path.join(os.path.dirname(os.path.dirname(__file__)), path)

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for file in files:
                ff = os.path.join(root, file)

                if "info" in file:
                    continue

                with open(ff, "rb") as fp:
                    temp_data = pickle.load(fp)
                    for key, items in temp_data.items():
                        if key not in data.keys():
                            data[key] = []
                        data[key] += items

    else:  # single file input
        with open(path, "rb") as fp:
            data = pickle.load(fp)

    if "fut_inds" not in data:
        raise IndexError("invalid data set for MixNet-Trainer")

    # numpify the data:
    for key, value in data.items():
        if key == "fut_inds":
            data[key] = np.array(value)
        else:
            data[key] = np.array(value, dtype=(np.float32))

    # taking only every n^th datapoint:
    if use_every_nth > 1.0:
        N = int(data["hist"].shape[0] / use_every_nth)

        # using random generator with seed for reproducible results:
        rng = np.random.default_rng(0)
        idxs = rng.choice(list(range(data["hist"].shape[0])), size=(N,), replace=False)

        for key in ["hist", "fut", "fut_inds", "left_bd", "right_bd"]:
            data[key] = data[key][idxs]

        data["id"] = list(range(data["hist"].shape[0]))

    return data
