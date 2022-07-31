"""
This debug visualization tool takes (autoamically) the latest logfile
in the logs/ folder and visualizes the predictions.

    Arguments:
    --velocity: Plot the velocity profile.
    --video: Create a video instead of the interactive time slider
    --dataset: Create a dataset for training etc. out of the ground truth data
"""
import sys
import os
import argparse
import tqdm

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

import matplotlib
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})
plt.rcParams.update({"font.family": "Times New Roman"})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"figure.autolayout": True})

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
MIX_NET_COL = TUM_BLUE
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
INDY_NET_COL = TUM_ORAN
HIST_COL = "black"
HIST_LS = "solid"
GT_COL = "black"
GT_LS = "dashed"
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

from matplotlib.lines import Line2D

import numpy as np

from mix_net.mix_net.utils.logging_helper import (
    read_all_data,
    recover_params,
    recover_trajectories,
)
from mix_net.mix_net.utils.helper import fill_with_nans

from tools.file_utils import list_dirs_with_file

ERROR_ANALYSIS_TIME_STEPS = [9, 19, 29, 39, 49]
VELOCITY_BOUNDS = [30.0, 60.0, np.inf]
HIST_LEN_BOUNDS = [15, 25, np.inf]


class PredictionLogEvaluator:
    """Class for visualizing the logfiles of the prediction module"""

    def __init__(self, args):
        """Initialize a PredictionLogVisualizer object."""

        self._logdirs = self._list_logdirs(args.logdir)
        self._data_only = args.data_only
        self._save_path = args.save_path
        self._MAE_tot = args.MAE_tot

        if "mix" in args.logdir:
            self._col = TUM_BLUE
        else:
            self._col = TUM_ORAN

        if args.save_path:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
                print("Created dir {}".format(args.save_path))

    def _list_logdirs(self, logdir):
        """Loads the data from the logfiles. If logdir is specified, every log
        from that folder will be loaded. If it is not specified, the
        very last available log file is chosen.

        args:
            logdir: (str), the dir that contains the logfiles to load.
        """
        assert os.path.exists(logdir) or (
            logdir == ""
        ), 'The provided logdir "{}" does not exist.'.format(logdir)

        # searching for the files to open:
        if logdir != "":
            logdirs = list_dirs_with_file([logdir], "prediction_data.csv")

            assert logdirs, 'There were no data or main log files found in "{}"'.format(
                logdir
            )
        else:
            print("No logdir provided. Taking the last log.")

            logdirs = [self._get_latest_dir_that_contains("prediction_data")]

        print("loading logs from: {}".format(logdir))

        return logdirs

    def _get_latest_dir_that_contains(self, word):
        """Finds the latest file in the logs directory that contains
        the expression given in word.
        """

        # Finding the latest log file:
        dir_list = list_dirs_with_file(["logs"], word)
        dir_list.sort(reverse=True)

        try:
            return dir_list[0]
        except IndexError:
            raise FileNotFoundError("No log file found!")

    def __call__(self):
        """
        Create histograms with errors in longitude and latitude according to the desired timesteps.
        """

        # initializing the error_dict:
        error_dict = {
            "lat_errors": [],
            "long_errors": [],
            "hist_lens": [],
            "vels": [],
            "num_vehs": [],
            "calc_times": [],
        }

        for logdir in self._logdirs:
            print("Processing data from {}".format(logdir))

            all_log_data, recovered_params = self._load_data(logdir)

            error_dict = self._process_logfile(
                all_log_data, recovered_params, error_dict
            )

        print("Creating figures...\n")

        self._create_total_mae_plot(error_dict)
        self._create_error_vs_horizon_plot(error_dict)
        self._create_error_vs_velocity_plot(error_dict)
        self._create_error_vs_histlen_plot(error_dict)
        self._create_error_vs_vehnum_plot(error_dict)
        self._create_calc_time_vs_vehnum_plot(error_dict)

        print("Figures are done.")

        if not self._MAE_tot:
            plt.show()

    def _load_data(self, logdir):
        """Loads the data from the logdir provided.

        args:
            logdir: [str], the path to the dir that contains the logfiles.

        returns:
            all_log_data: [dict], the actual data from the logs.
            recovered_params: [dict], the params that were used recovered
                from the main logfile.
        """

        data_file_path = os.path.join(logdir, "prediction_data.csv")
        info_file_path = os.path.join(logdir, "prediction_main.csv")

        # read in data:
        _, all_log_data = read_all_data(data_file_path, zip_horz=True)
        self._no_lines_data = len(all_log_data)

        self._t_rel_0 = all_log_data[0][0]

        # Ground truth data:
        self._trajectories = recover_trajectories(
            [obj_dict for _, obj_dict, _, _, _, _ in all_log_data]
        )

        # read in info log:
        recovered_params = recover_params(info_file_path)

        return all_log_data, recovered_params

    def _get_gt_from_trajs(self, ID, t_abs, pred_len, recovered_params):
        """Creates the ground truth trajectory for a given prediction based
        on the recovered trajectories and parameters.

        args:
            ID: [str], The id of the car.
            t_abs: [float], The absolute time of the prediction, for which the history is needed.
            pred_len: [int] The length of the prediction in number of timesteps
            recovered_params: [dict], the parameters that were used
        return:
            The ground truth trajectory in np.array with shape (2, N)
        """

        freq = float(int(recovered_params["MODEL_PARAMS"]["sampling_frequency"]))
        t_list = self._trajectories[ID]["t_list"]

        if t_list.size == 1:
            return np.vstack(
                (self._trajectories[ID]["x_list"], self._trajectories[ID]["y_list"])
            )

        t0 = t_abs
        t1 = t_abs + (pred_len - 1) / freq

        if t_list[-1] < t1:
            t1 = t_list[-1]

        # looking for sections, where no data has been received for a long time and
        # adjusting the end of the time interval accordingly:
        dt_max = 0.5
        i0 = np.argmax(t_list[t_list <= t0])
        i1 = np.argmax(t_list[t_list < t1]) + 1

        if i1 >= t_list.shape[0]:
            i1 = t_list.shape[0] - 1

        for i in range(i0, i1):
            if (t_list[i + 1] - t_list[i]) > dt_max:
                i1 = i
                N = np.floor((t_list[i1] - t0) * freq)
                t1 = t0 + N / freq
                break

        return self._interp_traj(ID, t0, t1, np.floor((t1 - t0) * freq))

    def _interp_traj(self, ID, t0, t1, N):
        """Interpolates the recovered trajectory of the vehicle with ID,
        between the given time boundaries t0(inclusive) and t1(inclusive).
        It returns N evenly spaced coordinates between them.

        args:
            ID: The ID of the vehicle.
            t0: (float), the beginning timepoint of the interpolation (inclusive)
            t1: (float), the ending timepoint of the interpolation (inclusive)
            N: (float), timestep of the interpolated data.
        returns:
            The interpolated data in np.array with shape (2, N)
        """

        time_array = np.linspace(t0, t1, int(N))

        xx = np.interp(
            time_array,
            self._trajectories[ID]["t_list"],
            self._trajectories[ID]["x_list"],
        )

        yy = np.interp(
            time_array,
            self._trajectories[ID]["t_list"],
            self._trajectories[ID]["y_list"],
        )

        return np.vstack((xx, yy))

    def _get_mae(self, gt, pred):
        """Calculates the mae error between the gt and the prediction.

        args:
            gt: np.array((2, N)), the ground truth trajectory.
            pred: np.array((2, N)), the predicted trajectory.
        """

        abs_errors = self._get_absolute_error(gt, pred)
        return np.sqrt(np.sum(abs_errors**2) / abs_errors.shape[0])

    def _get_absolute_error(self, gt, pred):
        """
        Calculate absolute errors as distance.

        args:
            gt: np.array((2, N)), the ground truth trajectory.
            pred: np.array((2, N)), the predicted trajectory.
        """
        ind_min = min(gt.shape[1], pred.shape[1])
        abs_errors = np.linalg.norm(gt[:, :ind_min] - pred[:, :ind_min], axis=0)

        return abs_errors

    def _get_lat_long_error(self, gt, pred):
        """
        Calculate absolute errors in longitude and latitude.

        args:
            gt: [np.array with shape=((2, N))], the ground truth trajectory.
            pred: [np.array with shape=((2, N))], the predicted trajectory.

        returns:
            error_lat: [np.array with shape=(N,)], the errors in the lateral direction
            error_lat: [np.array with shape=(N,)], the errors in the longitudinal direction
        """
        ind_min = min(gt.shape[1], pred.shape[1])
        gt_cut = gt[:, :ind_min]
        pred_cut = pred[:, :ind_min]

        diff = gt_cut - pred_cut
        error = np.linalg.norm(diff, axis=0)

        if gt.shape[1] <= 1:
            return np.abs(diff[0]), np.abs(diff[1])

        gt_tangents = np.zeros_like(gt_cut)
        gt_tangents[:, :-1] = gt_cut[:, 1:] - gt_cut[:, :-1]
        gt_tangents[:, -1] = gt_tangents[:, -2]
        gt_tangents /= np.linalg.norm(gt_tangents, axis=0, keepdims=True)

        # the projection of the error vector on the tangents:
        error_long = np.abs(np.sum((gt_tangents * diff), axis=0))

        # the lateral error from the whole and the longitudinal errors:
        error_lat = np.sqrt(np.power(error, 2) - np.power(error_long, 2))

        return error_lat, error_long

    def _process_logfile(self, all_log_data, recovered_params, error_dict):
        """processes the logfiles and creates a dictionary that contains the errors
        and some other data which will be used during plotting.

        args:
            all_log_data: [dict], every data as it was loaded by the load_all_data function.
            recovered_params: [dict], the params that were used.
            error_dict: [dict], every key contains a list. The lists are filled up so, that
                indexing is easy. The values which belong together are placed to the same index.
                Keys:
                    lat_errors: [list of lists], contains 5 lists that contain the lateral errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    long_errors: [list of lists], contains 5 lists that contain the longitudinal errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    hist_lens: [list], contains the length of the history for every prediction.
                    vels: [list], contains the average velocity of the groundtruth for every prediction.
                    num_vehs: [list], contains the number of veicles for every prediction.
                    calc_times: [list], contains the calculation time that was needed for a prediction.

        returns:
            The error dict updated with new data from all_log_data.
        """

        for data_sample in tqdm.tqdm(all_log_data):

            (
                _,
                _,
                hist,
                _,
                pred_dict,
                calc_time,
            ) = data_sample

            for pred_id in pred_dict.keys():

                if self._data_only and (
                    pred_dict[pred_id]["prediction_type"] != "data"
                ):
                    continue

                # not considering invalid and static predictions:
                if (
                    pred_dict[pred_id]["prediction_type"] == "invalid"
                    or pred_dict[pred_id]["prediction_type"] == "static"
                ):
                    continue

                # The prediction:
                pred_x = pred_dict[pred_id]["x"]
                pred_y = pred_dict[pred_id]["y"]
                pred = np.vstack((pred_x, pred_y))
                vehicle_id = pred_dict[pred_id]["vehicle_id"]

                # division by 1e9 is needed, because mod_object sends nanosecs:
                t_abs = float(pred_dict[pred_id]["t_abs_perception"]) / 1e9

                # Get ground truth
                gt = self._get_gt_from_trajs(
                    vehicle_id, t_abs, pred.shape[1], recovered_params
                )

                # Only use time steps that are available
                analyze_time_steps = [
                    i for i in ERROR_ANALYSIS_TIME_STEPS if i < gt.shape[1]
                ]

                # Get errors
                lat_error, long_error = self._get_lat_long_error(gt, pred)

                # get the datapoints which interest us:
                lat_error = lat_error[analyze_time_steps]
                long_error = long_error[analyze_time_steps]

                # Fill with nans as they have different lenghts
                error_dict["lat_errors"].append(
                    fill_with_nans(lat_error, len(ERROR_ANALYSIS_TIME_STEPS))
                )
                error_dict["long_errors"].append(
                    fill_with_nans(long_error, len(ERROR_ANALYSIS_TIME_STEPS))
                )

                # history length:
                hist_pid = hist[pred_id]
                if hist_pid is not None:
                    error_dict["hist_lens"].append(len(hist_pid))
                else:
                    error_dict["hist_lens"].append(np.nan)

                # velocity:
                if gt.shape[1] > 1:
                    gt_diff = np.linalg.norm((gt[:, 1:] - gt[:, :-1]), axis=0)
                    error_dict["vels"].append(
                        np.mean(gt_diff)
                        * float(recovered_params["MODEL_PARAMS"]["sampling_frequency"])
                    )
                else:
                    error_dict["vels"].append(error_dict["vels"][-1])

                # number of vehicles:
                error_dict["num_vehs"].append(len(list(pred_dict.keys())))

                # calculation times:
                error_dict["calc_times"].append(
                    calc_time * 1000.0
                )  # converting it to milliseconds

        return error_dict

    def _create_total_mae_plot(self, error_dict):
        """Creates the plot that shows the total mae reached in these logs.

        args:
            error_dict: [dict], every key contains a list. The lists are filled up so, that
                indexing is easy. The values which belong together are placed to the same index.
                Keys:
                    lat_errors: [list of lists], contains 5 lists that contain the lateral errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    long_errors: [list of lists], contains 5 lists that contain the longitudinal errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    hist_lens: [list], contains the length of the history for every prediction.
                    vels: [list], contains the average velocity of the groundtruth for every prediction.
                    num_vehs: [list], contains the number of veicles for every prediction.
                    calc_times: [list], contains the calculation time that was needed for a prediction.
        """
        # MAE with L2-Norm
        total_errors = np.linalg.norm(
            np.array(
                list(
                    zip(
                        np.array(error_dict["lat_errors"]),
                        np.array(error_dict["long_errors"]),
                    )
                )
            ),
            axis=1,
        )

        mae = np.nanmean(total_errors, axis=1)

        # still there remain nan values, if all of the errors were nans in a timestep.
        # (happens at the end of logfiles, where the groundtruth is too short.)
        not_nan_mask = ~np.isnan(mae)

        overall_mae = np.nanmean(mae)

        self._create_boxplots(
            [mae[not_nan_mask]],
            title="$\mathrm{MAE}_{\mathrm{L2}}$ = "
            + "{:.3f}".format(overall_mae)
            + " m",
            ax_titles=[""],
            x_label=None,
            y_label="$\mathrm{MAE}$ in m",
            y_lims=[(0.0, 16.0)],
            file_save_name="total_mae",
        )

        # printing some info:
        print("-" * 10 + " Total MAE (L2-Norm) " + "-" * 10)
        print(
            "Number of datapoints used for evaluation: {}".format(
                mae[not_nan_mask].size
            )
        )
        print(r"Overall MAE (L2-Norm) = {:.3f} m".format(overall_mae))
        print(
            "Overall average velocity in the logs: {:.3f} m/s".format(
                np.mean(error_dict["vels"])
            )
        )
        print("")

    def _create_error_vs_horizon_plot(self, error_dict):
        """Creates the plot that shows the errors with respect to the prediction horizon len.

        args:
            error_dict: [dict], every key contains a list. The lists are filled up so, that
                indexing is easy. The values which belong together are placed to the same index.
                Keys:
                    lat_errors: [list of lists], contains 5 lists that contain the lateral errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    long_errors: [list of lists], contains 5 lists that contain the longitudinal errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    hist_lens: [list], contains the length of the history for every prediction.
                    vels: [list], contains the average velocity of the groundtruth for every prediction.
                    num_vehs: [list], contains the number of veicles for every prediction.
                    calc_times: [list], contains the calculation time that was needed for a prediction.
        """

        total_error_containers = []
        lat_error_containers = []
        long_error_containers = []

        # creating the datacontainers for the boxplot:
        for pred_ts in range(len(ERROR_ANALYSIS_TIME_STEPS)):
            lat_errors_np = np.array(error_dict["lat_errors"])
            long_errors_np = np.array(error_dict["long_errors"])

            not_nan_mask = ~np.isnan(lat_errors_np[:, pred_ts])

            # derive L2 norm per step
            mae_errors = np.sqrt(
                np.power(lat_errors_np[not_nan_mask, pred_ts], 2)
                + np.power(long_errors_np[not_nan_mask, pred_ts], 2)
            )

            total_error_containers.append(mae_errors)
            lat_error_containers.append(lat_errors_np[not_nan_mask, pred_ts])
            long_error_containers.append(long_errors_np[not_nan_mask, pred_ts])

        self._create_boxplots(
            [total_error_containers, lat_error_containers, long_error_containers],
            title="$\mathrm{MAE}_{\mathrm{L2}} \mathrm{\,\,on\,\,} t_{\mathrm{pred}}$",
            ax_titles=[
                "$\mathrm{MAE}_{\mathrm{tot}}$",
                "$\mathrm{MAE}_{\mathrm{lat}}$",
                "$\mathrm{MAE}_{\mathrm{lon}}$",
            ],
            x_label="$t_{\mathrm{pred}}$ in s",
            y_label="$\mathrm{MAE}$ in m",
            y_lims=[(0.0, 26.0), (0.0, 7.0), (0.0, 26.0)],
            file_save_name="mae_pred_horizon",
        )

        # printing some info:
        print("-" * 10 + " Error vs prediction length " + "-" * 10)
        for i, datapoints in enumerate(total_error_containers):
            print(
                "Number of datapoints with {} timesteps of ground truth: {}".format(
                    ERROR_ANALYSIS_TIME_STEPS[i], len(datapoints)
                )
            )
        print("")

    def _create_error_vs_velocity_plot(self, error_dict):
        """Creates the plot that shows the errors with respect to the average velocity
        of the ground truth.

        args:
            error_dict: [dict], every key contains a list. The lists are filled up so, that
                indexing is easy. The values which belong together are placed to the same index.
                Keys:
                    lat_errors: [list of lists], contains 5 lists that contain the lateral errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    long_errors: [list of lists], contains 5 lists that contain the longitudinal errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    hist_lens: [list], contains the length of the history for every prediction.
                    vels: [list], contains the average velocity of the groundtruth for every prediction.
                    num_vehs: [list], contains the number of veicles for every prediction.
                    calc_times: [list], contains the calculation time that was needed for a prediction.
        """

        lat_error_containers = [[] for _ in range(len(VELOCITY_BOUNDS))]
        long_error_containers = [[] for _ in range(len(VELOCITY_BOUNDS))]
        total_error_containers = [[] for _ in range(len(VELOCITY_BOUNDS))]
        vel_bound_array = np.array(VELOCITY_BOUNDS)

        # creating the containers for the boxplots:
        for i, vel in enumerate(error_dict["vels"]):
            # finding in which group the velocity belongs:
            group_idx = np.sum(vel_bound_array < vel)

            if np.all(np.isnan(error_dict["lat_errors"][i])):
                continue

            lat_error_containers[group_idx].append(
                np.nanmean(error_dict["lat_errors"][i])
            )

            long_error_containers[group_idx].append(
                np.nanmean(error_dict["long_errors"][i])
            )

            total_errors = np.sqrt(
                np.power(error_dict["lat_errors"][i], 2)
                + np.power(error_dict["long_errors"][i], 2)
            )

            total_error_containers[group_idx].append(np.nanmean(total_errors))

        in_containers = [total_error_containers]
        ax_titles = [None]
        title_str = None
        if not self._MAE_tot:
            in_containers.append(lat_error_containers)
            in_containers.append(long_error_containers)
            ax_titles = [
                "$\mathrm{MAE}_{\mathrm{tot}}$",
                "$\mathrm{MAE}_{\mathrm{lat}}$",
                "$\mathrm{MAE}_{\mathrm{lon}}$",
            ]
            title_str = (
                "$\mathrm{MAE}_{\mathrm{L2}} \mathrm{\,\,on\,\,} v_{\mathrm{mean}}$"
            )

        self._create_boxplots(
            in_containers,
            title=title_str,
            ax_titles=ax_titles,
            x_label="$v$ in m/s",
            y_label="$\mathrm{MAE}$ in m",
            x_tick_labels=["$<30$", "$30-60$", "$60<$"],
            y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
            file_save_name="mae_vs_velocity",
        )

        # printing some info:
        print("-" * 10 + " Error vs velocity " + "-" * 10)
        for i, datapoints in enumerate(total_error_containers):
            print(
                "Number of datapoints with max {} [m/s] velocity: {}".format(
                    VELOCITY_BOUNDS[i], len(datapoints)
                )
            )
        print("")

    def _create_error_vs_histlen_plot(self, error_dict):
        """Creates the plot that shows the errors with respect to the history length.

        args:
            error_dict: [dict], every key contains a list. The lists are filled up so, that
                indexing is easy. The values which belong together are placed to the same index.
                Keys:
                    lat_errors: [list of lists], contains 5 lists that contain the lateral errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    long_errors: [list of lists], contains 5 lists that contain the longitudinal errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    hist_lens: [list], contains the length of the history for every prediction.
                    vels: [list], contains the average velocity of the groundtruth for every prediction.
                    num_vehs: [list], contains the number of veicles for every prediction.
                    calc_times: [list], contains the calculation time that was needed for a prediction.
        """

        lat_error_containers = [[] for _ in range(len(HIST_LEN_BOUNDS))]
        long_error_containers = [[] for _ in range(len(HIST_LEN_BOUNDS))]
        total_error_containers = [[] for _ in range(len(HIST_LEN_BOUNDS))]
        bounds_array = np.array(HIST_LEN_BOUNDS)

        for i, hist_len in enumerate(error_dict["hist_lens"]):
            if np.isnan(hist_len):
                continue

            # finding in which group the history length belongs:
            group_idx = np.sum(bounds_array < hist_len)

            if np.all(np.isnan(error_dict["lat_errors"][i])):
                continue

            lat_error_containers[group_idx].append(
                np.nanmean(error_dict["lat_errors"][i])
            )

            long_error_containers[group_idx].append(
                np.nanmean(error_dict["long_errors"][i])
            )

            total_errors = np.sqrt(
                np.power(error_dict["lat_errors"][i], 2)
                + np.power(error_dict["long_errors"][i], 2)
            )

            total_error_containers[group_idx].append(np.nanmean(total_errors))

        self._create_boxplots(
            [total_error_containers, lat_error_containers, long_error_containers],
            title="$\mathrm{MAE}_{\mathrm{L2}} \mathrm{\,\,on\,\,} t_{\mathrm{hist}}$",
            ax_titles=[
                "$\mathrm{MAE}_{\mathrm{tot}}$",
                "$\mathrm{MAE}_{\mathrm{lat}}$",
                "$\mathrm{MAE}_{\mathrm{lon}}$",
            ],
            x_label="$t_{\mathrm{hist}}$ in s",
            y_label="$\mathrm{MAE}$ in m",
            x_tick_labels=["0-1.5", "1.5-2.5", "2.5-3.0"],
            file_save_name="mae_vs_t_hist",
        )

        # printing some info:
        print("-" * 10 + " Error vs history length " + "-" * 10)
        for i, datapoints in enumerate(total_error_containers):
            print(
                "Number of datapoints with max {} [s] history length: {}".format(
                    HIST_LEN_BOUNDS[i] / 10.0, len(datapoints)
                )
            )
        print("")

    def _create_error_vs_vehnum_plot(self, error_dict):
        """Creates the plot that shows the errors with respect to the number of vehicles.

        args:
            error_dict: [dict], every key contains a list. The lists are filled up so, that
                indexing is easy. The values which belong together are placed to the same index.
                Keys:
                    lat_errors: [list of lists], contains 5 lists that contain the lateral errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    long_errors: [list of lists], contains 5 lists that contain the longitudinal errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    hist_lens: [list], contains the length of the history for every prediction.
                    vels: [list], contains the average velocity of the groundtruth for every prediction.
                    num_vehs: [list], contains the number of veicles for every prediction.
                    calc_times: [list], contains the calculation time that was needed for a prediction.
        """

        lat_error_containers = []
        long_error_containers = []
        total_error_containers = []

        for i, num_vehs in enumerate(error_dict["num_vehs"]):
            if num_vehs == 0:
                continue

            # adding empty container lists if this number of vehicles has not
            # been encountered yet.
            while len(lat_error_containers) < num_vehs:
                lat_error_containers.append([])
                long_error_containers.append([])
                total_error_containers.append([])

            if np.all(np.isnan(error_dict["lat_errors"][i])):
                continue

            lat_error_containers[num_vehs - 1].append(
                np.nanmean(error_dict["lat_errors"][i])
            )

            long_error_containers[num_vehs - 1].append(
                np.nanmean(error_dict["long_errors"][i])
            )

            total_errors = np.sqrt(
                np.power(error_dict["lat_errors"][i], 2)
                + np.power(error_dict["long_errors"][i], 2)
            )

            total_error_containers[num_vehs - 1].append(np.nanmean(total_errors))

        self._create_boxplots(
            [total_error_containers, lat_error_containers, long_error_containers],
            title="$\mathrm{MAE}_{\mathrm{L2}} \mathrm{\,\,on\,\,} n_{\mathrm{veh}}$",
            ax_titles=[
                "$\mathrm{MAE}_{\mathrm{tot}}$",
                "$\mathrm{MAE}_{\mathrm{lat}}$",
                "$\mathrm{MAE}_{\mathrm{lon}}$",
            ],
            x_label="$n_{\mathrm{veh}}$",
            y_label="$\mathrm{MAE}$ in m",
            y_lims=[(0.0, 16.0), (0.0, 16.0), (0.0, 16.0)],
            file_save_name="mae_vs_n_veh",
        )

        # printing some info:
        print("-" * 10 + " Error vs number of vehicles " + "-" * 10)
        for i, datapoints in enumerate(total_error_containers):
            print(
                "Number of datapoints with {} vehicles: {}".format(
                    i + 1, len(datapoints)
                )
            )
        print("")

    def _create_calc_time_vs_vehnum_plot(self, error_dict):
        """Creates the plot that shows the calculation time with respect to the number of vehicles.

        args:
            error_dict: [dict], every key contains a list. The lists are filled up so, that
                indexing is easy. The values which belong together are placed to the same index.
                Keys:
                    lat_errors: [list of lists], contains 5 lists that contain the lateral errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    long_errors: [list of lists], contains 5 lists that contain the longitudinal errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    hist_lens: [list], contains the length of the history for every prediction.
                    vels: [list], contains the average velocity of the groundtruth for every prediction.
                    num_vehs: [list], contains the number of veicles for every prediction.
                    calc_times: [list], contains the calculation time that was needed for a prediction.
                    calc_times: [list], contains the calculation time that was needed for a prediction.
        """

        calc_time_containers = []

        for i, num_vehs in enumerate(error_dict["num_vehs"]):
            if num_vehs == 0:
                continue

            # adding empty container lists if this number of vehicles has not
            # been encountered yet.
            while len(calc_time_containers) < num_vehs:
                calc_time_containers.append([])

            if np.isnan(error_dict["calc_times"][i]):
                continue

            calc_time_containers[num_vehs - 1].append(error_dict["calc_times"][i])

        self._create_boxplots(
            [calc_time_containers],
            title="$t_{\mathrm{calc}}$ in ms",
            ax_titles=[""],
            x_label="$n_{\mathrm{veh}}$",
            y_label="$t_{\mathrm{calc}}$ in ms",
            y_lims=[(4.0, 22.0)],
            file_save_name="t_calc",
        )

        # printing some info:
        print("-" * 10 + " Calculation time vs number of vehicles " + "-" * 10)
        for i, datapoints in enumerate(calc_time_containers):
            print(
                "Number of datapoints with {} vehicles: {}".format(
                    i + 1, len(datapoints)
                )
            )
        print("")

    def _create_boxplots(
        self,
        data_containers: list,
        title: str,
        ax_titles: list,
        x_label: str,
        y_label: str,
        x_tick_labels: list = None,
        y_lims: list = None,
        file_save_name: str = None,
    ):
        """Creates a plot with boxplots.

        args:
            data_containers: [list], contains in each item the data for one of the subplots, which are boxplots.
            title: [str], the title of the whole figure.
            ax_titles: [list of strings], the titles of the small boxplots.
            x_label: [str], the label of the x axes. (The same for every boxplot)
            y_label: [str], the label of the y axes. (The same for every boxplot)
            x_tick_labels: [list], the ticks to set on the x axis for the boxplots, if provided.
            y_lims: [list of tuples] The y limits of the plots if provided. Each tuple is in (bottom, top) format.
            file_save_name: [str], set file_save_name if desired
        """

        matplotlib.rcParams.update({"font.size": 32})

        num_boxplots = len(data_containers)
        if self._MAE_tot:
            num_boxplots = 1
            title = None
            ax_titles = [None]

        fig, ax_list = plt.subplots(1, num_boxplots, figsize=(num_boxplots * 6, 14))

        if num_boxplots == 1:
            ax_list = [ax_list]

        fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

        legend_elements = [
            Line2D(
                [0], [0], color=self._col, linestyle="dashed", linewidth=3, label="Mean"
            ),
            Line2D(
                [0],
                [0],
                color=self._col,
                linestyle="solid",
                linewidth=3,
                label="Median",
            ),
        ]

        props = dict(linewidth=3, color="k")
        props_mean = dict(linewidth=3, color=self._col)
        props_medi = dict(linewidth=3, color=self._col)
        for i, ax in enumerate(ax_list):
            ax.boxplot(
                data_containers[i],
                showfliers=False,
                meanline=True,
                showmeans=True,
                boxprops=props,
                whiskerprops=props,
                capprops=props,
                medianprops=props_medi,
                meanprops=props_mean,
            )
            ax.set(title=ax_titles[i])
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            if not x_label:
                plt.xticks(color="w")
            if not y_label:
                plt.yticks(color="w")

            ax.grid(True)
            ax.legend(handles=legend_elements, prop={"size": 30})

            if x_tick_labels is not None:
                ax.set_xticklabels(x_tick_labels)

            if y_lims is not None:
                ax.set_ylim(bottom=y_lims[i][0], top=y_lims[i][1])

            if self._MAE_tot:
                break

        if self._save_path:
            if file_save_name is None:
                file_name = title.replace(" ", "_") + ".pdf"
            else:
                if self._MAE_tot:
                    file_name = "MAE_tot_" + file_save_name + ".pdf"
                else:
                    file_name = file_save_name + ".pdf"
            plt.savefig(os.path.join(self._save_path, file_name), format="pdf")


if __name__ == "__main__":
    # Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        default="",
        help="The logdir. If not provided, the latest log is used.",
    )
    parser.add_argument(
        "--MAE_tot",
        action="store_true",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="If true, only the databased predictions are considered",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="",
        help="If provided, the plots will be saved here in pdf format.",
    )

    args = parser.parse_args()

    # create visualizer object:
    evaluator = PredictionLogEvaluator(args)

    evaluator()
