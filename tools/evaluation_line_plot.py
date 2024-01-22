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

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_LIGHT_BLUE = (100 / 255, 160 / 255, 200 / 255)
TUM_LIGHTER_BLUE = (152 / 255, 198 / 255, 234 / 255)
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
TUM_BLACK = (0, 0, 0)
GRAY = (104 / 255, 104 / 255, 104 / 255)
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

COLUMNWIDTH = 3.5
PAGEWIDTH = COLUMNWIDTH * 1
LABELFONTSIZE = 8  # IEEE is 8

FIGSIZE = (PAGEWIDTH, 9 / 16 * PAGEWIDTH)

INDY_NET_COL = GRAY
RAIL_BASED_COL = TUM_ORAN
MIX_NET_COL = TUM_BLUE
HIST_COL = TUM_BLACK
HIST_LS = "solid"
GT_COL = TUM_BLACK
GT_LS = "dashed"
BOUND_COL = BOUND_COL

from matplotlib.lines import Line2D
import numpy as np

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from mix_net.mix_net.utils.logging_helper import (
    read_all_data,
    recover_params,
    recover_trajectories,
)
from mix_net.mix_net.src.boundary_generator import BoundaryGenerator
from mix_net.mix_net.utils.helper import fill_with_nans

from tools.file_utils import list_dirs_with_file
from tools.evaluation import PredictionLogEvaluator

ERROR_ANALYSIS_TIME_STEPS = list(range(50))
ERROR_INTERVALS_STEPS = [9, 19, 29, 39, 49]
VELOCITY_BOUNDS = [30.0, 60.0, np.inf]
HIST_LEN_BOUNDS = [15, 25, np.inf]
TRACK_PATH = "mix_net/mix_net/data/map/traj_ltpl_cl_IMS_GPS.csv"


def update_matplotlib(labelfontsize=None):
    """Update matplotlib font style."""
    if labelfontsize is None:
        labelfontsize = LABELFONTSIZE
    plt.rcParams.update(
        {
            "font.size": labelfontsize,
            "font.family": "Times New Roman",
            "text.usetex": True,
            "figure.autolayout": True,
            "xtick.labelsize": labelfontsize * 1.0,
            "ytick.labelsize": labelfontsize * 1.0,
        }
    )


class MultiPredictionLogEvaluator:
    """Class for visualizing the logfiles of the prediction module"""

    def __init__(self, args):
        """Initialize a PredictionLogVisualizer object."""

        self._mixnet_logdirs = self._list_logdirs(args.mixnet_logdir)
        self._indynet_logdirs = self._list_logdirs(args.indynet_logdir)
        self._data_only = args.data_only
        self._save_path = args.save_path

        self._benchmark_col = INDY_NET_COL
        self._rail_based_col = RAIL_BASED_COL
        self._mixnet_col = MIX_NET_COL

        args.logdir = args.mixnet_logdir
        args.RMSE_tot = True
        args.rail = False
        self._predictionlogevaluator = PredictionLogEvaluator(args)

        params = {
            "MODEL_PARAMS": {"dist": 20, "view": 400},
            "track_path": os.path.join(repo_path, TRACK_PATH),
        }
        self._boundary_generator = BoundaryGenerator(params=params)
        self._with_raceline = True

        self.model_visz_dict = {
            "BENCHMARK": {"col": INDY_NET_COL, "name": "BENCHMARK"},
            "RAIL": {"col": RAIL_BASED_COL, "name": "RAIL"},
            "MIXNET": {"col": MIX_NET_COL, "name": "MIXNET"},
        }

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
        Create a line plot with confidence intervals which illustrates the error distribution of
        the 2 networks on the prediction horizon.
        """

        # MixNet dataloading
        mixnet_error_dict = self._get_empty_error_dict()
        rail_error_dict = self._get_empty_error_dict()

        for logdir in self._mixnet_logdirs:
            print("Processing data from {}".format(logdir))

            all_log_data, recovered_params = self._load_data(logdir)

            mixnet_error_dict, rail_error_dict = self._process_logfile(
                all_log_data, recovered_params, mixnet_error_dict, rail_error_dict
            )

        # IndyNet dataloading
        indynet_error_dict = self._get_empty_error_dict()

        for logdir in self._indynet_logdirs:
            print("Processing data from {}".format(logdir))

            all_log_data, recovered_params = self._load_data(logdir)

            indynet_error_dict = self._process_logfile(
                all_log_data,
                recovered_params,
                indynet_error_dict,
            )

        # Plotting
        print("Creating figure...\n")

        # something is fucked up, and I only have 50 errors instead of 51 (together with timestep 0).
        # Since I can not figure out easily, I simply adjust the x axis, so that the plot goes from
        # 0.0 - 5.0 as it should be:
        self.x_array = np.linspace(0, 5, 50)

        # Plots along horizon
        self._create_errors_plot(
            mixnet_error_dict, indynet_error_dict, rail_error_dict, key="tot"
        )
        self._create_errors_plot(
            mixnet_error_dict, indynet_error_dict, rail_error_dict, key="lat"
        )
        self._create_errors_plot(
            mixnet_error_dict, indynet_error_dict, rail_error_dict, key="long"
        )

        # Plots along objects
        self._eval_vs_velocity(indynet_error_dict, rail_error_dict, mixnet_error_dict)
        self._eval_vs_pred_len(indynet_error_dict, rail_error_dict, mixnet_error_dict)
        self._eval_vs_n_obj(indynet_error_dict, rail_error_dict, mixnet_error_dict)
        self._eval_vs_hist_len(indynet_error_dict, rail_error_dict, mixnet_error_dict)
        self._eval_calc_time_vs_n_obj(
            indynet_error_dict, rail_error_dict, mixnet_error_dict
        )

        print("Figure is done.")

        plt.show()

    @staticmethod
    def _get_empty_error_dict():
        "return empty error dict"
        return {
            "tot_errors": [],
            "lat_errors": [],
            "long_errors": [],
            "hist_lens": [],
            "vels": [],
            "num_vehs": [],
            "calc_times": [],
        }

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

    def _get_errors(self, gt, pred):
        """
        Calculate absolute, longitude and latitude error

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
        error = np.linalg.norm(diff, axis=0)  # RMSE

        if gt.shape[1] <= 1:
            return error, np.abs(diff[0]), np.abs(diff[1])

        gt_tangents = np.zeros_like(gt_cut)
        gt_tangents[:, :-1] = gt_cut[:, 1:] - gt_cut[:, :-1]
        gt_tangents[:, -1] = gt_tangents[:, -2]
        gt_tangents /= np.linalg.norm(gt_tangents, axis=0, keepdims=True)

        # the projection of the error vector on the tangents:
        error_long = np.abs(np.sum((gt_tangents * diff), axis=0))

        # the lateral error from the whole and the longitudinal errors:
        error_lat = np.sqrt(np.power(error, 2) - np.power(error_long, 2))

        return error, error_lat, error_long

    def _process_logfile(
        self, all_log_data, recovered_params, error_dict, rail_error_dict=None
    ):
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
                state_dict,
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
                tot_error, lat_error, long_error = self._get_errors(gt, pred)

                # get the datapoints which interest us:
                lat_error = lat_error[analyze_time_steps]
                long_error = long_error[analyze_time_steps]

                # Fill with nans as they have different lenghts
                error_dict["tot_errors"].append(
                    fill_with_nans(tot_error, len(ERROR_ANALYSIS_TIME_STEPS))
                )
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

                if rail_error_dict is not None:
                    # Get rail-based prediction
                    _, translation, _ = state_dict[pred_id]
                    pred_rail = self._boundary_generator.get_rail_pred(
                        translation=translation,
                        pred=pred,
                        with_raceline=self._with_raceline,
                    )

                    # Get errors
                    tot_error_rail, lat_error_rail, long_error_rail = self._get_errors(
                        gt, pred_rail
                    )
                    # get the datapoints which interest us:
                    lat_error_rail = lat_error_rail[analyze_time_steps]
                    long_error_rail = long_error_rail[analyze_time_steps]

                    # Fill with nans as they have different lenghts
                    rail_error_dict["tot_errors"].append(
                        fill_with_nans(tot_error_rail, len(ERROR_ANALYSIS_TIME_STEPS))
                    )
                    rail_error_dict["lat_errors"].append(
                        fill_with_nans(lat_error_rail, len(ERROR_ANALYSIS_TIME_STEPS))
                    )
                    rail_error_dict["long_errors"].append(
                        fill_with_nans(long_error_rail, len(ERROR_ANALYSIS_TIME_STEPS))
                    )
                    # history length:
                    hist_pid = hist[pred_id]
                    if hist_pid is not None:
                        rail_error_dict["hist_lens"].append(len(hist_pid))
                    else:
                        rail_error_dict["hist_lens"].append(np.nan)
                    # velocity:
                    if gt.shape[1] > 1:
                        gt_diff = np.linalg.norm((gt[:, 1:] - gt[:, :-1]), axis=0)
                        rail_error_dict["vels"].append(
                            np.mean(gt_diff)
                            * float(
                                recovered_params["MODEL_PARAMS"]["sampling_frequency"]
                            )
                        )
                    else:
                        rail_error_dict["vels"].append(rail_error_dict["vels"][-1])

                    # number of vehicles:
                    rail_error_dict["num_vehs"].append(len(list(pred_dict.keys())))

                    # calculation times:
                    rail_error_dict["calc_times"].append(
                        calc_time * 1000.0
                    )  # converting it to milliseconds

        if rail_error_dict is None:
            return error_dict
        return error_dict, rail_error_dict

    def _create_errors_plot(
        self, mixnet_error_dict, indynet_error_dict, rail_error_dict, key="tot"
    ):
        """Creates the plot about the rmse errors on the prediction horizon.

        args:
            mixnet_error_dict: [dict], error data of MixNet. Every key contains a list. The lists are filled up so, that
                indexing is easy. The values which belong together are placed to the same index.
                Keys:
                    lat_errors: [list of lists], contains 5 lists that contain the lateral errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    long_errors: [list of lists], contains 5 lists that contain the longitudinal errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS

            indynet_error_dict: [dict], error data of IndyNet. Every key contains a list. The lists are filled up so, that
                indexing is easy. The values which belong together are placed to the same index.
                Keys:
                    lat_errors: [list of lists], contains 5 lists that contain the lateral errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
                    long_errors: [list of lists], contains 5 lists that contain the longitudinal errors for every
                        prediction at the timesteps specified in ERROR_ANALYSIS_TIME_STEPS
        """

        mixnet_errors = []
        indynet_errors = []
        rail_errors = []

        error_key = key + "_errors"
        # creating the datacontainers for the boxplot:
        for pred_ts in range(len(ERROR_ANALYSIS_TIME_STEPS)):
            mixnet_errors_np = np.array(mixnet_error_dict[error_key])
            indynet_errors_np = np.array(indynet_error_dict[error_key])
            rail_errors_np = np.array(rail_error_dict[error_key])

            mixnet_not_nan_mask = ~np.isnan(mixnet_errors_np[:, pred_ts])
            indynet_not_nan_mask = ~np.isnan(indynet_errors_np[:, pred_ts])
            rail_not_nan_mask = ~np.isnan(rail_errors_np[:, pred_ts])

            mixnet_errors.append(mixnet_errors_np[mixnet_not_nan_mask, pred_ts])
            indynet_errors.append(indynet_errors_np[indynet_not_nan_mask, pred_ts])
            rail_errors.append(rail_errors_np[rail_not_nan_mask, pred_ts])

        # calculating the quantiles of the data:
        mixnet_means = np.array([np.mean(data) for data in mixnet_errors])
        mixnet_medians = np.array(
            [np.quantile(timestep, 0.5) for timestep in mixnet_errors]
        )
        mixnet_Q1 = np.array(
            [np.quantile(timestep, 0.25) for timestep in mixnet_errors]
        )
        mixnet_Q3 = np.array(
            [np.quantile(timestep, 0.75) for timestep in mixnet_errors]
        )
        mixnet_quantile_data = [mixnet_means, mixnet_medians, mixnet_Q1, mixnet_Q3]

        indynet_means = np.array([np.mean(data) for data in indynet_errors])
        indynet_medians = np.array(
            [np.quantile(timestep, 0.5) for timestep in indynet_errors]
        )
        indynet_Q1 = np.array(
            [np.quantile(timestep, 0.25) for timestep in indynet_errors]
        )
        indynet_Q3 = np.array(
            [np.quantile(timestep, 0.75) for timestep in indynet_errors]
        )
        indynet_quantile_data = [indynet_means, indynet_medians, indynet_Q1, indynet_Q3]

        rail_means = np.array([np.mean(data) for data in rail_errors])
        rail_medians = np.array(
            [np.quantile(timestep, 0.5) for timestep in rail_errors]
        )
        rail_Q1 = np.array([np.quantile(timestep, 0.25) for timestep in rail_errors])
        rail_Q3 = np.array([np.quantile(timestep, 0.75) for timestep in rail_errors])
        rail_quantile_data = [rail_means, rail_medians, rail_Q1, rail_Q3]

        if self._with_raceline:
            addstr = key + "_rl"
        else:
            addstr = key
        self._create_plot_over_horizon(
            mixnet_means,
            indynet_means,
            title="RMSE_" + addstr + "_pred_horizon",
            x_label="$t_{\mathrm{pred}}$ in s",
            y_label="RMSE in m",
            rail_val=rail_means,
        )
        self._create_plot_over_horizon(
            mixnet_medians,
            indynet_medians,
            title="RMSE_" + addstr + "_pred_horizon_median",
            x_label="$t_{\mathrm{pred}}$ in s",
            y_label="RMSE in m",
            rail_val=rail_medians,
        )
        self._create_lineplots_with_interval(
            mixnet_quantile_data,
            indynet_quantile_data,
            title="RMSE_" + addstr + "_pred_horizon_quantile",
            x_label="$t_{\mathrm{pred}}$ in s",
            y_label="RMSE in m",
            y_lims=None,
            rail_data=rail_quantile_data,
        )

        # printing some info:
        print("-" * 10 + " " + key + " Errors on the Prediction Horizon " + "-" * 10)
        print("BENCHMARK (INDYET)")
        for i, datapoints in enumerate(indynet_errors):
            print(
                "Number of datapoints with {} timesteps of ground truth: {}".format(
                    ERROR_ANALYSIS_TIME_STEPS[i], len(datapoints)
                )
            )
        print("")

        print("RAIL")
        for i, datapoints in enumerate(rail_errors):
            print(
                "Number of datapoints with {} timesteps of ground truth: {}".format(
                    ERROR_ANALYSIS_TIME_STEPS[i], len(datapoints)
                )
            )
        print("")

        print("MIXNET")
        for i, datapoints in enumerate(mixnet_errors):
            print(
                "Number of datapoints with {} timesteps of ground truth: {}".format(
                    ERROR_ANALYSIS_TIME_STEPS[i], len(datapoints)
                )
            )

        print("")

    def _create_plot_over_horizon(
        self,
        mixnet_val: np.ndarray,
        indynet_val: np.ndarray,
        title: str,
        x_label: str,
        y_label: str,
        rail_val: np.ndarray = None,
    ):
        """Creates a plot with boxplots.

        args:
            mixnet_data: [list], Contains the mean, median, Q1 and Q3 values of MixNet for the analyzis timesteps.
                                 each entry is an np.array of size(num_of_timesteps,)
            indynet_data: [list], Contains the mean, median, Q1 and Q3 data of IndyNet for the analyzis timesteps.
                                 each entry is an np.array of size(num_of_timesteps,)
            title: [str], the title of the whole figure.
            x_label: [str], the label of the x axes. (The same for every boxplot)
            y_label: [str], the label of the y axes. (The same for every boxplot)
            x_tick_labels: [list], the ticks to set on the x axis for the boxplots, if provided.
            y_lims: [list of tuples] The y limits of the plots if provided. Each tuple is in (bottom, top) format.
        """
        _, ax = plt.subplots(nrows=1, ncols=1, figsize=FIGSIZE)

        ax.plot(
            self.x_array,
            indynet_val,
            color=self._benchmark_col,
            label="BENCHMARK",
        )
        if rail_val is not None:
            ax.plot(
                self.x_array,
                rail_val,
                color=self._rail_based_col,
                label="RAIL",
            )
            title += "_rail"

        ax.plot(
            self.x_array,
            mixnet_val,
            color=self._mixnet_col,
            label="MIXNET",
        )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        plt.tight_layout()

        ax.legend(fontsize=LABELFONTSIZE)

        ax.xaxis.set_ticks([k for k in range(6)])
        ax.set_xlim(0, 5)
        ax.set_ylim(0)
        ax.grid(True)

        if self._save_path:
            file_name = title.replace(" ", "_") + ".pdf"
            plt.savefig(os.path.join(self._save_path, file_name))

        plt.close()

    def _create_lineplots_with_interval(
        self,
        mixnet_data: list,
        indynet_data: list,
        title: str,
        x_label: str,
        y_label: str,
        x_tick_labels: list = None,
        y_lims: tuple = None,
        rail_data: list = None,
    ):
        """Creates a plot with boxplots.

        args:
            mixnet_data: [list], Contains the mean, median, Q1 and Q3 values of MixNet for the analyzis timesteps.
                                 each entry is an np.array of size(num_of_timesteps,)
            indynet_data: [list], Contains the mean, median, Q1 and Q3 data of IndyNet for the analyzis timesteps.
                                 each entry is an np.array of size(num_of_timesteps,)
            title: [str], the title of the whole figure.
            x_label: [str], the label of the x axes. (The same for every boxplot)
            y_label: [str], the label of the y axes. (The same for every boxplot)
            x_tick_labels: [list], the ticks to set on the x axis for the boxplots, if provided.
            y_lims: [list of tuples] The y limits of the plots if provided. Each tuple is in (bottom, top) format.
        """

        mixnet_means = mixnet_data[0]
        mixnet_medians = mixnet_data[1]
        mixnet_Q1 = mixnet_data[2]
        mixnet_Q3 = mixnet_data[3]

        indynet_means = indynet_data[0]
        indynet_medians = indynet_data[1]
        indynet_Q1 = indynet_data[2]
        indynet_Q3 = indynet_data[3]

        if rail_data is not None:
            rail_means = rail_data[0]
            rail_medians = rail_data[1]
            rail_Q1 = rail_data[2]
            rail_Q3 = rail_data[3]

        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.gca()

        # fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

        legend_elements = [
            Line2D(
                [0],
                [0],
                color=self._benchmark_col,
                linestyle="solid",
                label="BENCHMARK",
            ),
            Line2D(
                [0],
                [0],
                color=self._mixnet_col,
                linestyle="solid",
                label="MIXNET",
            ),
        ]

        # plotting with interval:
        ax.fill_between(
            self.x_array,
            indynet_Q1,
            indynet_Q3,
            alpha=0.5,
            facecolor=self._benchmark_col,
        )

        if rail_data is not None:
            ax.fill_between(
                self.x_array,
                rail_Q1,
                rail_Q3,
                alpha=0.5,
                facecolor=self._rail_based_col,
            )
            legend_elements.append(
                Line2D(
                    [0],
                    [0],
                    color=self._rail_based_col,
                    linestyle="solid",
                    label="RAIL",
                )
            )
            title += "_rail"

        ax.fill_between(
            self.x_array, mixnet_Q1, mixnet_Q3, alpha=0.5, facecolor=self._mixnet_col
        )

        ax.plot(
            self.x_array,
            indynet_means,
            color=self._benchmark_col,
            linestyle="dashed",
        )
        ax.plot(
            self.x_array,
            indynet_medians,
            color=self._benchmark_col,
            linestyle="solid",
        )
        if rail_data is not None:
            ax.plot(
                self.x_array,
                rail_means,
                color=self._rail_based_col,
                linestyle="dashed",
            )
            ax.plot(
                self.x_array,
                rail_medians,
                color=self._rail_based_col,
                linestyle="solid",
            )
        ax.plot(
            self.x_array,
            mixnet_means,
            color=self._mixnet_col,
            linestyle="dashed",
        )
        ax.plot(
            self.x_array,
            mixnet_medians,
            color=self._mixnet_col,
            linestyle="solid",
        )

        ax.set_xlabel(x_label, fontsize=LABELFONTSIZE)
        ax.set_ylabel(y_label, fontsize=LABELFONTSIZE)
        ax.legend(handles=legend_elements, loc="upper left")

        ax.xaxis.set_ticks([k for k in range(6)])
        ax.set_xlim(0, 5)
        ax.grid(True)

        if x_tick_labels is not None:
            ax.set_xticklabels(x_tick_labels)

        ax.set_xlim(left=0, right=5)
        if y_lims is not None:
            ax.set_ylim(bottom=y_lims[0], top=y_lims[1])
        else:
            ax.set_ylim(0)

        if self._save_path:
            file_name = title.replace(" ", "_") + ".pdf"
            plt.savefig(os.path.join(self._save_path, file_name))

    @staticmethod
    def get_labels(vel_bounds, no_int=False):
        """Get label for bin plots.

        3 bins: smaller than, between, greater than.
        """
        x_tick_labels = []
        for j in range(len(vel_bounds)):
            if j == 0:
                if no_int:
                    x_tick_labels.append("$<{}$".format(vel_bounds[j]))
                else:
                    x_tick_labels.append("$<{}$".format(int(vel_bounds[j])))

                continue

            if j == len(vel_bounds) - 1 and vel_bounds[j] == np.inf:
                if no_int:
                    x_tick_labels.append("${}<$".format(vel_bounds[j - 1]))
                else:
                    x_tick_labels.append("${}<$".format(int(vel_bounds[j - 1])))
                continue

            if no_int:
                x_tick_labels.append("${}-{}$".format(vel_bounds[j - 1], vel_bounds[j]))
            else:
                x_tick_labels.append(
                    "${}-{}$".format(int(vel_bounds[j - 1]), int(vel_bounds[j]))
                )

        return x_tick_labels

    @staticmethod
    def set_box_color(bp, color):
        """Set box colors.

        Args:
            bp (Boxplot): Matplotlib boxplot.
            color (string): color string.
        """
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)
        plt.setp(bp["means"], color=color)

    def _eval_vs_velocity(self, indynet_error_dict, rail_error_dict, mixnet_error_dict):
        indynet_containers = (
            self._predictionlogevaluator._create_error_vs_velocity_plot(
                error_dict=indynet_error_dict, no_plots=True
            )
        )
        rail_containers = self._predictionlogevaluator._create_error_vs_velocity_plot(
            error_dict=rail_error_dict, no_plots=True
        )
        mixnet_containers = self._predictionlogevaluator._create_error_vs_velocity_plot(
            error_dict=mixnet_error_dict, no_plots=True
        )

        idx_tuples = [0]  # only total erros
        in_containers = [
            [mixnet_containers[idx_t] for idx_t in idx_tuples],
            [rail_containers[idx_t] for idx_t in idx_tuples],
            [indynet_containers[idx_t] for idx_t in idx_tuples],
        ]

        x_label = "$v_{\mathrm{obj}}$ in m/s"
        y_label_list = [
            "$\mathrm{RMSE}$ in m",
            "$\mathrm{RMSE}_{\mathrm{lat}}$ in m",
            "$\mathrm{RMSE}_{\mathrm{lon}}$ in m",
        ]
        y_label_list = y_label_list[: len(in_containers[0])]

        x_tick_labels = self.get_labels(VELOCITY_BOUNDS)
        model_str_list = ["MIXNET", "RAIL", "BENCHMARK"]

        self._create_boxplots(
            data_container_list=in_containers,
            title=None,
            ax_titles=[None for _ in range(len(in_containers[0]))],
            x_label=x_label,
            y_label_list=y_label_list,
            file_save_name="RMSE_box_vs_velocity",
            x_tick_labels=x_tick_labels,
            sensor_str_list=model_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
        )

    def _eval_vs_pred_len(self, indynet_error_dict, rail_error_dict, mixnet_error_dict):
        indynet_containers = self._predictionlogevaluator._create_error_vs_horizon_plot(
            error_dict=indynet_error_dict, no_plots=True
        )
        rail_containers = self._predictionlogevaluator._create_error_vs_horizon_plot(
            error_dict=rail_error_dict, no_plots=True
        )
        mixnet_containers = self._predictionlogevaluator._create_error_vs_horizon_plot(
            error_dict=mixnet_error_dict, no_plots=True
        )

        idx_tuples = [0]  # only total erros
        in_containers = [
            [mixnet_containers[idx_t] for idx_t in idx_tuples],
            [rail_containers[idx_t] for idx_t in idx_tuples],
            [indynet_containers[idx_t] for idx_t in idx_tuples],
        ]

        x_label = "$t_{\mathrm{pred}}$ in s"
        y_label_list = [
            "$\mathrm{RMSE}$ in m",
            "$\mathrm{RMSE}_{\mathrm{lat}}$ in m",
            "$\mathrm{RMSE}_{\mathrm{lon}}$ in m",
        ]
        y_label_list = y_label_list[: len(in_containers[0])]
        x_tick_labels = self.get_labels(
            [str(j + 1) for j in range(len(ERROR_INTERVALS_STEPS))]
        )
        model_str_list = ["MIXNET", "RAIL", "BENCHMARK"]

        self._create_boxplots(
            title=None,
            data_container_list=in_containers,
            ax_titles=[None for _ in range(len(in_containers[0]))],
            x_label=x_label,
            y_label_list=y_label_list,
            file_save_name="RMSE_box_vs_pred_len",
            x_tick_labels=x_tick_labels,
            sensor_str_list=model_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
        )

    def _eval_vs_n_obj(self, indynet_error_dict, rail_error_dict, mixnet_error_dict):
        indynet_containers = self._predictionlogevaluator._create_error_vs_vehnum_plot(
            error_dict=indynet_error_dict, no_plots=True
        )
        rail_containers = self._predictionlogevaluator._create_error_vs_vehnum_plot(
            error_dict=rail_error_dict, no_plots=True
        )
        mixnet_containers = self._predictionlogevaluator._create_error_vs_vehnum_plot(
            error_dict=mixnet_error_dict, no_plots=True
        )

        idx_tuples = [0]  # only total erros
        in_containers = [
            [mixnet_containers[idx_t] for idx_t in idx_tuples],
            [rail_containers[idx_t] for idx_t in idx_tuples],
            [indynet_containers[idx_t] for idx_t in idx_tuples],
        ]

        x_label = "$n_{\mathrm{obj}}$"
        y_label_list = [
            "$\mathrm{RMSE}$ in m",
            "$\mathrm{RMSE}_{\mathrm{lat}}$ in m",
            "$\mathrm{RMSE}_{\mathrm{lon}}$ in m",
        ]
        y_label_list = y_label_list[: len(in_containers[0])]
        x_tick_labels = [str(j + 1) for j in range(len(indynet_containers[0]))]
        model_str_list = ["MIXNET", "RAIL", "BENCHMARK"]

        self._create_boxplots(
            title=None,
            data_container_list=in_containers,
            ax_titles=[None for _ in range(len(in_containers[0]))],
            x_label=x_label,
            y_label_list=y_label_list,
            file_save_name="RMSE_box_vs_n_obj",
            x_tick_labels=x_tick_labels,
            sensor_str_list=model_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
        )

    def _eval_vs_hist_len(self, indynet_error_dict, rail_error_dict, mixnet_error_dict):
        indynet_containers = self._predictionlogevaluator._create_error_vs_histlen_plot(
            error_dict=indynet_error_dict, no_plots=True
        )
        rail_containers = self._predictionlogevaluator._create_error_vs_histlen_plot(
            error_dict=rail_error_dict, no_plots=True
        )
        mixnet_containers = self._predictionlogevaluator._create_error_vs_histlen_plot(
            error_dict=mixnet_error_dict, no_plots=True
        )

        idx_tuples = [0]  # only total erros
        in_containers = [
            [mixnet_containers[idx_t] for idx_t in idx_tuples],
            [rail_containers[idx_t] for idx_t in idx_tuples],
            [indynet_containers[idx_t] for idx_t in idx_tuples],
        ]

        x_label = "$n_{\mathrm{hist}}$"
        y_label_list = [
            "$\mathrm{RMSE}$ in m",
            "$\mathrm{RMSE}_{\mathrm{lat}}$ in m",
            "$\mathrm{RMSE}_{\mathrm{lon}}$ in m",
        ]
        y_label_list = y_label_list[: len(in_containers[0])]
        x_tick_labels = self.get_labels(HIST_LEN_BOUNDS)
        model_str_list = ["MIXNET", "RAIL", "BENCHMARK"]

        self._create_boxplots(
            title=None,
            data_container_list=in_containers,
            ax_titles=[None for _ in range(len(in_containers[0]))],
            x_label=x_label,
            y_label_list=y_label_list,
            file_save_name="RMSE_box_vs_hist_len",
            x_tick_labels=x_tick_labels,
            sensor_str_list=model_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
        )

    def _eval_calc_time_vs_n_obj(
        self, indynet_error_dict, rail_error_dict, mixnet_error_dict
    ):
        indynet_containers = (
            self._predictionlogevaluator._create_calc_time_vs_vehnum_plot(
                error_dict=indynet_error_dict, no_plots=True
            )
        )
        mixnet_containers = (
            self._predictionlogevaluator._create_calc_time_vs_vehnum_plot(
                error_dict=mixnet_error_dict, no_plots=True
            )
        )

        in_containers = [
            [mixnet_containers],
            [indynet_containers],
        ]

        x_label = "$n_{\mathrm{obj}}$"
        y_label_list = [
            "$t_{\mathrm{calc}}$ in ms",
        ]
        x_tick_labels = [str(j + 1) for j in range(len(mixnet_containers))]
        model_str_list = ["MIXNET", "BENCHMARK"]

        self._create_boxplots(
            title=None,
            data_container_list=in_containers,
            ax_titles=[None for _ in range(len(in_containers[0]))],
            x_label=x_label,
            y_label_list=y_label_list,
            file_save_name="calc_time_vs_n_obj",
            x_tick_labels=x_tick_labels,
            sensor_str_list=model_str_list,
            # y_lims=[(0.0, 22.0), (0.0, 11.0), (0.0, 22.0)],
        )

    def _create_boxplots(
        self,
        data_container_list: list,
        title: str,
        ax_titles: list,
        x_label: str,
        y_label_list: list,
        file_save_name: str,
        x_tick_labels: list = None,
        y_lims: list = None,
        sensor_str_list: list = ["lidar_cluster"],
        w_tot: float = 0.6,
    ):
        """Create a plot with boxplots.

        args:
            data_containers: [list], contains in each item the data for one of the subplots, which are boxplots.
            title: [str], the title of the whole figure.
            ax_titles: [list of strings], the titles of the small boxplots.
            x_label: [str], the label of the x axes. (The same for every boxplot)
            y_label: [str], the label of the y axes. (The same for every boxplot)
            file_save_name: [str], set file_save_name
            x_tick_labels: [list], the ticks to set on the x axis for the boxplots, if provided.
            y_lims: [list of tuples] The y limits of the plots if provided. Each tuple is in (bottom, top) format.
        """
        num_models = len(data_container_list)
        num_boxplots = len(data_container_list[0])
        num_splits = len(data_container_list[0][0])

        fig, ax_list = plt.subplots(
            1, num_boxplots, figsize=(PAGEWIDTH, PAGEWIDTH / (num_boxplots * 6) * 10)
        )

        if num_boxplots == 1:
            ax_list = [ax_list]

        # fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

        props = dict(linewidth=1.5, color="k")
        colcols = [
            self.model_visz_dict[sensor_str]["col"] for sensor_str in sensor_str_list
        ]

        def get_lims(idx=0, scale=1.1, sym=True):
            _max = -np.inf
            _min = np.inf
            for k in range(num_models):
                for j in range(num_splits):
                    if not data_container_list[k][idx][j]:
                        continue

                    q25 = np.quantile(data_container_list[k][idx][j], 0.25)
                    q75 = np.quantile(data_container_list[k][idx][j], 0.75)
                    w_low = q25 - 1.5 * (q75 - q25)
                    w_up = q75 + 1.5 * (q75 - q25)

                    _min = np.min([_min, w_low])
                    _max = np.max([_max, w_up])

            _min *= scale
            _max *= scale

            if sym:
                abs_max = np.max([np.abs(_min), np.abs(_max)])
                return (-abs_max, abs_max)
            return (_min, _max)

        def get_same_lims(x_lims, y_lims):
            if x_lims[1] > y_lims[1]:
                return x_lims

            return y_lims

        x_res_lims = get_lims(idx=0)
        if num_boxplots > 1:
            y_res_lims = get_lims(idx=1)
            x_res_lims = get_same_lims(x_res_lims, y_res_lims)

        for i, ax in enumerate(ax_list):
            idx_list = []
            bp_pair = [
                [dd[i][k] for dd in data_container_list] for k in range(num_splits)
            ]

            # plot per split
            split_idx = []
            for jx in range(num_splits):
                bp_active_idx = [j for j, bb in enumerate(bp_pair[jx]) if bool(bb)]
                active_tuples = len(bp_active_idx)
                if active_tuples > 1:
                    pp = [
                        jx + 1 - w_tot * 0.8 * (jj / (num_models - 1) - 0.5)
                        for jj in range(active_tuples)
                    ]
                else:
                    pp = [jx + 1]

                idx_list += bp_active_idx
                if bp_active_idx:
                    split_idx.append(jx + 1)

                for j, idx in enumerate(bp_active_idx):
                    bp = bp_pair[jx][idx]
                    if idx < len(colcols):
                        _col = colcols[idx]
                    else:
                        _col = "k"
                    ppx = pp[j]
                    boxplt = ax.boxplot(
                        bp,
                        showfliers=False,
                        meanline=True,
                        showmeans=False,
                        boxprops=props,
                        whiskerprops=props,
                        capprops=props,
                        medianprops=props,
                        meanprops=props,
                        positions=[ppx],
                        widths=w_tot / num_models,
                    )
                    self.set_box_color(boxplt, _col)
                    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            if ax_titles[i]:
                ax.set(title=ax_titles[i])

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label_list[i])

            if not x_label:
                plt.xticks(color="w")
            if not y_label_list[i]:
                plt.yticks(color="w")

            ax.grid(True)

            if x_tick_labels is not None:
                ax.set_xticks(split_idx)
                ax.set_xticklabels([x_tick_labels[xt - 1] for xt in split_idx])

            if y_lims is not None:
                ax.set_ylim(bottom=0, top=y_lims[i][1])
            elif i < 2:
                ax.set_ylim(bottom=0, top=x_res_lims[1])
            else:
                _lims = get_lims(idx=i, scale=1.2, sym=True)
                ax.set_ylim(bottom=0, top=_lims[1])

        if self._save_path:
            file_name = file_save_name + ".pdf"
            plt.savefig(os.path.join(self._save_path, file_name))

        plt.close()


if __name__ == "__main__":
    # Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mixnet_logdir",
        type=str,
        default="data/evaluation_data/1_mixnet",
        help="The logdir of MixNet. Has to be provided!",
    )
    parser.add_argument(
        "--indynet_logdir",
        type=str,
        default="data/evaluation_data/2_indynet",
        help="The logdir of IndyNet. Has to be provided!",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="If true, only the databased predictions are considered",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="data/evaluation_data/line_plots",
        help="If provided, the plots will be saved here in pdf format.",
    )

    args = parser.parse_args()

    update_matplotlib()

    # create visualizer object:
    evaluator = MultiPredictionLogEvaluator(args)

    evaluator()
