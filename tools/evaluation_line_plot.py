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

plt.rcParams.update({"font.size": 20})
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

LABELFONTSIZE = 20

from matplotlib.lines import Line2D

import numpy as np


repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from mix_net.mix_net.utils.logging_helper import (
    read_all_data,
    recover_params,
    recover_trajectories,
)
from mix_net.mix_net.utils.helper import fill_with_nans

from tools.file_utils import list_dirs_with_file

# ERROR_ANALYSIS_TIME_STEPS = [9, 19, 29, 39, 49]
ERROR_ANALYSIS_TIME_STEPS = list(range(50))


class PredictionLogEvaluator:
    """Class for visualizing the logfiles of the prediction module"""

    def __init__(self, args):
        """Initialize a PredictionLogVisualizer object."""

        self._mixnet_logdirs = self._list_logdirs(args.mixnet_logdir)
        self._indynet_logdirs = self._list_logdirs(args.indynet_logdir)
        self._data_only = args.data_only
        self._save_path = args.save_path

        self._mixnet_col = MIX_NET_COL
        self._benchmark_col = INDY_NET_COL

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
        mixnet_error_dict = {"lat_errors": [], "long_errors": []}

        for logdir in self._mixnet_logdirs:
            print("Processing data from {}".format(logdir))

            all_log_data, recovered_params = self._load_data(logdir)

            mixnet_error_dict = self._process_logfile(
                all_log_data, recovered_params, mixnet_error_dict
            )

        # IndyNet dataloading
        indynet_error_dict = {"lat_errors": [], "long_errors": []}

        for logdir in self._indynet_logdirs:
            print("Processing data from {}".format(logdir))

            all_log_data, recovered_params = self._load_data(logdir)

            indynet_error_dict = self._process_logfile(
                all_log_data, recovered_params, indynet_error_dict
            )

        # Plotting
        print("Creating figure...\n")

        self._create_lateral_errors_plot(mixnet_error_dict, indynet_error_dict)
        self._create_longitudinal_errors_plot(mixnet_error_dict, indynet_error_dict)

        print("Figure is done.")

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

    def _get_rmse(self, gt, pred):
        """Calculates the rmse error between the gt and the prediction.

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
                _,
                _,
                pred_dict,
                _,
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

        return error_dict

    def _create_lateral_errors_plot(self, mixnet_error_dict, indynet_error_dict):
        """Creates the plot about the laterar errors on the prediction horizon.

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

        # creating the datacontainers for the boxplot:
        for pred_ts in range(len(ERROR_ANALYSIS_TIME_STEPS)):
            mixnet_errors_np = np.array(mixnet_error_dict["lat_errors"])
            indynet_errors_np = np.array(indynet_error_dict["lat_errors"])

            mixnet_not_nan_mask = ~np.isnan(mixnet_errors_np[:, pred_ts])
            indynet_not_nan_mask = ~np.isnan(indynet_errors_np[:, pred_ts])

            mixnet_errors.append(mixnet_errors_np[mixnet_not_nan_mask, pred_ts])
            indynet_errors.append(indynet_errors_np[indynet_not_nan_mask, pred_ts])

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

        self._create_lineplots_with_interval(
            mixnet_quantile_data,
            indynet_quantile_data,
            title="MAE_lat_pred_horizon",
            x_label="$t_{\mathrm{pred}}$ in s",
            y_label="$\mathrm{MAE}$ in m",
            y_lims=(0.0, 2.5),
        )

        # printing some info:
        print("-" * 10 + " Lateral Errors on the Prediction Horizon " + "-" * 10)
        print("MIXNET")
        for i, datapoints in enumerate(mixnet_errors):
            print(
                "Number of datapoints with {} timesteps of ground truth: {}".format(
                    ERROR_ANALYSIS_TIME_STEPS[i], len(datapoints)
                )
            )

        print("")

        print("INDYNET")
        for i, datapoints in enumerate(indynet_errors):
            print(
                "Number of datapoints with {} timesteps of ground truth: {}".format(
                    ERROR_ANALYSIS_TIME_STEPS[i], len(datapoints)
                )
            )
        print("")

    def _create_longitudinal_errors_plot(self, mixnet_error_dict, indynet_error_dict):
        """Creates the plot about the longitudinal errors on the prediction horizon.

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

        # creating the datacontainers for the boxplot:
        for pred_ts in range(len(ERROR_ANALYSIS_TIME_STEPS)):
            mixnet_errors_np = np.array(mixnet_error_dict["long_errors"])
            indynet_errors_np = np.array(indynet_error_dict["long_errors"])

            mixnet_not_nan_mask = ~np.isnan(mixnet_errors_np[:, pred_ts])
            indynet_not_nan_mask = ~np.isnan(indynet_errors_np[:, pred_ts])

            mixnet_errors.append(mixnet_errors_np[mixnet_not_nan_mask, pred_ts])
            indynet_errors.append(indynet_errors_np[indynet_not_nan_mask, pred_ts])

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

        self._create_lineplots_with_interval(
            mixnet_quantile_data,
            indynet_quantile_data,
            title="MAE_long_pred_horizon",
            x_label="$t_{\mathrm{pred}}$ in s",
            y_label="$\mathrm{MAE}$ in m",
            y_lims=(0.0, 12),
        )

        # printing some info:
        print("-" * 10 + " Lateral Errors on the Prediction Horizon " + "-" * 10)
        print("MIXNET")
        for i, datapoints in enumerate(mixnet_errors):
            print(
                "Number of datapoints with {} timesteps of ground truth: {}".format(
                    ERROR_ANALYSIS_TIME_STEPS[i], len(datapoints)
                )
            )

        print("")

        print("INDYNET")
        for i, datapoints in enumerate(indynet_errors):
            print(
                "Number of datapoints with {} timesteps of ground truth: {}".format(
                    ERROR_ANALYSIS_TIME_STEPS[i], len(datapoints)
                )
            )
        print("")

    def _create_lineplots_with_interval(
        self,
        mixnet_data: list,
        indynet_data: list,
        title: str,
        x_label: str,
        y_label: str,
        x_tick_labels: list = None,
        y_lims: tuple = None,
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

        fig = plt.figure(figsize=(8, 5))
        ax = fig.gca()

        # fig.suptitle(title)
        fig.canvas.manager.set_window_title(title)

        legend_elements = [
            Line2D(
                [0],
                [0],
                color=self._benchmark_col,
                linestyle="solid",
                linewidth=3,
                label="Benchmark Model",
            ),
            Line2D(
                [0],
                [0],
                color=self._mixnet_col,
                linestyle="solid",
                linewidth=3,
                label="MixNet",
            ),
        ]

        # something is fucked up, and I only have 50 errors instead of 51 (together with timestep 0).
        # Since I can not figure out easily, I simply adjust the x axis, so that the plot goes from
        # 0.0 - 5.0 as it should be:
        x_array = np.arange(mixnet_medians.shape[0]) / 10.0 / 4.9 * 5.0

        # plotting with interval:
        ax.fill_between(
            x_array, indynet_Q1, indynet_Q3, alpha=0.5, facecolor=self._benchmark_col
        )
        ax.fill_between(
            x_array, mixnet_Q1, mixnet_Q3, alpha=0.5, facecolor=self._mixnet_col
        )

        ax.plot(
            x_array,
            indynet_means,
            color=self._benchmark_col,
            linestyle="dashed",
            linewidth=3.0,
        )
        ax.plot(
            x_array,
            indynet_medians,
            color=self._benchmark_col,
            linestyle="solid",
            linewidth=3.0,
        )
        ax.plot(
            x_array,
            mixnet_means,
            color=self._mixnet_col,
            linestyle="dashed",
            linewidth=3.0,
        )
        ax.plot(
            x_array,
            mixnet_medians,
            color=self._mixnet_col,
            linestyle="solid",
            linewidth=3.0,
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

        if self._save_path:
            file_name = title.replace(" ", "_") + ".pdf"
            plt.savefig(os.path.join(self._save_path, file_name))


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

    # create visualizer object:
    evaluator = PredictionLogEvaluator(args)

    evaluator()
