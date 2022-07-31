"""
This debug visualization tool takes (autoamically) the latest logfile
in the logs/ folder and visualizes the predictions.

    Arguments:
    --velocity: Plot the velocity profile.
    --dataset: Create a dataset for training etc. out of the ground truth data
"""

# Settings
ZOOM_VEH_ID = None  # Vehicle ID that should be zoomed to
import os
import sys
import argparse
import matplotlib
import tqdm
import math

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.use("TkAgg")

import numpy as np

from mix_net.mix_net.utils.logging_helper import (
    read_all_data,
    read_info_data,
    recover_params,
    recover_trajectories,
)
from mix_net.mix_net.utils.helper import fill_with_nans
from mix_net.mix_net.utils.geometry import get_heading
from mix_net.mix_net.src.boundary_generator import BoundaryGenerator
from tools.visualization_utils import (
    VehicleArtist,
    VelocityArtist,
    SliderGroup,
    COLOR_DICT,
)
from tools.file_utils import list_dirs_with_file

ERROR_ANALYSIS_TIME_STEPS = [9, 19, 29, 39, 49]
TRACK_PATH = "mix_net/mix_net/data/map/traj_ltpl_cl_IMS_GPS.csv"
CONFIG_PATH = "mix_net/mix_net/config/main_params.ini"
SAVE_DIR = "data/evaluation_data/smoothness"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


class PredictionLogVisualizer:
    """Class for visualizing the logfiles of the prediction module"""

    def __init__(self, args, zoom_veh_id, draw_uncertainty=False, yolo=False):
        """Initialize a PredictionLogVisualizer object.

        args:
            - zoom_veh_id: The id of the vehicle to which the visualization
                        is zoomed if provided.
            - plt_velocity: Whether to plot the velocity or not. Default: False
            - draw_uncertainty: Whether to plot the uncertainty or not. Default: False
        """
        self.args = args
        self._zoom_veh_id = zoom_veh_id
        self._plt_velocity = args.velocity
        self._draw_uncertainty = draw_uncertainty
        self._vehicle_artists = []

        self._load_log_data(args.logdir)

        if "track_path" not in self._recovered_params:
            self._recovered_params["track_path"] = TRACK_PATH

        self._boundary_generator = BoundaryGenerator(
            params=self._recovered_params,
        )

        # used for being able to keep zoomed in for multiple timesteps:
        self._xlim_def = None
        self._ylim_def = None

        if not yolo:
            self._init_figure()

            if args.analyze:
                self.analyze_prediction_errors()
            self._plot_received_ids_over_time()

    def _get_latest_dir_that_contains(self, word):
        """Finds the latest file in the logs directory that contains
        the expression given in word.
        """

        # Finding the latest log file:
        dir_list = list_dirs_with_file(["mix_net/mix_net/logs"], word)
        dir_list.sort(reverse=True)

        try:
            return dir_list[0]
        except IndexError:
            raise FileNotFoundError("No log file found!")

    def _load_log_data(self, logdir):
        """Loads the data from the logfiles. If logdir is specified, the
        logs from that folder are loaded. If it is not specified, the
        very last available log file is chosen.

        args:
            logdir: (str), the dir that contains the logfiles to load.
        """

        # searching for the files to open:
        if os.path.exists(logdir):
            data_file_path = os.path.join(logdir, "prediction_data.csv")
            info_file_path = os.path.join(logdir, "prediction_main.csv")

            if (not os.path.exists(data_file_path)) or (
                not os.path.exists(info_file_path)
            ):
                print(
                    'The logdir "{}" does not contain every necessary file.'.format(
                        logdir
                    )
                )
                logdir = self._get_latest_dir_that_contains("prediction_data")
                data_file_path = os.path.join(logdir, "prediction_data.csv")
                info_file_path = os.path.join(logdir, "prediction_main.csv")

        else:
            if logdir == "":
                print("No logdir provided. Taking the last log.")
            elif not os.path.exists(logdir):
                print(
                    'The provided logdir "{}" does not exist. Taking the last log.'.format(
                        logdir
                    )
                )

            logdir = self._get_latest_dir_that_contains("prediction_data")
            data_file_path = os.path.join(logdir, "prediction_data.csv")
            info_file_path = os.path.join(logdir, "prediction_main.csv")

        print("loading data from: {}".format(logdir))

        # read in data:
        _, self._all_log_data = read_all_data(data_file_path, zip_horz=True)
        self._no_lines_data = len(self._all_log_data)

        self._t_rel_0 = self._all_log_data[0][0]

        # Ground truth data:
        self._trajectories = recover_trajectories(
            [obj_dict for _, obj_dict, _, _, _, _ in self._all_log_data]
        )

        # if too many data, take every 10th step
        if self._no_lines_data > 1000:
            self._step = 10
        else:
            self._step = 1

        # read in info log:
        self._info_dict = read_info_data(info_file_path)
        self._recovered_params = recover_params(info_file_path)

        print("Recovered params:")
        print(self._recovered_params)
        print("")

    def _get_gt_from_trajs(self, ID, t_abs, pred_len):
        """Creates the ground truth trajectory for a given prediction based
        on the recovered trajectories and parameters.

        args:
            ID: The id of the car.
            t_abs: The absolute time of the prediction, for which the history is needed.
            pred_len: The length of the prediction in number of timesteps
        return:
            The ground truth trajectory in np.array with shape (2, N)
        """

        freq = float(int(self._recovered_params["MODEL_PARAMS"]["sampling_frequency"]))
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

    def _get_hist_from_trajs(self, ID, t_abs):
        """Recreates the history based on the recovered trajectories
        and the parameters.

        args:
            ID: The id of the car.
            t_abs: The absolute time of the prediction, for which the history is needed.
        return:
            The history in np.array with shape (2, N)
        """

        freq = float(int(self._recovered_params["MODEL_PARAMS"]["sampling_frequency"]))
        obs_len = float(
            int(self._recovered_params["OBJ_HANDLING_PARAMS"]["max_obs_length"])
        )
        t_list = self._trajectories[ID]["t_list"]

        t0 = t_abs - obs_len / freq
        t1 = t_abs

        if t0 < t_list[0]:
            t0 = t_list[0]

        # looking for sections, where no data has been received for a long time and
        # adjusting the end of the time interval accordingly:
        dt_max = 0.5
        i0 = np.argmax(t_list[t_list <= t0])
        i1 = np.argmax(t_list[t_list <= t1])
        if i1 >= t_list.shape[0]:
            i1 = t_list.shape[0] - 1

        for i in range(i1, i0, -1):
            if (t_list[i] - t_list[i - 1]) > dt_max:
                i1 = i
                N = np.floor((t1 - t_list[i1]) * freq)
                t0 = t1 - N / freq
                break

        return self._interp_traj(ID, t0, t1, obs_len + 1)

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

    def _plot_received_ids_over_time(self):
        """visualization tool to visualize which IDs were received in each timestep."""

        all_vehicle_ids = list(self._trajectories.keys())

        bool_received_matrix = np.full(
            (len(all_vehicle_ids), len(self._all_log_data)), False
        )

        for i, data in enumerate(self._all_log_data):
            pred_ids = list(data[4].keys())
            act_IDs = [data[4][pred_id]["vehicle_id"] for pred_id in pred_ids]

            for j, ID in enumerate(all_vehicle_ids):
                bool_received_matrix[j, i] = ID in act_IDs

        fig = plt.figure("Received IDS over Time")
        ax = fig.add_subplot(111)
        ax.imshow(
            bool_received_matrix,
            aspect="auto",
            cmap=plt.cm.binary,
            interpolation="nearest",
        )

        ax.set_yticks(np.arange(len(all_vehicle_ids)))
        ax.set_yticklabels(all_vehicle_ids)
        ax.set_xlabel("Time step")
        ax.set_title("Objects received per Time step")

    def _init_figure(self):
        """Initializes the figure"""

        self._fig = plt.figure("Prediction Debug Tool")

        if self._plt_velocity:
            self._ax = self._fig.add_subplot(121)
            self._ax2 = self._fig.add_subplot(122)

            # setting title and axis labels of ax2:
            self._ax2.set_title("Predicted and Ground Truth velocity profiles")
            self._ax2.set_xlabel("Prediction Timestep [-]")
            self._ax2.set_ylabel("Velocity [m/s]")
        else:
            self._ax = self._fig.add_subplot(111)

        # setting axis labels of ax1:
        self._ax.set_xlabel("X [m]")
        self._ax.set_ylabel("Y [m]")

        self._ax.axis("equal")

        plt.subplots_adjust(bottom=0.25)

        # slider for the global simulation time:
        self._global_slider = SliderGroup(
            fig=self._fig,
            left=0.2,
            bottom=0.1,
            width=0.6,
            height=0.04,
            max_val=self._no_lines_data - 1,
            step=self._step,
            text="Global Timestep",
            callback=self._update,
        )

        # slider for the local simulation time within a prediction:
        self._local_slider = SliderGroup(
            fig=self._fig,
            left=0.2,
            bottom=0.05,
            width=0.6,
            height=0.04,
            max_val=10,
            step=1,
            text="Local Timestep",
            callback=self._update_local_timestep_markers,
        )

        # connecting the pick event to the _on_pick() method:
        self._fig.canvas.mpl_connect("pick_event", self._on_pick)

    def _draw_all_predictions(
        self,
        pred_dict,
        hist,
        boundaries,
        t_rel,
        calc_time_avg=None,
    ):
        """Draws the predictions of every vehicle."""

        # storing the previous text_enabled values:
        self._visibility_list = [
            artist.get_visibility() for artist in self._vehicle_artists
        ]
        self._text_enabled_list = [
            artist.text_is_enabled for artist in self._vehicle_artists
        ]

        self._get_axlim()

        self._ax.cla()
        if self._plt_velocity:
            self._ax2.cla()

        self._vehicle_artists = []
        if self._plt_velocity:
            self._velocity_artists = []

        max_pred_len = 0

        for pred_id in pred_dict.keys():

            # The prediction:
            pred_x = pred_dict[pred_id]["x"]
            pred_y = pred_dict[pred_id]["y"]
            pred = np.vstack((pred_x, pred_y))
            vehicle_id = pred_dict[pred_id]["vehicle_id"]

            if max_pred_len < pred.shape[1]:
                max_pred_len = pred.shape[1]

            # division by 1e9 is needed, because mod_object sends nanosecs:
            t_abs = float(pred_dict[pred_id]["t_abs_perception"]) / 1e9

            # color:
            color = (
                COLOR_DICT[str(vehicle_id)]
                if str(vehicle_id) in COLOR_DICT.keys()
                else "tab:blue"
            )

            if "invalid" in self._info_dict[pred_id]:
                # Creating the artist for the vehicle:
                self._vehicle_artists.append(
                    VehicleArtist(
                        ax=self._ax,
                        fig=self._fig,
                        label="ID {} ({})".format(pred_id, vehicle_id),
                        hist=None,
                        pred=pred,
                        gt=None,
                        boundaries=None,
                        covariance=None,
                        info=self._compose_info_text(pred_id, pred_dict, -1.0),
                        color=color,
                    )
                )
            else:
                # If there is no real history log, it is created from the gt_dict:
                if hist[pred_id] is None:
                    hist_pid = self._get_hist_from_trajs(vehicle_id, t_abs)
                else:
                    hist_pid = np.array(hist[pred_id]).T
                    # Remove all zeros (that were added before feeding it to the nets):
                    hist_pid = np.ma.masked_equal(hist_pid, 0)

                # if there is no logged boundary, get it from the array that was loaded from the file.
                if boundaries[pred_id] is None:
                    (
                        left_bound,
                        right_bound,
                    ) = self._boundary_generator.get_bounds_between_points(
                        pred[:, 0], pred[:, -1]
                    )
                    boundaries_pid = np.array([left_bound, right_bound])
                else:
                    boundaries_pid = np.array(boundaries[pred_id])

                # getting the covariance:
                if self._draw_uncertainty:
                    covariance = [pred_dict[pred_id]["cov"]]
                else:
                    covariance = None

                # Ground Truth:
                gt = self._get_gt_from_trajs(vehicle_id, t_abs, pred.shape[1])

                # calculating the root mean square error:
                rmse = self._get_rmse(gt, pred)

                # composing the info text for the vehicle:
                info = self._compose_info_text(pred_id, pred_dict, rmse)

                # Creating the artist for the vehicle:
                self._vehicle_artists.append(
                    VehicleArtist(
                        ax=self._ax,
                        fig=self._fig,
                        label="ID {} ({})".format(pred_id, vehicle_id),
                        hist=hist_pid,
                        pred=pred,
                        gt=gt,
                        boundaries=boundaries_pid,
                        covariance=covariance,
                        info=info,
                        color=color,
                    )
                )

                if self._plt_velocity:
                    dt = 1.0 / float(
                        self._recovered_params["MODEL_PARAMS"]["sampling_frequency"]
                    )
                    # Ground Truth
                    gt_dx = np.diff(gt, axis=1)
                    gt_v = np.linalg.norm(gt_dx, axis=0) / dt

                    # Data-based
                    pred_dx = np.diff(pred, axis=1)
                    pred_v = np.linalg.norm(pred_dx, axis=0) / dt

                    # Adding a new container and plotting:
                    self._velocity_artists.append(
                        VelocityArtist(
                            ax=self._ax2,
                            label="ID {} ({})".format(
                                pred_id, pred_dict[pred_id]["vehicle_id"]
                            ),
                            pred_v=pred_v,
                            pred_t=np.arange(pred_v.shape[0]),
                            gt_v=gt_v,
                            gt_t=np.arange(gt_v.shape[0]),
                            color=color,
                        )
                    )

                    # creating legend for the velocity plot:
                    legend_elements = [
                        Line2D(
                            [0], [0], color="k", linestyle="solid", label="Prediction"
                        ),
                        Line2D(
                            [0],
                            [0],
                            color="k",
                            linestyle="dashed",
                            label="Ground Truth",
                        ),
                    ]

                    # setting legend, title and labels:
                    self._ax2.legend(handles=legend_elements)
                    self._ax2.set_title("Predicted and Ground Truth velocity profiles")
                    self._ax2.set_xlabel("Prediction Timestep [-]")
                    self._ax2.set_ylabel("Velocity [m/s]")

            # Zooming to the vehicle with the specified ID:
            if pred_dict[pred_id]["vehicle_id"] == self._zoom_veh_id:
                self._ax.set_xlim(
                    [
                        np.min(pred_x) - 10,
                        np.max(pred_x) + 10,
                    ]
                )
                self._ax.set_ylim(
                    [
                        np.min(pred_y) - 10,
                        np.max(pred_y) + 10,
                    ]
                )

        # plotting the raceline:
        # self._ax.plot(self._raceline_helper.line[:, 0], self._raceline_helper.line[:, 1], "k.")

        self._set_title(t_rel, calc_time_avg)

        self._set_interactive_legend()

        self._restore_previous_visibility()

        self._set_axlim()

        self._local_slider.update_max_val(max_pred_len - 1)

        self._update_local_timestep_markers(self._local_slider.val)

    def _get_axlim(self):
        """Gets some data about the current limits of the axes of the
        plot, so that the same ratios can be set back to the new plot
        by calling _set_axlim()
        """

        if self._xlim_def is None or self._ylim_def is None:
            # x
            self._ax_min_ratio_x = 0.0
            self._ax_max_ratio_x = 1.0

            # y
            self._ax_min_ratio_y = 0.0
            self._ax_max_ratio_y = 1.0
        else:
            xlim = self._ax.get_xlim()
            ylim = self._ax.get_ylim()

            # x
            x_length = self._xlim_def[1] - self._xlim_def[0]
            self._ax_min_ratio_x = (xlim[0] - self._xlim_def[0]) / x_length
            self._ax_max_ratio_x = (xlim[1] - self._xlim_def[0]) / x_length

            # y
            y_length = self._ylim_def[1] - self._ylim_def[0]
            self._ax_min_ratio_y = (ylim[0] - self._ylim_def[0]) / y_length
            self._ax_max_ratio_y = (ylim[1] - self._ylim_def[0]) / y_length

    def _set_axlim(self):
        """Sets the axis limits according to what was set in the
        previous timestep and hence logged by the _get_axlim() method.
        """
        # getting the actual maximum (default) axes limits before resizing them.
        self._xlim_def = self._ax.get_xlim()
        self._ylim_def = self._ax.get_ylim()

        # x
        x_length = self._xlim_def[1] - self._xlim_def[0]
        x_min = self._xlim_def[0] + self._ax_min_ratio_x * x_length
        x_max = self._xlim_def[0] + self._ax_max_ratio_x * x_length
        self._ax.set_xlim(x_min, x_max)

        # y
        y_length = self._ylim_def[1] - self._ylim_def[0]
        y_min = self._ylim_def[0] + self._ax_min_ratio_y * y_length
        y_max = self._ylim_def[0] + self._ax_max_ratio_y * y_length
        self._ax.set_ylim(y_min, y_max)

    def _compose_info_text(self, pred_id, pred_dict, rmse):
        """creating the info text that should be displayed next to the vehicle"""

        info = "ID: {0} ({1}); RMSE: {2:.1f}".format(
            pred_id, pred_dict[pred_id]["vehicle_id"], rmse
        )

        try:
            for i in self._info_dict[pred_id]:
                info += ", " + i
        except KeyError:
            print(
                "could not find info in the main log file\
                    for vehicle ID {}".format(
                    pred_id
                )
            )

        # just breaking the line, if it is too long:
        if len(info) > 30:
            if "collision" in info:
                i = info.index("collision") - 1
                info = info[:i] + "\n" + info[i:]

        return info

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
            gt: np.array((2, N)), the ground truth trajectory.
            pred: np.array((2, N)), the predicted trajectory.
        """
        ind_min = min(gt.shape[1], pred.shape[1])
        diff = gt[:, :ind_min] - pred[:, :ind_min]

        if gt.shape[1] <= 1:
            return np.abs(diff[0]), np.abs(diff[1])
        else:
            rotation_list = get_heading(pred[:, :ind_min].transpose())

        diff_T = diff.transpose()
        rotated_diff_list = []

        rot_mat = np.array(
            [
                [np.cos(rotation_list), -np.sin(rotation_list)],
                [np.sin(rotation_list), np.cos(rotation_list)],
            ]
        )

        for i, diff_point in enumerate(diff_T):
            rotated_diff_list.append(np.matmul(diff_point, rot_mat[:, :, i]))

        rotated_diff = np.array(rotated_diff_list)
        x_lat = rotated_diff[:, 0]
        y_long = rotated_diff[:, 1]

        return np.abs(x_lat), np.abs(y_long)

    def _on_pick(self, event):
        """Handles the event when one has clicked on the legend."""

        if event.artist in list(self._legend_artist_dict.keys()):
            # getting which artist it is and inverting its visibility:
            leg = event.artist
            for artist in self._legend_artist_dict[leg]:
                artist.invert_visibility()

            self._fig.canvas.draw()

    def _update(self, val):
        """The method to call, when the plot has to be updated.
        i.e when the slider has changed.
        """

        t_abs, _, hist, boundaries, pred_dict, calc_time_avg = self._all_log_data[
            int(val)
        ]

        try:
            t_rel = t_abs - self._t_rel_0
        except Exception:
            t_rel = -1

        self._draw_all_predictions(
            pred_dict, hist, boundaries, calc_time_avg=calc_time_avg, t_rel=t_rel
        )

        self._fig.canvas.draw()

    def _update_local_timestep_markers(self, val):
        """Sets the local timestep markers to the val^th timestep
        all of the plots.
        """

        for artist in self._vehicle_artists:
            artist.set_timestep_marker(val)

        self._fig.canvas.draw()

    def _set_title(self, t_rel, calc_time_avg):
        """Sets the title with some actual time info."""

        title_string = ""
        if t_rel:
            title_string += "Relative Time: {0:.2f} s".format(t_rel)

        if calc_time_avg:
            title_string += "     Calc Time: {0:.1f} ms".format(calc_time_avg * 1e3)

        self._ax.set_title(title_string)

        # setting axis labels of ax1:
        self._ax.set_xlabel("X [m]")
        self._ax.set_ylabel("Y [m]")

    def _set_interactive_legend(self):
        """Sets the interactive legend."""

        # creating the legend:
        self._legend = self._ax.legend()

        # creating dict to map the legend lines to the artists:
        self._legend_artist_dict = {}
        for i, leg in enumerate(self._legend.get_lines()):
            # setting clicking precision in pts:
            leg.set_picker(True)
            leg.set_pickradius(5)
            if self._plt_velocity:
                try:
                    self._legend_artist_dict[leg] = [
                        self._vehicle_artists[i],
                        self._velocity_artists[i],
                    ]
                except Exception:
                    self._legend_artist_dict[leg] = [self._vehicle_artists[i]]
            else:
                self._legend_artist_dict[leg] = [self._vehicle_artists[i]]

            # also, we set the legend for the corresponding vehicle artist:
            self._vehicle_artists[i].set_legend(leg)

    def _restore_previous_visibility(self):
        """setting the visibility of the cars according to the
        previous timestep.
        """

        if len(self._visibility_list) == len(self._vehicle_artists):
            for i, artist in enumerate(self._vehicle_artists):
                artist.set_text_enabled(self._text_enabled_list[i])
                artist.set_visibility(self._visibility_list[i])

                if self._plt_velocity:
                    self._velocity_artists[i].set_visibility(self._visibility_list[i])

        # If there was a change in the number of autos, every vehicle is
        # visible by defualt, and the visibility list is initialized accordingly:
        else:
            self._visibility_list = [True] * len(self._vehicle_artists)
            self._text_enabled_list = [False] * len(self._vehicle_artists)

    def analyze_prediction_errors(self):
        """
        Create histograms with errors in longitude and latitude according to the desired timesteps.
        """

        all_vehicle_ids = list(self._trajectories.keys())
        n_vehicles = len(all_vehicle_ids)

        # Create new figure
        fig, ax_list = plt.subplots(n_vehicles, 2)
        fig.canvas.set_window_title("Prediction Errors")

        iter_data = iter(self._all_log_data)

        lat_error_dict = {}
        long_error_dict = {}
        for veh_id in all_vehicle_ids:
            lat_error_dict[veh_id] = []
            long_error_dict[veh_id] = []

        print("Calculating Errors (this may take a while ...):")
        for line, data_sample in tqdm.tqdm(enumerate(iter_data)):

            (
                _,
                _,
                _,
                _,
                pred_dict,
                _,
            ) = data_sample

            for pred_id in pred_dict:

                # The prediction:
                pred_x = pred_dict[pred_id]["x"]
                pred_y = pred_dict[pred_id]["y"]
                pred = np.vstack((pred_x, pred_y))
                vehicle_id = pred_dict[pred_id]["vehicle_id"]

                # division by 1e9 is needed, because mod_object sends nanosecs:
                t_abs = float(pred_dict[pred_id]["t_abs_perception"]) / 1e9

                # Get ground truth
                gt = self._get_gt_from_trajs(vehicle_id, t_abs, pred.shape[1])

                # Only use time steps that are available
                analyze_time_steps = [
                    i for i in ERROR_ANALYSIS_TIME_STEPS if i < gt.shape[1]
                ]

                # Crop gt and pred
                gt = gt[:, analyze_time_steps]
                pred = pred[:, analyze_time_steps]

                # Get errors
                lat_error, long_error = self._get_lat_long_error(gt, pred)

                # Fill with nans as they have different lenghts
                lat_error_dict[vehicle_id].append(
                    fill_with_nans(lat_error, len(ERROR_ANALYSIS_TIME_STEPS))
                )
                long_error_dict[vehicle_id].append(
                    fill_with_nans(long_error, len(ERROR_ANALYSIS_TIME_STEPS))
                )

        # Ax labels
        ax_list[-1, 0].set(xlabel="Lateral Error")
        ax_list[-1, 1].set(xlabel="Longitudinal Error")

        print("Parsing Errors:")
        for i, error_dict in enumerate([lat_error_dict, long_error_dict]):
            for ax_idx, veh_id in enumerate(tqdm.tqdm(all_vehicle_ids)):

                ax_list[ax_idx, 0].set(ylabel=veh_id)
                data_list = []
                # Iterate over time steps
                for pred_ts in range(len(ERROR_ANALYSIS_TIME_STEPS)):
                    errors_list = list(error_dict[veh_id])
                    # Create data as needed for multiple boxplots
                    data_list.append(
                        [i[pred_ts] for i in errors_list if not math.isnan(i[pred_ts])]
                    )
                    ax_list[ax_idx, i].boxplot(data_list, showfliers=False)

    def visualize(self):
        """Visualizes the log"""

        self._update(0)
        plt.show()


if __name__ == "__main__":

    # Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir",
        type=str,
        default="",
        help="The logdir. If not provided, the latest log is used.",
    )
    parser.add_argument("--velocity", action="store_true", default=False)
    parser.add_argument("--analyze", action="store_true", default=False)
    args = parser.parse_args()

    # create visualizer object:
    visualizer = PredictionLogVisualizer(
        args=args,
        zoom_veh_id=ZOOM_VEH_ID,
        draw_uncertainty=False,
    )

    visualizer.visualize()
