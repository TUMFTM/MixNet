import os
import sys
import numpy as np
import torch
import copy
from typing import Tuple

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from utils.handler_interface import HandlerInterface
from src.mix_net import MixNet
from src.boundary_generator import BoundaryGenerator
from utils.map_utils import get_track_paths
from utils.line_helper import LineHelper
from utils.geometry import (
    get_heading,
    angle_between,
    transform_trajectory,
    retransform_trajectory,
)


class MixNetHandler(HandlerInterface):
    """Handler class for the IndyNet neural network for prediction."""

    def __init__(self, net: MixNet, bg: BoundaryGenerator, params, main_logger=None):
        """Initializes the IndyNetHandler object."""

        super().__init__()

        self.net = net.float()
        self._bg = bg
        self._main_logger = main_logger
        self._params = params

        self._device = net.device

        self._create_line_helpers()

        self._get_convinience_matrices()

        self._get_time_matrix()

    def predict(
        self, obs_storage: dict, prediction_id: int, physics_pred: dict = None
    ) -> Tuple[dict, dict, dict, dict, int]:
        """Carries out the predictions.

        args:
            obs_storage: (dict), the observation storage that was received from mod_objects
                and which already only contains the IDs which have to be predicted
                data based.
            prediction_id: (int), the prediction ID of the next prediction.

        returns:
            pred_dict: (dict), the dictionary which then directly can be added to the prediction
                dict in the main,
            log_hist: (dict), the dict of history logs,
            log_obj: (dict), the dict of object logs,
            log_boundaries: (dict), the log of boundary logs,
            prediction_id: (int), The next prediction ID.
            physics_pred: [dict], the physics based prediction from the tracker. I it is said so in the params,
                it is also considered for the velocity profile.
        """

        self._received_ids = list(obs_storage)

        pred_dict = {}
        log_hist, log_obj, log_boundaries = {}, {}, {}

        if obs_storage == {}:
            return pred_dict, log_hist, log_obj, log_boundaries, prediction_id

        # Generate input for neural network
        (
            hist,
            left_bound,
            right_bound,
            translations,
            rotations,
        ) = self.generate_network_input(obs_storage)

        # Neural Network
        with torch.no_grad():
            mixers, vels, accels = self.net(hist, left_bound, right_bound)

        mixers = mixers.cpu().numpy()
        vels = vels.cpu().numpy()
        accels = accels.cpu().numpy()

        # mixing the paths:
        x_mixes = (self._lines_x @ mixers.T).T  # (num_vehs, line_length)
        y_mixes = (self._lines_y @ mixers.T).T  # (num_vehs, line_length)
        arc_mixes = (self._line_arcs @ mixers.T).T  # (num_vehs, line_length + 1)

        arc_dists = self._get_arc_dists(
            vels, accels, translations, x_mixes, y_mixes, arc_mixes, obs_storage
        )

        for idx, (obj_id, obj_vals) in enumerate(obs_storage.items()):
            # interpolating:
            pred_x = np.interp(
                arc_dists[idx, :],
                arc_mixes[idx, :],
                np.hstack((x_mixes[idx, :], x_mixes[idx, 0])),
            )

            pred_y = np.interp(
                arc_dists[idx, :],
                arc_mixes[idx, :],
                np.hstack((y_mixes[idx, :], y_mixes[idx, 0])),
            )

            pred = np.stack((pred_x, pred_y), axis=1)

            # checking if the prediction should be overriden by physics based prediction:
            if (
                self._params["MIX_NET_PARAMS"]["safety_physics_override"]
                and physics_pred is not None
                and self._sketchy_prediction(
                    pred, obj_vals["xy_positions"][0, :], obj_vals["yaw"]
                )
            ):
                time_array = np.array([pred.t for pred in physics_pred[obj_id]["pred"]])

                pred = np.array(
                    [(pred.x, pred.y) for pred in physics_pred[obj_id]["pred"]]
                )

                heading_list = [pred.heading for pred in physics_pred[obj_id]["pred"]]
                prediction_type_log = "data-physics-override"

            else:
                pred = self._correct_pred_beginning(pred, translations[idx, :])

                # Calc time_list with relative times
                time_array = np.linspace(
                    0,
                    pred.shape[0] * self._params["MIX_NET_PARAMS"]["dt"],
                    pred.shape[0],
                )

                # Get heading
                heading_list = get_heading(pred)

                prediction_type_log = "data"

            cov_list = np.zeros((pred.shape[0], 2, 2))

            # Add data prediction to pred_dict
            pred_dict[prediction_id] = {
                "valid": True,
                "vehicle_id": obj_id,
                "prediction_type": "data",
                "t_abs_perception": obj_vals["t_abs_perception"],
                "t": time_array,
                "x": pred[:, 0],
                "y": pred[:, 1],
                "cov": cov_list,
                "heading": heading_list,
            }

            # Log
            log_hist[prediction_id] = self.pre_log_hist[obj_id]

            log_obj[prediction_id] = (
                obj_id,
                translations[idx],
                obj_vals["t_abs_perception"],
            )

            log_boundaries[prediction_id] = [
                retransform_trajectory(
                    left_bound[idx, :, :].to("cpu").numpy(),
                    translations[idx],
                    rotations[idx],
                ),
                retransform_trajectory(
                    right_bound[idx, :, :].to("cpu").numpy(),
                    translations[idx],
                    rotations[idx],
                ),
            ]

            if self._main_logger is not None:
                self._main_logger.info(
                    self._get_logger_string(
                        prediction_id,
                        len(obj_vals["xy_positions"]),
                        prediction_type_log,
                        obj_id,
                        mixers[idx, :].tolist(),
                    )
                )

            # Count predictions
            prediction_id += 1

        return pred_dict, log_hist, log_obj, log_boundaries, prediction_id

    def generate_network_input(self, obs_storage: dict):
        """Generates the network inputs for the IndyNet.

        args:
            obs_storage: (dict), the observation storage that was received from mod_objects
                and which already only contains the IDs which have to be predicted
                data based.

        returns: tuple, the inputs for the IndyNet:
            hist: [tensor with shape=(max_hist_len, batch_size, 2)],
            left_bound: [tensor with shape=(bound_len, batch_size, 2)],
            right_bound: [tensor with shape=(bound_len, batch_size, 2)],
            translations: [np.array with shape=(batch_size, 2)], the actual positions before the transformation.
            rotations: [np.array with shape=(batch_size,)], the actual rotations before the transformation.
        """

        # some frequently used variables:
        max_obs_length = self._params["OBJ_HANDLING_PARAMS"]["max_obs_length"]

        hist_list = []
        translations = []
        rotations = []
        left_bounds, right_bounds = [], []

        self.pre_log_hist = {}

        for obj_id, obj_vals in obs_storage.items():
            # shrink size
            if obj_vals["xy_positions"].shape[0] > max_obs_length:
                obj_vals["xy_positions"] = obj_vals["xy_positions"][:max_obs_length, :]

            self.pre_log_hist[obj_id] = copy.deepcopy(obj_vals["xy_positions"])

            # Fill different input lengths with zeros
            hist_zeros = np.zeros((max_obs_length, 2))

            # collecting the last states for boundary generation and transformation:
            translation = copy.deepcopy(obj_vals["xy_positions"][0])
            translations.append(translation)

            # Get Boundaries
            left_bound, right_bound = self._bg.get_boundaries_single(translation)

            # getting the rotation:
            rotation = angle_between(left_bound[1, :] - left_bound[0, :], [1, 0])
            rotations.append(rotation)

            # Transforming the history and CHANGING HISTORY ORDER!!!!:
            hist_zeros[: obj_vals["xy_positions"].shape[0], :] = transform_trajectory(
                obj_vals["xy_positions"][::-1, :].copy(), translation, rotation
            )
            hist_list.append(hist_zeros)

            # Transforming the boundaries and filling up the lists:
            left_bounds.append(transform_trajectory(left_bound, translation, rotation))
            right_bounds.append(
                transform_trajectory(right_bound, translation, rotation)
            )

        if len(hist_list) == 0:
            return None, None, None, None, None

        # numpifying:
        hist = np.array(hist_list)
        translations = np.array(translations)
        left_bounds = np.array(left_bounds)
        right_bounds = np.array(right_bounds)

        # create tensors:
        hist = torch.from_numpy(hist).float().to(self._device)
        left_bound = torch.from_numpy(left_bounds).float().to(self._device)
        right_bound = torch.from_numpy(right_bounds).float().to(self._device)

        return hist, left_bound, right_bound, translations, rotations

    def _create_line_helpers(self):
        """Creates the LineHelper objects."""

        (_, centerline, bound_right, bound_left, raceline) = get_track_paths(
            track_path=self._params["MIX_NET_PARAMS"]["map_file_path"],
            bool_raceline=True,
        )

        self._left_bd = LineHelper(bound_left)
        self._right_bd = LineHelper(bound_right)
        self._centerline = LineHelper(centerline)
        self._raceline = LineHelper(raceline)

    def _get_convinience_matrices(self):
        """Creates some matrices which can directly be multiplied by the mixer vectors and hence
        are making the calculations more convinent.
        """

        self._lines_x = np.stack(
            (
                self._left_bd.line[:, 0],
                self._right_bd.line[:, 0],
                self._centerline.line[:, 0],
                self._raceline.line[:, 0],
            ),
            axis=1,
        )

        self._lines_y = np.stack(
            (
                self._left_bd.line[:, 1],
                self._right_bd.line[:, 1],
                self._centerline.line[:, 1],
                self._raceline.line[:, 1],
            ),
            axis=1,
        )

        self._line_arcs = np.stack(
            (
                self._left_bd.arc_lens,
                self._right_bd.arc_lens,
                self._centerline.arc_lens,
                self._raceline.arc_lens,
            ),
            axis=1,
        )

    def _get_time_matrix(self):
        """Creates a matrix which make the velocity profile calculation easier.
        If the matrix is multiplied with the section-wise acceleration that is one of the
        outputs of the network, it will create a relative velocity profile that can be
        added to the initial velocity profile.

        The matrix has the size (N, M) where N is the number of timesteps and M is the
        number of sections (components) of the vel profile.
        """

        self._time_profile_matrix = np.zeros(
            (self._params["MIX_NET_PARAMS"]["pred_len"], 5), dtype=np.float32
        )

        for i in range(5):
            self._time_profile_matrix[(i * 10) : ((i + 1) * 10), i] = np.linspace(
                0.1, 1.0, 10
            )

            self._time_profile_matrix[((i + 1) * 10) :, i] = 1.0

    def _get_arc_dists(
        self, vels, accels, translations, x_mixes, y_mixes, arc_mixes, obs_storage
    ):
        """Calculates the velocity profiles based on the initial velocites and the
        section-wise constant accelerations. Then it calculates the displacements and
        hence returns the arc distances of the prediction steps.

        args:
            vels: [np.array with shape=(batch_size, 1)], the initial velocities of the cars.
            accels: [np.array with shape=(batch_size, num_of_accel_sections)],
                the section_wise constant accelerations.
            translations: [np.array with shape=(batch_size, 2)], the initial positions.
            x_mixes: [np.array with shape=(batch_size, line_len)], the x coordinates of the mixture line
            y_mixes: [np.array with shape=(batch_size, line_len)], the y coordinates of the mixture line
            arc_mixes: [np.array with shape=(batch_size, line_len + 1)], the arc lengths corresponding to
                the points of the mixture line.
            obs_storage: [dict], observation storage from the tracker. If it is said so in the params,
                current velocity is taken from there.

        returns:
            arc_dists: [np.array with shape(batch_size, pred_len + 1)]
        """

        vel_profile = self.get_vel_profile(vels, accels, obs_storage)

        arc_dists = np.zeros(
            (vels.shape[0], (self._params["MIX_NET_PARAMS"]["pred_len"] + 1))
        )

        # finding the arc distance of the initial position:
        for veh_idx, pos in enumerate(translations):
            """First we are trying to find the two points which are closest to the car.
            The first point (i0 index) is the closest one. After that, we have to decide,
            which of the 2 neighbouring indices is closer to the position. The second closest
            points index is i1.
            """
            i0 = self._centerline.get_nearest_ind(pos)
            i_next = i0 + 1 if i0 < (self._centerline.line.shape[0] - 1) else 0
            i_prev = i0 - 1 if i0 > 0 else (self._centerline.line.shape[0] - 1)

            closest_point = np.array([x_mixes[veh_idx, i0], y_mixes[veh_idx, i0]])
            next_path_point = np.array(
                [x_mixes[veh_idx, i_next], y_mixes[veh_idx, i_next]]
            )

            pos_to_closest = closest_point - pos
            tangent = next_path_point - closest_point

            # checking whether the car is before or after the point which is the closest to it on the path:
            i1 = i_next if tangent @ pos_to_closest > 0 else i_prev

            pos_to_other = np.array([x_mixes[veh_idx, i1], y_mixes[veh_idx, i1]]) - pos

            """Until now we have found the 2 points which are closest to the car. In the followings
            we are trying to find the exact arc length corresponding to the car. This is found by
            interpolating the arc length between the 2 previously found points. The only complication is
            around the beginning/end of the track. This is why all this magic is written here.
            """

            # The lines have N points but the arc length list has N+1 points, because there are
            # 2 arc distances corresponding to the very first point. (0 and line length). In the
            # followings we decide whether we should change some of the indices to get the correct
            # arc length values from this list:
            if (i0 == 0) and (i1 == (self._centerline.line.shape[0] - 1)):
                i0 = self._centerline.line.shape[0]
            elif (i0 == (self._centerline.line.shape[0] - 1)) and (i1 == 0):
                i1 = self._centerline.line.shape[0]

            # calculating the distance along the mix path:
            l0 = np.linalg.norm(pos_to_closest)
            l1 = np.linalg.norm(pos_to_other)
            ratio = l0 / (l0 + l1 + 1e-8)

            # the arc distance of the actual position:
            arc_dist_init = (1.0 - ratio) * arc_mixes[veh_idx, i0] + ratio * arc_mixes[
                veh_idx, i1
            ]

            # integrating the velocity profile forward:
            rel_dists = np.cumsum(
                (self._params["MIX_NET_PARAMS"]["dt"] * vel_profile[veh_idx, :])
            )

            # filling up the matrix that contains the arc distances for each vehicle:
            arc_dists[veh_idx, 0] = arc_dist_init
            arc_dists[veh_idx, 1:] = arc_dist_init + rel_dists

            # pushing back the distances which are over the finishline:
            arc_dists[
                veh_idx, arc_dists[veh_idx, :] > arc_mixes[veh_idx, -1]
            ] -= arc_mixes[veh_idx, -1]

        return arc_dists

    def get_vel_profile(self, vels, accels, obs_storage):
        """Calculates the velocity profile based on the initial velocities
        and the section-wise constant accelerations.

        args:
            vels: [np.array with shape=(batch_size, 1)], the initial velocities of the cars.
            accels: [np.array with shape=(batch_size, num_of_accel_sections)],
                the section_wise constant accelerations.
            obs_storage: [dict], the object storage from the tracker. If it is said so in the params,
                it is used to assume constan velocity for the prediction.

        returns:
            vel_profile: [np.array with shape=(batch_size, pred_len)]: the velocity profiles
        """

        if self._params["MIX_NET_PARAMS"]["physics_based_const_vel"]:
            vel_profile = np.array(
                [
                    np.ones(self._params["MIX_NET_PARAMS"]["pred_len"])
                    * obs_storage[ID]["v"]
                    for ID in self._received_ids
                ]
            )

            vel_profile = (
                np.ones((vels.shape[0], self._params["MIX_NET_PARAMS"]["pred_len"]))
                * vel_profile
            )
        else:
            if self._params["MIX_NET_PARAMS"]["physics_based_init_vel"]:
                physics_vels = np.expand_dims(
                    np.array([obs_storage[ID]["v"] for ID in self._received_ids]), 1
                )
                init_vels = (
                    np.ones((vels.shape[0], self._params["MIX_NET_PARAMS"]["pred_len"]))
                    * physics_vels
                )

            else:
                init_vels = (
                    np.ones((vels.shape[0], self._params["MIX_NET_PARAMS"]["pred_len"]))
                    * vels
                )

            rel_vels = (self._time_profile_matrix @ accels.T).T

            vel_profile = init_vels + rel_vels

        return vel_profile

    def _correct_pred_beginning(self, pred, pos):
        """The prediction does not necessarily start exactly from the current position,
        so it is corrected here. As a result, the very first point will be exactly at
        the current position of the car and the first couple of points will also be
        adjusted to create a smooth trajectory.

        args:
            pred: [np.array with shape=(pred_len + 1, 2)]
            pos: [np.array with shape=(2,)]: the actual position of the vehicle.

        returns:
            pred: [np.array with shape=(pred_len + 1, 2)]: the adjusted prediction.
        """

        diff = pos - pred[0, :]

        interp_len = 10
        multipliers = np.linspace(1.0, 0.0, num=interp_len)

        # creating a matrix from the multipliers by stacking the vector twice into it and
        # multiplying the initial position difference with it. The adding it to the first
        # couple of points of the prediction and hence correcting it:
        pred[:interp_len, :] += (
            np.tile(multipliers[:, np.newaxis], (1, 2)) * diff[np.newaxis, :]
        )

        return pred

    def _sketchy_prediction(self, pred, pos, yaw):
        """Checks whether a prediciton can be considered thrustworthy or not.
        It is decided based on how far the first point was predicted from the actual position.

        args:
            pred: [np.array with shape=(N, 2)], the interpolated but not yet corrected trajectory.
            pos: [np.array with shape=(2,)], the actual position of the vehicle.
            yaw: [float], the yaw orientation of the vehicle.

        returns:
            true if the prediction is sketchy.
        """

        pos_to_pred = pred[0, :] - pos

        # the convention sais: yaw = 0 in the y direction, yaw = 90 in the -x direction:
        normal = np.array([-np.sin(yaw - (np.pi / 2)), np.cos(yaw - (np.pi / 2))])

        normal_dist = np.abs(normal @ pos_to_pred)
        self._initial_error = normal_dist

        return normal_dist > self._params["MIX_NET_PARAMS"]["override_error"]

    def _get_logger_string(
        self, prediction_id, obs_len, prediction_type, obj_key, mixers
    ):
        """Composes the message to be logged."""
        # some frequently used params:
        sampling_frequency = 1.0 / self._params["MIX_NET_PARAMS"]["dt"]
        data_min_obs_length = self._params["MIX_NET_PARAMS"]["data_min_obs_length"]

        log_str = "Prediction-ID {:d}: ".format(prediction_id)

        log_str += "Long observation (hist={:.1f} > min_obs_lenght={:.1f}) ".format(
            obs_len / sampling_frequency, data_min_obs_length
        )

        log_str += "- {}-prediction applied, ".format(prediction_type)

        log_str += "id = {}, ".format(obj_key)

        log_str += "mixers = " + "[{:s}], ".format(
            ", ".join(["{:.3f}".format(mixer) for mixer in mixers])
        )

        log_str += "initial error = {:.2f}".format(self._initial_error)

        return log_str
