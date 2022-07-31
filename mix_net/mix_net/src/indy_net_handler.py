import os
import sys
import numpy as np
import torch
import copy
from typing import Tuple

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from utils.handler_interface import HandlerInterface
from src.indy_net import IndyNet
from src.boundary_generator import BoundaryGenerator
from utils.cuda import cudanize
from utils.geometry import (
    get_heading,
    angle_between,
    transform_trajectory,
    retransform_trajectory,
    retransform_cov,
)


class IndyNetHandler(HandlerInterface):
    """Handler class for the IndyNet neural network for prediction."""

    def __init__(
        self,
        net: IndyNet,
        bg: BoundaryGenerator,
        params: dict,
        main_logger=None,
    ):
        """Initializes the IndyNetHandler object."""

        super().__init__()

        self._net = net.cuda() if params["use_cuda"] else net
        self._bg = bg
        self._main_logger = main_logger
        self._params = params
        self._use_cuda = params["use_cuda"]

    def predict(
        self,
        obs_storage: dict,
        prediction_id: int,
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
        """
        pred_dict = {}
        log_hist, log_obj, log_boundaries = {}, {}, {}

        if obs_storage == {}:
            return pred_dict, log_hist, log_obj, log_boundaries, prediction_id

        # some frequently used params:
        sampling_frequency = self._params["MODEL_PARAMS"]["sampling_frequency"]
        data_min_obs_length = self._params["MODEL_PARAMS"]["data_min_obs_length"]

        # Generate input for neural network
        (
            hist,
            left_bound,
            right_bound,
            translations,
            rotations,
        ) = self._generate_network_input(obs_storage)

        # Neural Network
        fut_preds = self._net(hist, left_bound, right_bound).detach().cpu().numpy()

        # Iterate over NN predictions
        for idx, obj_key in enumerate(obs_storage.keys()):

            fut_pred = np.expand_dims(fut_preds[:, idx, :], axis=1)

            # Calc time_list with relative times
            time_array = np.linspace(
                0,
                fut_pred.shape[0] / sampling_frequency,
                fut_pred.shape[0] + 1,
            )

            # Add (0,0) according to convention starting with t=0.0s
            pred_out = np.concatenate((np.zeros((1, 1, 5)), fut_pred), axis=0)

            # Retransform positions
            xy_global = retransform_trajectory(
                pred_out[:, 0, :2], translations[idx], rotations[idx]
            )

            # Retransform covariances
            cov_array = retransform_cov(pred_out, rotations[idx])

            # Get heading
            heading_list = get_heading(xy_global)

            # Get t_abs_perception
            t_abs_perception = obs_storage[obj_key]["t_abs_perception"]

            # Add data prediction to pred_dict
            pred_dict[prediction_id] = {
                "valid": True,
                "vehicle_id": obj_key,
                "prediction_type": "data",
                "t_abs_perception": t_abs_perception,
                "t": time_array,
                "x": xy_global[:, 0],
                "y": xy_global[:, 1],
                "cov": cov_array,
                "heading": heading_list,
            }

            # Log
            log_hist[prediction_id] = retransform_trajectory(
                self.pre_log_hist[idx, :, :], translations[idx, :], rotations[idx]
            )

            log_obj[prediction_id] = (
                obj_key,
                translations[idx, :],
                t_abs_perception,
            )

            log_boundaries[prediction_id] = [
                retransform_trajectory(
                    self.log_left_bounds[idx, :, :],
                    translations[idx, :],
                    rotations[idx],
                ),
                retransform_trajectory(
                    self.log_right_bounds[idx, :, :],
                    translations[idx, :],
                    rotations[idx],
                ),
            ]

            if self._main_logger is not None:
                self._main_logger.info(
                    "Prediction-ID {0:d}: Long observation (hist={1:.1f} > min_obs_lenght={2:.1f}) - data-prediction applied, id = {3}".format(
                        prediction_id,
                        len(obs_storage[obj_key]["xy_positions"]) / sampling_frequency,
                        data_min_obs_length,
                        obj_key,
                    )
                )

            # Count predictions
            prediction_id += 1

        return pred_dict, log_hist, log_obj, log_boundaries, prediction_id

    def _generate_network_input(self, obs_storage: dict):
        """Generates the network inputs for the IndyNet.

        args:
            obs_storage: (dict), the observation storage that was received from mod_objects
                and which already only contains the IDs which have to be predicted
                data based.

        returns: tuple, the inputs for the IndyNet:
            hist: [tensor with shape=(max_hist_len, batch_size, 2)],
            left_bound: [tensor with shape=(bound_len, batch_size, 2)],
            right_bound: [tensor with shape=(bound_len, batch_size, 2)],
            translations: [np.array with shape=(batch_size, 2)],
            rotations: [np.array with shape=(batch_size,)]
        """

        # some frequently used variables:
        max_obs_length = self._params["OBJ_HANDLING_PARAMS"]["max_obs_length"]

        hist_list = []
        translations = []
        rotations = []
        left_bounds, right_bounds = [], []

        for obj_vals in obs_storage.values():
            # Fill different input lengths with zeros
            hist_zeros = np.zeros((max_obs_length, 2))

            # collecting the last states for boundary generation and transformation:
            translation = copy.deepcopy(obj_vals["xy_positions"][0, :])
            translations.append(translation)

            # Get Boundaries
            left_bound, right_bound = self._bg.get_boundaries_single(translation)

            # getting the rotation:
            rotation = angle_between(left_bound[1, :] - left_bound[0, :], [1, 0])
            rotations.append(rotation)

            # Transforming the history and CHANGING HISTORY ORDER!!!!:
            hist_zeros[: obj_vals["xy_positions"].shape[0], :] = transform_trajectory(
                obj_vals["xy_positions"][::-1, :], translation, rotation
            )
            hist_list.append(hist_zeros)

            # also transforming the boundaries and adding them to the list:
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

        # Log hist
        self.pre_log_hist = copy.deepcopy(hist)

        # Transform hist
        hist = np.swapaxes(hist, 0, 1)

        # Log Boundaries
        self.log_left_bounds = copy.deepcopy(left_bounds)
        self.log_right_bounds = copy.deepcopy(right_bounds)

        # Swap axes
        left_bound = np.swapaxes(np.array(left_bounds), 0, 1)
        right_bound = np.swapaxes(np.array(right_bounds), 0, 1)

        # Cudanize
        if self._use_cuda:
            hist, left_bound, right_bound = cudanize(hist, left_bound, right_bound)
        else:
            hist = torch.as_tensor(hist).float()
            left_bound = torch.as_tensor(left_bound).float()
            right_bound = torch.as_tensor(right_bound).float()

        return hist, left_bound, right_bound, translations, rotations
