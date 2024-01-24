"""Main functionalities for the inference of vehicle trajectory prediction."""
# Standard imports
import os
import sys
import json
import signal

repo_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_path)

# Set threads to 1
os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=1

# Third party imports
import numpy as np

import rclpy

# Custom imports
from utils.ros2_interface import ROS2Handler
from utils.setup_helpers import (
    create_path_dict,
    stamp2time,
)

from src.mix_net import MixNet
from src.indy_net import IndyNet
from src.mix_net_handler import MixNetHandler
from src.indy_net_handler import IndyNetHandler
from src.rulebased_interaction import RuleBasedInteraction
from src.boundary_generator import BoundaryGenerator

PATH_DICT = create_path_dict()


class MixNetNode(ROS2Handler):
    """Class object central for trajectory prediction."""

    def __init__(self):
        """Create logger, get all params, initiliaze ROS2 node and parameters."""
        # Initialize ros2 handler
        super().__init__(path_dict=PATH_DICT)

        # Initialize rule based interaction
        self.rule_based_interaction = RuleBasedInteraction(
            all_params=self.params,
            logger=self.main_logger,
        )

        # Create boundary generator
        self.boundary_gen = BoundaryGenerator(params=self.params)

        # creating the model specific model handler:
        self.get_model_handler()

        # Module state
        self.module_state = 20
        self.main_logger.info(f"STATE: mod_prediction state = {self.module_state}")

    def get_model_handler(self):
        """Initialize the model handler that is needed for the specified model."""
        # Choose inference net
        if "indy" in self.model_type.lower():
            net_class = IndyNet
            handler_class = IndyNetHandler
        else:
            net_class = MixNet
            handler_class = MixNetHandler

        # Load params
        with open(self.params["INF_MODEL_PARAMS"]["param_file"], "r") as fp:
            net_params = json.load(fp)
        net_params["use_cuda"] = self.params["INF_MODEL_PARAMS"]["use_cuda"]

        # Initialize neural network
        net = net_class(net_params)

        # Load model weights
        net.load_model_weights(
            weights_path=self.params["INF_MODEL_PARAMS"]["weights_file_path"]
        )

        # Intialize handler
        self.model_handler = handler_class(
            net=net,
            bg=self.boundary_gen,
            params=self.params,
            main_logger=self.main_logger,
        )

    def timer_callback(self):
        """Step function as ROS 2 callback."""
        # Module state
        self.module_state = 30
        self.start_time = stamp2time(*self.get_clock().now().seconds_nanoseconds())

        # prediction
        self.predict()

        # send prediction
        self.send_prediction()

        # send watchdog
        self.send_watchdog_state()

        # Reset input
        self.tracked_objects = None
        self.physics_prediction = None

        # Exit node
        if self.params["ci"]:
            sys.exit(0)

    def predict(self):
        """Calculate everything necessary for the prediction of a timestep."""
        # Differentiate None, {} and {..}
        self.pred_dict = None

        # Check if tracking-input received
        if self.tracked_objects is None:
            return

        # Time measurement
        st_time = stamp2time(*self.get_clock().now().seconds_nanoseconds())

        # Initialize dicts
        self.pred_dict = {}
        self.log_hist = {}
        self.log_obj = {}
        self.log_boundaries = {}
        self.deleted_histories = {}

        # log the list of received IDs:
        self.log_received_ids()

        # handle invalid objects:
        self.handle_invalid_obs()

        # Check for static objects
        self.check_for_static_object()

        # Check for short history
        self.physics_based_prediction()

        # Databased prediction
        self.data_based_prediction()

        if self.params["INTERACTION_PARAMS"]["rule_based"]:
            # Rule-based interactions
            self.pred_dict = self.rule_based_interaction.apply_to_predictions(
                self.pred_dict
            )

        # Log prediction result
        if bool(self.pred_dict):
            self.log_pred_data(
                st_time=st_time,
            )

        # Safely remove EGO vehicle out of pred_dict
        self.remove_ego_pred()

    def log_received_ids(self):
        """Log which IDs were received in this timestep."""
        self.main_logger.info(("Received IDs: {}".format(list(self.tracked_objects))))

    def handle_invalid_obs(self):
        """Look for objects which have obj_valid false value.

        These are removed from the further processing but are logged.
        """
        remove_list = []

        for obj_key, obj_vals in self.tracked_objects.items():
            # skipping valid objects:
            if (
                self.physics_prediction is None
                or obj_key not in self.physics_prediction
            ):
                is_valid = False
            else:
                is_valid = self.physics_prediction[obj_key]["valid"]

            if is_valid:
                continue

            remove_list.append(obj_key)

            # Return the last position as prediction:
            self.pred_dict[self.prediction_id] = {
                "valid": False,
                "vehicle_id": obj_key,
                "prediction_type": "invalid",
                "t_abs_perception": obj_vals["t_abs_perception"],
                "t": 0.0,
                "x": obj_vals["xy_positions"][0][0],
                "y": obj_vals["xy_positions"][0][1],
                "heading": obj_vals["yaw"],
            }

            # Logging:
            self.main_logger.info(
                "Prediction-ID {}: Invalid object - no prediction, id = {}".format(
                    self.prediction_id, obj_key
                )
            )

            self.log_non_data_based_obj(obj_key, obj_vals)

        # Remove keys
        self.remove_from_tracked_objects(remove_list)

    def check_for_static_object(self):
        """Check if an object is static (= remeins below stat_vel_threshhold).

        If static add static prediction to prediction dict and remove object from obs_smpl_freq
        """
        remove_list = []

        for obj_key, obj_vals in self.tracked_objects.items():
            if obj_vals["v"] < self.stat_vel_threshhold:
                # Static obstacle: Remove object from dict
                remove_list.append(obj_key)

                # Predict the past average positions for the future
                time_array = np.arange(
                    0,
                    self.stat_prediction_horizon + 1 / self.sampling_frequency,
                    1 / self.sampling_frequency,
                )

                self.pred_dict[self.prediction_id] = {
                    "valid": True,
                    "vehicle_id": obj_key,
                    "prediction_type": "static",
                    "t_abs_perception": obj_vals["t_abs_perception"],
                    "t": time_array,
                    "x": np.full(
                        time_array.shape,
                        np.mean(obj_vals["xy_positions"][0][0]),
                    ),
                    "y": np.full(
                        time_array.shape,
                        np.mean(obj_vals["xy_positions"][0][1]),
                    ),
                    "heading": np.full(time_array.shape, obj_vals["yaw"]),
                }

                # Logger message for static obstacle
                self.main_logger.info(
                    "Prediction-ID {}: Classified object as static - no prediction, id = {}".format(
                        self.prediction_id, obj_key
                    )
                )

                self.log_non_data_based_obj(obj_key, obj_vals)

        # Remove keys
        self.remove_from_tracked_objects(remove_list)

    def physics_based_prediction(self):
        """Use physics based prediction when the history is short.

        If objects are not static, it is checked whether their history
        length is long enough and whether the velocity and acceleration profiles
        lie within the defined boundaries. If any of these fail, physics based prediction
        is sent for the object. The objects which have been predicted like this are then
        removed from the observation storage. The rest is sent on to data based prediction.
        """
        remove_list = []

        for obj_key, obj_vals in self.tracked_objects.items():
            # Check if rather data_based prediction should be performed
            if len(obj_vals["xy_positions"]) >= int(
                self.data_min_obs_length * self.sampling_frequency
            ):
                curr_vel = obj_vals["v"]
                curr_acc = 0.0

                # If the velocity and acceleration constraints check out,
                # and vehicle is not in pit lane, the for-loop is breaked,
                # because this object should be predicted data based:
                if (
                    curr_vel >= self.data_min_velocity
                    and np.abs(curr_acc) <= self.data_max_acceleration
                ):
                    continue

                # else, log the reason for physics based prediction:
                if curr_vel < self.data_min_velocity:
                    if np.abs(curr_acc) > self.data_max_acceleration:
                        phys_pred_reason = "low vel and high acc"
                    else:
                        phys_pred_reason = "low vel"
                else:
                    phys_pred_reason = "high acc"

            # else, log the reason for physics based prediction:
            else:
                phys_pred_reason = "too short history"

            remove_list.append(obj_key)

            # don't carry out prediction if there is no physics based prediction received.
            # the ID of the car will still be removed from the observation storage.
            if (
                self.physics_prediction is None
                or obj_key not in self.physics_prediction
            ):
                continue

            phys_prediction = self.physics_prediction[obj_key]

            # Get t_abs_perception
            t_abs_perception = obj_vals["t_abs_perception"]

            # Change to rail-based prediction
            pred_str = "physics-prediction"
            pred_type_str = "rail"

            if phys_prediction["valid"]:
                self.main_logger.info(
                    "Prediction-ID {0:d}: {1:s} applied, reason: {2}, history length: (hist={3:.1f} vs. min_obs_length={4:.1f}), id = {5}".format(
                        self.prediction_id,
                        pred_str,
                        phys_pred_reason,
                        len(obj_vals["xy_positions"]) / self.sampling_frequency,
                        self.data_min_obs_length,
                        obj_key,
                    )
                )
            else:
                self.main_logger.info("Prediction-ID {0:d}: invalid")

            self.pred_dict[self.prediction_id] = {
                "valid": phys_prediction["valid"],
                "vehicle_id": obj_key,
                "prediction_type": pred_type_str,
                "t_abs_perception": t_abs_perception,
                "t": [pred.t for pred in phys_prediction["pred"]],
                "x": [pred.x for pred in phys_prediction["pred"]],
                "y": [pred.y for pred in phys_prediction["pred"]],
                "heading": [pred.heading for pred in phys_prediction["pred"]],
            }

            self.log_hist[self.prediction_id] = None
            self.log_obj[self.prediction_id] = (
                obj_key,
                obj_vals["xy_positions"][0],
                t_abs_perception,
            )
            self.log_boundaries[self.prediction_id] = None

            self.prediction_id += 1

        self.remove_from_tracked_objects(remove_list)

    def data_based_prediction(self):
        """Carry out the data based prediction on the remaining objects."""
        # carrying out the prediction with one of the prediction models:
        if self.model_type == "IndyNet":
            (
                pred_dict,
                log_hist,
                log_obj,
                log_boundaries,
                prediction_id,
            ) = self.model_handler.predict(self.tracked_objects, self.prediction_id)
        else:  # MixNet
            (
                pred_dict,
                log_hist,
                log_obj,
                log_boundaries,
                prediction_id,
            ) = self.model_handler.predict(
                obs_storage=self.tracked_objects,
                prediction_id=self.prediction_id,
                physics_pred=self.physics_prediction,
            )

        # updating the prediction dict:
        self.pred_dict.update(pred_dict)

        # updating the log dicts:
        self.log_hist.update(log_hist)
        self.log_obj.update(log_obj)
        self.log_boundaries.update(log_boundaries)

        # updating the prediction counter:
        self.prediction_id = prediction_id

        # Remove keys
        for pred in pred_dict.values():
            del self.tracked_objects[pred["vehicle_id"]]

    def log_pred_data(self, st_time: int):
        """Log data of prediction iteration.

        Args:
            st_time (int): start time in ns.
        """
        self.data_logger.log_pred_data(
            stamp2time(*self.get_clock().now().seconds_nanoseconds()),
            self.log_obj,
            self.log_hist,
            self.log_boundaries,
            self.pred_dict,
            stamp2time(*self.get_clock().now().seconds_nanoseconds()) - st_time,
            log_params=self._log_params,
        )

    def remove_ego_pred(self):
        """Remove ego from prediction dict."""
        for pred_id in self.pred_dict:
            if "ego" in str(self.pred_dict[pred_id]["vehicle_id"]):
                del self.pred_dict[pred_id]
                self.main_logger.info("remove ego from pred_dict")
                break

    def remove_from_tracked_objects(self, remove_list: list):
        """Remove object from tracked objects.

        functions is called after object is processed.

        Args:
            remove_list (list): list with object-IDs to be removed.
        """
        # Remove keys
        for key in remove_list:
            del self.tracked_objects[key]

    def log_non_data_based_obj(self, obj_key: int, obj_vals: dict):
        """Log non-data-based prediction.

        Args:
            obj_key (int): object ID
            obj_vals (dict): values of tracked object (position, time, etc.)

        """
        self.log_hist[self.prediction_id] = None
        self.log_obj[self.prediction_id] = (
            obj_key,
            obj_vals["xy_positions"][0],
            obj_vals["t_abs_perception"],
        )
        self.log_boundaries[self.prediction_id] = None
        self.prediction_id += 1


def main(args=None):
    """Execute mix_net node."""
    rclpy.init(args=args)

    mix_net_node = MixNetNode()

    rclpy.spin(mix_net_node)

    signal.signal(signal.SIGINT, mix_net_node.shutdown_ros)

    rclpy.shutdown()

    mix_net_node.destroy_node()


if __name__ == "__main__":
    main()
