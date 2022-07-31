# Standard imports
import sys
import os
import time
import json

# Third party imports
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.msg import SetParametersResult

import numpy as np
import torch

# Custom imports
from tum_msgs.msg import (
    TUMModuleStatus,
    TUMPredictedObjects,
    TUMPredictedObject,
    TUMPredState,
    TUMTrackedObjects,
)

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from utils.setup_helpers import setup_logger, get_params
from utils.logging_helper import log_param_dict


class ROS2Handler(Node):
    def __init__(self, path_dict: dict):
        """Intialize ROS 2 nodes."""
        super().__init__("mix_net_node")

        # Set health stats
        self.watchdog = 0
        self.module_state = 10
        self.control_module_state = -1
        self.control_is_connected = False

        # Set history depth for qos
        history_depth = 1

        # Declare parameters
        self.declare_parameter(
            name="frequency",
            descriptor=ParameterDescriptor(dynamic_typing=True),
            value=20.0,
        )
        self.declare_parameter(
            name="track",
            value="LVMS",
        ),
        self.declare_parameter(
            name="ci",
            value=False,
        )
        self.declare_parameter(
            name="use_cuda",
            value=False,
        )
        # Set paramters callback function
        self.add_on_set_parameters_callback(self._set_parameters_callback)

        # Initialize parameters
        self.params = {
            "frequency": float(self.get_parameter("frequency").value),
            "track": self.get_parameter("track").value,
            "use_sim_time": self.get_parameter("use_sim_time").value,
            "ci": self.get_parameter("ci").value,
            "use_cuda": self.get_parameter("use_cuda").value,
        }

        if not torch.cuda.is_available():
            self.params["use_cuda"] = False

        self.__is_dynamic_typing = ["frequency"]

        # Setup logging
        self.main_logger, self.data_logger = setup_logger(
            path_dict=path_dict,
        )

        # Get params from config folder
        self.params.update(get_params(path_dict=path_dict))

        self.__init_parameterization(path_dict=path_dict)

        log_param_dict(param_dict=self.params, logger=self.main_logger)

        # Module state
        self.module_state = 10
        self.main_logger.info(f"STATE: mod_prediction state = {self.module_state}")

        # PUBLISHER NODES #
        # ROS2 Predicted Output to Local Planner
        self.predicted_objects_publisher = self.create_publisher(
            msg_type=TUMPredictedObjects,
            topic="/mod_prediction/PredictedObjects",
            qos_profile=history_depth,
        )

        # ROS2 Watchdog to Control
        self.watchdog_state_to_controller = self.create_publisher(
            msg_type=TUMModuleStatus,
            topic="/mod_prediction/status_prediction",
            qos_profile=history_depth,
        )
        self.watchdog_state_to_controller_msg = TUMModuleStatus()

        # SUBSCRIBER NODES #
        # ROS2 Control-WatchDog to Prediction
        self.watchdog_controller_state = self.create_subscription(
            msg_type=TUMModuleStatus,
            topic="/mod_control/status_control",
            callback=self.receive_software_state,
            qos_profile=history_depth,
        )
        self.watchdog_controller_state_msg = None

        # ROS2 Tracked Objects
        self.tracked_objects_subscriber = self.create_subscription(
            msg_type=TUMTrackedObjects,
            topic="/mod_tracking/TrackedObjects",
            callback=self.receive_tracked_objects,
            qos_profile=history_depth,
        )
        self.tracked_objects = None

        self.physcis_prediction_subscriber = self.create_subscription(
            msg_type=TUMPredictedObjects,
            topic="/mod_tracking/PredictedObjects",
            callback=self.receive_physics_prediction,
            qos_profile=history_depth,
        )
        self.physics_prediction = None

        # Setup variables
        self.prediction_id = 0
        self.pred_dict = None
        self.log_hist = {}
        self.log_obj = {}
        self.log_boundaries = {}
        self.deleted_histories = {}

        time_period = 1.0 / self.params["frequency"]
        self.timer = self.create_timer(time_period, self.timer_callback)

        # Ready
        self.main_logger.info("--- Listening for objects lists ---")

    def __init_parameterization(self, path_dict: dict):
        """Initialize Objects Class."""
        # prediction function parameters
        self.model_type = self.params["MODEL_PARAMS"]["model_type"]
        self.sampling_frequency = self.params["MODEL_PARAMS"]["sampling_frequency"]
        self.data_min_obs_length = self.params["MODEL_PARAMS"]["data_min_obs_length"]
        self.stat_vel_threshhold = self.params["MODEL_PARAMS"]["stat_vel_threshhold"]
        self.stat_prediction_horizon = self.params["MODEL_PARAMS"][
            "stat_prediction_horizon"
        ]
        self.data_max_acceleration = self.params["MODEL_PARAMS"][
            "data_max_acceleration"
        ]
        self.data_min_velocity = self.params["MODEL_PARAMS"]["data_min_velocity"]

        # Net params
        if "indy" in self.model_type.lower():
            net_key = "indy_net"
        else:
            net_key = "mix_net"
        self.add_net_param_paths(net_key=net_key, path_dict=path_dict)

        # Logging parameters
        self._log_params = self.params["LOGGING_PARAMS"]

    def add_net_param_paths(self, net_key: str, path_dict: dict):
        """Load params and weights for nets."""
        if net_key not in ("mix_net", "indy_net"):
            raise NotImplementedError("Name '{}' is not supported".format(net_key))

        # Create string snippets
        header = net_key.upper() + "_PARAMS"
        path_str = net_key + "_path"
        param_str = net_key + "_params"
        weights_str = net_key + "_weights"

        # Store name of model
        self.params["INF_MODEL_PARAMS"] = {
            "name": net_key,
            "param_file": os.path.join(
                path_dict[path_str], self.params["FILE_NAMES"][param_str]
            ),
        }

        # Load net params
        with open(self.params["INF_MODEL_PARAMS"]["param_file"], "r") as fp:
            self.params["INF_MODEL_PARAMS"].update(json.load(fp))

        # Overwrite cuda usage from ros2 params
        self.params["INF_MODEL_PARAMS"]["use_cuda"] = self.params["use_cuda"]

        # Add offline values
        if header in self.params:
            self.params["INF_MODEL_PARAMS"].update(self.params[header])

        # ADd net params path
        self.params["INF_MODEL_PARAMS"]["param_file_path"] = os.path.join(
            path_dict[path_str],
            self.params["FILE_NAMES"][param_str],
        )
        self.check_path_exists(self.params["INF_MODEL_PARAMS"]["param_file_path"])

        # Add net weights path
        self.params["INF_MODEL_PARAMS"]["weights_file_path"] = os.path.join(
            path_dict[path_str],
            self.params["FILE_NAMES"][weights_str],
        )
        self.check_path_exists(self.params["INF_MODEL_PARAMS"]["weights_file_path"])

        # Add map path
        self.params["INF_MODEL_PARAMS"]["map_file_path"] = self.params["track_path"]
        self.params["MIX_NET_PARAMS"]["map_file_path"] = self.params["track_path"]

    @staticmethod
    def check_path_exists(file_path: str):
        """Check if path exists."""
        assert os.path.exists(file_path), "{} does not exist! ({})".format(
            os.path.basename(file_path),
            file_path,
        )

    def _set_parameters_callback(self, params):
        """
        Callback function to set parameters.

        Arguments:
            params: Parameters to set, <list>.

        Returns:
            Parameter set result (success), <SetParametersResult>.
        """

        for param in params:
            if param.name in self.params:

                if param.name == rclpy.time_source.USE_SIM_TIME_NAME:
                    self._time_source._on_parameter_event(
                        [Parameter("use_sim_time", Parameter.Type.BOOL, param.value)]
                    )
                    self.params[param.name] = param.value
                elif param.name in self.__is_dynamic_typing:
                    self.params[param.name] = param.value
                elif isinstance(self.params[param.name], float):
                    self.params[param.name] = float(param.value)
                elif isinstance(self.params[param.name], int):
                    self.params[param.name] = int(param.value)
                elif isinstance(self.params[param.name], bool):
                    self.params[param.name] = bool(param.value)
                elif isinstance(self.params[param.name], str):
                    self.params[param.name] = str(param.value)

                self.main_logger.info(
                    "PARAMETER SET: {} = {}, type = {}".format(
                        param.name,
                        self.params[param.name],
                        type(self.params[param.name]),
                    )
                )
            else:
                return SetParametersResult(successful=False)

        return SetParametersResult(successful=True)

    def receive_software_state(self, controller_state):
        """Receive overall software state from mod_control.

        udp interface to mod_control
        message: [watchdog:none:uint8, status:none:uint8]

        data_size = 2 * 4 bytes (uint)
        """
        # Set connection status to true
        if not self.control_is_connected:
            self.main_logger.info("CONNECTED: connected to mod_control")
            self.control_is_connected = True

        # Set overall software state
        if controller_state.status != self.control_module_state:
            self.main_logger.info(
                "STATE: mod_control state = {}".format(controller_state.status)
            )
            self.control_module_state = controller_state.status

    def receive_tracked_objects(self, msg):
        """receive_tracked_objects _summary_

        _extended_summary_

        Args:
            msg (_type_): _description_
        """
        if msg.empty:
            return

        self.tracked_objects = {
            track_obj.object_id: {
                "vehicle_id": track_obj.object_id,
                "t_abs_perception": track_obj.t_abs_perception,
                "xy_positions": np.array(
                    [
                        (obj_track.x, obj_track.y)
                        for j, obj_track in enumerate(track_obj.track)
                        if j < self.params["OBJ_HANDLING_PARAMS"]["max_obs_length"]
                    ]
                ),
                "yaw": track_obj.track[0].yaw,
                "v": track_obj.track[0].v,
            }
            for track_obj in msg.objects
        }

    def receive_physics_prediction(self, msg):
        """receive_physics_prediction _summary_

        _extended_summary_

        Args:
            msg (_type_): _description_
        """
        if msg.empty:
            return

        self.physics_prediction = {
            pred_obj.vehicle_id: {
                "vehicle_id": pred_obj.vehicle_id,
                "t_abs_perception": pred_obj.t_abs_perception,
                "pred": pred_obj.pred,
                "prediction_type": pred_obj.prediction_type,
                "valid": pred_obj.valid,
            }
            for pred_obj in msg.objects
        }

    def shutdown_ros(self, sig=None, frame=None, ci_test=False):
        """Close all ros nodes."""
        self.main_logger.info("Closing ROS Nodes...")
        self.destroy_node()
        rclpy.shutdown()
        time.sleep(0.5)
        self.main_logger.info("Shutdown complete!")
        if ci_test:
            print(
                "target_state = {}, module_state = {}".format(
                    ci_test, self.module_state
                )
            )
            if self.module_state != ci_test:
                print("target_state != module_state")
                sys.exit(1)

        sys.exit(0)

    def send_prediction(self):
        """ROS sender to send prediction-dict to mod_local_planner."""
        predicted_objects = TUMPredictedObjects()

        # send prediction
        if self.pred_dict is not None:
            predicted_objects.empty = not bool(self.pred_dict)

            id_list = [*self.pred_dict.keys()]
            for num_ids in id_list:
                predicted_object = TUMPredictedObject()
                predicted_object.prediction_id = int(num_ids)
                predicted_object.valid = bool(self.pred_dict[num_ids]["valid"])
                predicted_object.vehicle_id = str(self.pred_dict[num_ids]["vehicle_id"])
                predicted_object.prediction_type = str(
                    self.pred_dict[num_ids]["prediction_type"]
                )
                predicted_object.t_abs_perception = int(
                    self.pred_dict[num_ids]["t_abs_perception"]
                )

                if self.pred_dict[num_ids]["valid"]:
                    for num_preds in range(0, len(self.pred_dict[num_ids]["t"])):
                        predstate = TUMPredState()
                        predstate.t = self.pred_dict[num_ids]["t"][num_preds]
                        predstate.x = self.pred_dict[num_ids]["x"][num_preds]
                        predstate.y = self.pred_dict[num_ids]["y"][num_preds]
                        predstate.heading = self.pred_dict[num_ids]["heading"][
                            num_preds
                        ]
                        predicted_object.pred.append(predstate)
                else:
                    predstate = TUMPredState()
                    predstate.t = self.pred_dict[num_ids]["t"]
                    predstate.x = self.pred_dict[num_ids]["x"]
                    predstate.y = self.pred_dict[num_ids]["y"]
                    predstate.heading = self.pred_dict[num_ids]["heading"]
                    predicted_object.pred.append(predstate)

                predicted_objects.objects.append(predicted_object)

        # publish topic
        self.predicted_objects_publisher.publish(predicted_objects)

    def send_watchdog_state(self):
        """Ros sender to send watchdog 0..255 and state to mod_control."""
        self.watchdog = (self.watchdog + 1) % 255
        self.watchdog_state_to_controller_msg.watchdog = self.watchdog
        self.watchdog_state_to_controller_msg.status = self.module_state
        self.watchdog_state_to_controller.publish(self.watchdog_state_to_controller_msg)
        self.send_watchdog_time = time.time()
