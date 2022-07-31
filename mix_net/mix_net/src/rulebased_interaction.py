import os
import sys
import numpy as np

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from utils.geometry import (
    check_collision,
    check_collision_rect,
    get_heading,
)
from utils.sort import quicksort, sort_collisions
from src.overtake_decision import OvertakeDecisionMaker


class RuleBasedInteraction(object):
    def __init__(self, all_params, logger):
        self.params = all_params["INTERACTION_PARAMS"]
        self.params["track_path"] = all_params["track_path"]
        self.params["sampling_frequency"] = all_params["MODEL_PARAMS"][
            "sampling_frequency"
        ]
        self.logger = logger

        self.define_safety_distances()

        # Collision Function
        if self.params["collision_check"] == "euclidean":
            self.collision_fn = self.detect_collisions
        else:
            self.collision_fn = self.detect_collisions_rectangle

        # Overtake Decision Maker
        self.ot_decision_maker = OvertakeDecisionMaker(
            track_path=self.params["track_path"]
        )

    def define_safety_distances(self):
        """Define safety distance in lateral and longitudinal direction."""
        self.params["lat_safety_dist_m"] = (
            self.params["lat_veh_half_m"] + self.params["lat_safety_m"]
        )
        self.params["long_safety_dist_m"] = (
            self.params["long_veh_half_m"] + self.params["long_safety_m"]
        )
        self.params["max_overlap_dist_m"] = np.sqrt(
            self.params["lat_veh_half_m"] ** 2 + self.params["long_veh_half_m"] ** 2
        ) + np.sqrt(
            self.params["lat_safety_dist_m"] ** 2
            + self.params["long_safety_dist_m"] ** 2
        )

    def apply_to_predictions(self, pred_dict):
        """Modify predictions that collide with each other according to the race rules."""
        # Loop n-times over according to params
        self.pred_dict = pred_dict
        for iteration in range(self.params["no_iterations"]):

            # Create a list for collisions
            collision_list = self.collision_fn()

            # If there is a collision
            if len(collision_list) > 0:
                # Get a priority for all vehicles involved in collisions
                involved_predictions = [
                    collision["pred_ids"] for collision in collision_list
                ]
                involved_predictions_set = set(
                    [item for sublist in involved_predictions for item in sublist]
                )

                # Get the priority list
                priority_list = self.get_race_order(involved_predictions_set)  # noqa

                # Sort collision list so that the one with higehst prio is handled first
                sorted_collision_list = sort_collisions(collision_list, priority_list)

                # Avoid the collisions
                for collision in sorted_collision_list:
                    # Priority on ego vehicle in last iteration if activated
                    if (
                        self.params["priority_on_ego"]
                        and (iteration + 1) == self.params["no_iterations"]
                    ):
                        # Only adjust those predictions where ego is involved as leading vehicle
                        if (
                            self.pred_dict[collision["leading_pred_id"]]["vehicle_id"]
                            != "ego"
                        ):
                            continue

                    # Dont adjust static predictions
                    if (
                        self.pred_dict[collision["following_pred_id"]][
                            "prediction_type"
                        ]
                        != "static"
                    ):
                        self.adjust_prediction(collision)
                    else:
                        self.logger.info(
                            "Iteration {}: Prediction with IDs {} (following) and {} (leading) was not adjusted. Following object is static".format(
                                iteration,
                                collision["following_pred_id"],
                                collision["leading_pred_id"],
                            )
                        )

        return self.pred_dict

    def detect_collisions(self):
        """Detect collisions between all trajectory predictions based on euclidean distance.

        Returns:
            [list]: [list of dicts where every dict describes a collision with keys
                     * pred_ids
                     * time_step]

        """
        collision_list = []
        # Check all predicted trajectories for collisions
        pred_keys = [key for key, value in self.pred_dict.items() if value["valid"]]
        # Combine every possible pair only once
        for count, pred_id_1 in enumerate(pred_keys):
            for pred_id_2 in pred_keys[count + 1 :]:
                positions_1 = np.vstack(
                    (self.pred_dict[pred_id_1]["x"], self.pred_dict[pred_id_1]["y"])
                ).T
                positions_2 = np.vstack(
                    (self.pred_dict[pred_id_2]["x"], self.pred_dict[pred_id_2]["y"])
                ).T

                for ts in range(min(positions_1.shape[0], positions_2.shape[0])):
                    # Check every point for collisions
                    if check_collision(
                        positions_1[ts, :],
                        positions_2[ts, :],
                        approx_radius=self.params["approx_radius"],
                    ):
                        # Collision detected
                        collision_dict = {
                            "pred_ids": [pred_id_1, pred_id_2],
                            "time_step": ts,
                        }
                        collision_list.append(collision_dict)
                        self.logger.info(
                            "Collision detected between predictions with IDs {} ({}) and {} ({}) at timestep {}".format(
                                pred_id_1,
                                positions_1[ts],
                                pred_id_2,
                                positions_2[ts],
                                ts,
                            )
                        )

                        # Only temporal first collision needed
                        break

        return collision_list

    def detect_collisions_rectangle(self):
        """Detect rectangle collisions between all trajectory predictions.

        Returns:
            [list]: [list of dicts where every dict describes a collision with keys
                     * pred_ids
                     * time_step]

        """
        collision_list = []
        collided_list = []
        # Check all predicted trajectories for collisions
        pred_keys = [key for key, value in self.pred_dict.items() if value["valid"]]
        # Combine every possible pair only once
        # Order is important as only pred_id_1 is assigned with safety box
        for pred_id_1 in pred_keys:
            for pred_id_2 in pred_keys:
                if pred_id_1 == pred_id_2:
                    continue

                # Skip if collison was already detected
                if sorted([pred_id_1, pred_id_2]) in collided_list:
                    continue

                poses_1 = np.stack(
                    (
                        self.pred_dict[pred_id_1]["x"],
                        self.pred_dict[pred_id_1]["y"],
                        self.pred_dict[pred_id_1]["heading"],
                    ),
                    axis=1,
                )
                poses_2 = np.stack(
                    (
                        self.pred_dict[pred_id_2]["x"],
                        self.pred_dict[pred_id_2]["y"],
                        self.pred_dict[pred_id_2]["heading"],
                    ),
                    axis=1,
                )

                for ts in range(min(poses_1.shape[0], poses_2.shape[0])):
                    # Check every point for collisions
                    if (
                        np.linalg.norm(poses_1[ts, :2] - poses_1[ts, :2])
                        > self.params["max_overlap_dist_m"]
                    ):
                        continue

                    bool_rectangle = check_collision_rect(
                        poses_1[ts, :],
                        poses_2[ts, :],
                        lat_veh_half_m=self.params["lat_veh_half_m"],
                        long_veh_half_m=self.params["long_veh_half_m"],
                        lat_safety_dist_m=self.params["lat_safety_dist_m"],
                        long_safety_dist_m=self.params["long_safety_dist_m"],
                    )

                    if bool_rectangle:
                        # Collision detected
                        collided_list.append(sorted([pred_id_1, pred_id_2]))
                        collision_dict = {
                            "pred_ids": [pred_id_1, pred_id_2],
                            "time_step": ts,
                        }
                        collision_list.append(collision_dict)
                        self.logger.info(
                            "Collision detected between predictions with IDs "
                            "{} ({}) and {} ({}) at timestep {}".format(
                                pred_id_1,
                                poses_1[ts],
                                pred_id_2,
                                poses_2[ts],
                                ts,
                            )
                        )

                        # Only temporal first collision needed
                        break

        return collision_list

    def compare_positions(self, id1, id2):
        """Return True if id 1 is in front of id2, else return false."""
        # Call predictions for ids
        pred_1_t0 = np.array([self.pred_dict[id1]["x"][0], self.pred_dict[id1]["y"][0]])
        pred_2_t0 = np.array([self.pred_dict[id2]["x"][0], self.pred_dict[id2]["y"][0]])
        pred_2_t1 = np.array([self.pred_dict[id2]["x"][1], self.pred_dict[id2]["y"][1]])

        follower_vect = pred_2_t1 - pred_2_t0
        follower_to_leader = pred_1_t0 - pred_2_t0

        # checking the scalar product, to determine whether id1 is
        # in front of id2:
        return (follower_vect @ follower_to_leader) > 0

    def get_race_order(self, prediction_ids):
        """Order a set of prediction IDs by their race order (position in the race).

        Args:
            prediction_ids ([set]): [A set of prediction ids]
        """
        prediction_ids = list(prediction_ids)

        sorted_prediction_ids = quicksort(prediction_ids, self.compare_positions)

        return sorted_prediction_ids

    def adjust_prediction(self, collision):
        """Adjust the prediction according to a given collision.

        Args:
            collision ([dict]): [collsion dict]
        """
        following_pred = self.pred_dict[collision["following_pred_id"]]
        leading_pred = self.pred_dict[collision["leading_pred_id"]]
        coll_ts = int(collision["time_step"])

        # Get the overtake decision (left or right):
        ot_dec = self.ot_decision_maker.get_overtake_direction(
            pos_leading=np.array([leading_pred["x"][0], leading_pred["y"][0]]),
            pos_following=np.array([following_pred["x"][0], following_pred["y"][0]]),
            overtake_margin=self.params["lat_overtake_dist"],
        )

        # getting the adjusted prediction:
        following_pred_adj, max_adjustment = self.ot_decision_maker.adjust_prediction(
            leading_pred=np.vstack((leading_pred["x"], leading_pred["y"])),
            following_pred=np.vstack((following_pred["x"], following_pred["y"])),
            ts=coll_ts,
            decision=ot_dec,
            approx_radius=self.params["approx_radius"],
            ot_length_ts=int(
                self.params["lanechange_time"] * self.params["sampling_frequency"]
            ),
            min_overtake_dist=self.params["lat_overtake_dist"],
            logger=self.logger,
        )

        # for logging:
        # adj_vect = following_pred_adj[:, coll_ts] - np.array(
        #     [following_pred["x"][coll_ts], following_pred["y"][coll_ts]]
        # )
        # adj_dist = np.linalg.norm(adj_vect)

        # Adjust the predictions in pred_dict
        self.pred_dict[collision["following_pred_id"]]["x"] = following_pred_adj[0, :]
        self.pred_dict[collision["following_pred_id"]]["y"] = following_pred_adj[1, :]

        # recalculate the heading:
        self.pred_dict[collision["following_pred_id"]]["heading"] = get_heading(
            following_pred_adj.T
        )

        # Log message
        self.logger.info(
            "Prediction with ID {} adjusted to {} maneuver with lat distance of {:.2f}".format(
                collision["following_pred_id"],
                ot_dec,
                max_adjustment,
            )
        )
