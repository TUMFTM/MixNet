import sys
import os
import time
import numpy as np
from scipy import interpolate
from copy import copy

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from utils.map_utils import get_track_paths, get_track_kinematics, get_arc
from utils.geometry import calc_distance
from utils.map_utils import line_intersection


class BoundaryGenerator:
    def __init__(self, params: dict):
        self.dist = int(params["MODEL_PARAMS"]["dist"])
        self.view = int(params["MODEL_PARAMS"]["view"])

        self.numel = int(self.view / self.dist) + 1

        track_path = params["track_path"]

        (
            self.arc_center,
            self.center_line,
            self.bound_right_xy,
            self.bound_left_xy,
            track_width,
        ) = get_track_paths(track_path, bool_track_width=True)

        # for bool_outofbounds to provide yaw_angle
        center_yaw = get_track_kinematics(
            path_name="centerline",
            track_path=track_path,
        )[3]

        # interpolate boundaries function
        track_kinematics = np.hstack(
            [
                self.bound_right_xy,
                self.bound_left_xy,
                np.expand_dims(track_width, 1),
                np.expand_dims(center_yaw, 1),
            ]
        )
        self.fn_interpol_tracks = interpolate.interp1d(
            self.arc_center, track_kinematics, axis=0
        )

        self.max_arc = float(self.arc_center[-1])
        self.num_center = len(track_kinematics)
        self.vec_center = np.array(
            [
                self.center_line[(i + 1) % self.num_center, :] - self.center_line[i, :]
                for i in range(self.num_center)
            ]
        )
        self.center_line_batch = self.center_line.reshape(1, -1, 2)

        # variables to get_boundaries
        self.center_line_batch = self.center_line.reshape(1, -1, 2)

        # Create CLEARLY aranged dicts for boundaries
        right_arc = get_arc(self.bound_right_xy)
        self.boundary_right = {
            "x": self.bound_right_xy[:, 0],
            "y": self.bound_right_xy[:, 1],
            "xy": self.bound_right_xy,
            "interp_xy": interpolate.interp1d(
                right_arc, self.bound_right_xy, axis=0
            ),  # get interpolated x,y of arc
            "arc": right_arc,
            "interp_arc": interpolate.interp2d(
                self.bound_right_xy[:, 0],
                self.bound_right_xy[:, 1],
                right_arc,
                copy=False,
            ),  # get interpolated arc out of x,y
        }

        left_arc = get_arc(self.bound_left_xy)
        self.boundary_left = {
            "x": self.bound_left_xy[:, 0],
            "y": self.bound_left_xy[:, 1],
            "xy": self.bound_left_xy,
            "interp_xy": interpolate.interp1d(
                left_arc, self.bound_left_xy, axis=0
            ),  # get interpolated x,y of arc
            "arc": left_arc,
            "interp_arc": interpolate.interp2d(
                self.bound_left_xy[:, 0], self.bound_left_xy[:, 1], left_arc, copy=False
            ),  # get interpolated arc out of x,y
        }

    def get_boundaries_single(self, translation):
        """Gets boundaries for a position stored in "translation".

        args:
            translations: [np.array with shape=(2,)]

        returns:
            pillar_left: [np.array with shape=(num_of_bound_points, 2)]
            pillar_right: [np.array with shape=(num_of_bound_points, 2)]
        """
        indi_min = (
            np.argmin(np.linalg.norm(self.center_line - translation, axis=1))
            % self.num_center
        )

        v_center = self.vec_center[indi_min]
        v_veh = translation - self.center_line[indi_min]

        delta_arc = np.linalg.norm(
            np.dot(v_center, v_veh) / np.dot(v_center, v_center) * v_center
        )
        arc_start = self.arc_center[indi_min] + delta_arc

        arc_pols = (arc_start + np.linspace(0, self.view, self.numel)) % self.max_arc

        track_bounds_pol = self.fn_interpol_tracks(arc_pols)

        pillar_left = track_bounds_pol[:, 2:4]
        pillar_right = track_bounds_pol[:, :2]

        return pillar_left, pillar_right

    def get_boundaries(self, translations):
        """Gets boundaries for a batch of positions in "translation".

        args:
            translations: [np.array with shape=(batch_size, 2)]

        returns:
            pillars_left: [np.array with shape=(batch_size, num_of_bound_points, 2)]
            pillars_right: [np.array with shape=(batch_size, num_of_bound_points, 2)]
        """
        deltas = self.center_line_batch - translations.reshape(-1, 1, 2)
        indis_min = np.argmin(np.einsum("ijk,ijk->ij", deltas, deltas), axis=1).tolist()

        v_centers = self.vec_center[indis_min, :].T
        v_vehs = translations - self.center_line_batch[0, indis_min, :]

        delta_arcs = np.linalg.norm(
            np.einsum("ji,ij->i", v_centers, v_vehs) / sum(v_centers**2) * v_centers,
            axis=0,
        )

        arc_starts = self.arc_center[indis_min] + delta_arcs
        arc_pols = (
            arc_starts.reshape(-1, 1) + np.linspace(0, self.view, self.numel)
        ) % self.max_arc
        track_bounds_pols = self.fn_interpol_tracks(arc_pols)

        pillars_left = track_bounds_pols[:, :, 2:4]
        pillars_right = track_bounds_pols[:, :, :2]

        return pillars_left, pillars_right

    def get_bounds_between_points(self, p1, p2):
        """Returns the boundaries that is between the 2 points p1 and p2.
        p1 and p2 are points on the track, so first the nearest points of the
        boundaries to them are searched and then the section between
        them is returned.
        """
        # index of the closest point to the first point:
        deltas = self.center_line[:, :2] - p1
        dist_sqrt = np.sum((deltas**2), axis=1)
        i0 = np.argmin(dist_sqrt)

        # index of the closest point to the first point:
        deltas = self.center_line[:, :2] - p2
        dist_sqrt = np.sum((deltas**2), axis=1)
        i1 = np.argmin(dist_sqrt) + 1

        if i0 < i1:
            return self.bound_left_xy[i0:i1, :2], self.bound_right_xy[i0:i1, :2]
        else:
            left = np.vstack((self.bound_left_xy[i0:, :2], self.bound_left_xy[:i1, :2]))
            right = np.vstack(
                (self.bound_right_xy[i0:, :2], self.bound_right_xy[:i1, :2])
            )

            return left, right

    def track_fn_batch(self, translation, bounds_buffer=0.0):
        """
        checks if the detected objects are on track or a not relevant object outside from track (e.g.tribune)
        Parameters
        _ _ _ _ _
        translation: np.array nx2 (#n detected objects)
            x and y position of the n objects in global coordinates
        Returns
        _ _ _ _
        bool_outofbounds: bool
            true: yes object is out of defined bounds / false: object is important for tracking
        rel_positions: array
            relative position on track: value between 0.0 and 1.0
        yaw_angle: array
            yaw_angle calculated from relative positoin with offline data from centerline
        """

        trla = translation.reshape(-1, 1, 2)
        deltas = self.center_line_batch - trla

        self.indis_min = np.argmin(
            np.einsum("ijk,ijk->ij", deltas, deltas), axis=1
        ).tolist()

        v_centers = self.vec_center[self.indis_min, :].T
        v_vehs = translation - self.center_line_batch[0, self.indis_min, :]

        delta_arcs = np.linalg.norm(
            np.einsum("ji,ij->i", v_centers, v_vehs) / sum(v_centers**2) * v_centers,
            axis=0,
        )
        arc_starts = (self.arc_center[self.indis_min] + delta_arcs) % self.max_arc
        rel_positions = arc_starts / self.max_arc

        track_bounds_pols = self.fn_interpol_tracks(arc_starts)
        self.dists_right = np.linalg.norm(
            translation - track_bounds_pols[:, :2], axis=1
        )
        self.dists_left = np.linalg.norm(
            translation - track_bounds_pols[:, 2:4], axis=1
        )
        track_widths = track_bounds_pols[:, 4]
        bool_outofbounds = np.max(
            np.stack(
                [
                    self.dists_right - track_widths > bounds_buffer,
                    self.dists_left - track_widths > bounds_buffer,
                ]
            ),
            axis=0,
        )

        return bool_outofbounds, rel_positions

    def project_to_track(self, point_array):
        """Project a point to the track limits if outside the track by keeping the velocity profile.

        Args:
            point_array ([np.array]): [n_points x 2]

        Returns:
            [np.array]: [[n_points x 2] modified so that all points are within or at the boundary of the race track]
        """

        bool_out_of_bounds_list, _ = self.track_fn_batch(point_array)

        # Calc distance profile of point array
        distances = [
            calc_distance(point_array[i], point_array[i + 1])
            for i in range(len(point_array) - 1)
        ]

        for idx, p in enumerate(bool_out_of_bounds_list):
            # first time not in track

            if p:
                if self.dists_left[idx] > self.dists_right[idx]:
                    matching_bound = self.boundary_right
                else:
                    matching_bound = self.boundary_left

                if not bool_out_of_bounds_list[idx - 1]:

                    point_traj_inside = point_array[idx - 1]
                    point_traj_outside = point_array[idx]

                    point_track_before = matching_bound["xy"][self.indis_min[idx] - 1]

                    # transition end of track to beginning of track (closing the loop)
                    if (self.indis_min[idx] + 1) >= len(matching_bound["xy"]):
                        point_track_after = matching_bound["xy"][
                            1
                        ]  # not 0 to make sure its the point after
                    else:
                        point_track_after = matching_bound["xy"][
                            self.indis_min[idx] + 1
                        ]

                    interection_point = line_intersection(
                        [point_traj_inside, point_traj_outside],
                        [point_track_before, point_track_after],
                    )

                    arc_dist = calc_distance(interection_point, point_traj_outside)

                    intersection_arc = matching_bound["interp_arc"](
                        interection_point[0], interection_point[1]
                    )

                    current_arc = intersection_arc + arc_dist

                else:
                    if idx == 0:
                        current_arc = matching_bound["interp_arc"](
                            point_array[idx][0], point_array[idx][1]
                        )
                    arc_dist = distances[idx - 1]
                    current_arc += arc_dist

                new_xy = matching_bound["interp_xy"](current_arc)

                point_array[idx] = new_xy

        return point_array


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    translations = np.array(
        [
            [-2.57541844e02, -9.79702849e02],
            [-3.17327297e02, -1.37190465e03],
            [-3.10201453e02, -1.77754611e03],
            [-2.96246351e02, -2.18263374e03],
            [9.51867784e-02, -2.40865280e03],
            [3.72751888e02, -2.29514492e03],
            [4.35005805e02, -1.90348543e03],
            [4.27879086e02, -1.49784448e03],
            [4.12282239e02, -1.09290184e03],
            [1.14699047e02, -8.66694173e02],
        ]
    )

    rotations = [
        -2.3231418365509726,
        -1.5529399329205043,
        -1.5534939268676697,
        -1.4300325681183468,
        -0.05828855271554859,
        0.8121244144497923,
        1.588674215884788,
        1.5880549012751173,
        1.7285836415048585,
        3.0706985522964385,
    ]

    n_cals = 100

    boundary_generator = BoundaryGenerator(view=400, dist=20)
    left_bound_list = []
    right_bound_list = []
    for trans, rot in zip(list(translations), rotations):

        left_bound, right_bound = boundary_generator.get_boundaries_single(trans, rot)
        left_bound_list.append(left_bound)
        right_bound_list.append(right_bound)
    left_bound2, right_bound2 = boundary_generator.get_boundaries(
        translations, rotations
    )
    try:
        for j in range(len(left_bound_list)):
            np.linalg.norm(left_bound_list[j] - left_bound2[j]) + np.linalg.norm(
                right_bound_list[j] - right_bound2[j]
            ) < 1e-2
    except ValueError:
        print("invalid boundaries")

    tic = time.time()
    left_bound_list = []
    right_bound_list = []
    for k in range(n_cals):
        for trans, rot in zip(list(translations), rotations):

            left_bound, right_bound = boundary_generator.get_boundaries_single(
                trans, rot
            )
            left_bound_list.append(left_bound)
            right_bound_list.append(right_bound)

    toc1 = time.time() - tic
    print(
        "Elapsed time for {} x {} objects:\t{:.3f} ms".format(
            n_cals, len(rotations), toc1 * 1000
        )
    )

    tic = time.time()
    for k in range(n_cals):
        left_bound2, right_bound2 = boundary_generator.get_boundaries(
            translations, rotations
        )

    toc = time.time() - tic
    print(
        "Elapsed time for {} x {} objects:\t{:.3f} ms".format(
            n_cals, len(rotations), toc * 1000
        )
    )
    print(
        "Relative change of calculation time:\t{:.1f} %".format(
            (toc - toc1) / toc1 * 100
        )
    )

    plt.plot(
        boundary_generator.bound_left_xy[:, 0],
        boundary_generator.bound_left_xy[:, 1],
    )
    plt.plot(
        boundary_generator.bound_right_xy[:, 0],
        boundary_generator.bound_right_xy[:, 1],
    )

    plt.axis("equal")

    sample_point = [
        [float(a), float(b)] for a, b in zip(range(420, 500), range(-1480, -1400))
    ]

    sample_point2 = [
        [float(a), float(b)] for a, b in zip(range(500, 400, -1), range(-1400, -1300))
    ]

    sample_point_array = np.array(sample_point + sample_point2)
    sample_point_array_cpy = copy(sample_point_array)

    st_time = time.time()
    point_array = boundary_generator.project_to_track(sample_point_array)
    print("Time for projecting outliers to track limits: ", time.time() - st_time)

    plt.scatter(point_array[:, 0], point_array[:, 1])
    plt.scatter(sample_point_array_cpy[:, 0], sample_point_array_cpy[:, 1])

    plt.show()
