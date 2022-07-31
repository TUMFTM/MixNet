import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon


def pi_range(yaw):
    """Clip yaw to (-pi, +pi]."""
    if yaw <= -np.pi:
        yaw += 2 * np.pi
        return yaw
    elif yaw > np.pi:
        yaw -= 2 * np.pi
        return yaw
    return yaw


def get_heading(trajectory):
    """Calculates heading angles from x and y coordinates.
    IMPORTANT:
    - Convention says that y-axis has a heading of 0. And 90° is at -x!
    - Out of n x,y coordinates we can only calculate n-1 heading angles. To keep the desired format of length n we repeat the last entry.

    Args:
        trajectory (np.array): The (x, y) points of the trajectory
    """

    # differences between the successive trajectory points.
    delta = trajectory[1:, :] - trajectory[:-1, :]

    # initializing and filling up the heading array:
    heading = np.zeros(trajectory.shape[0])
    heading[:-1] = np.arctan2(-delta[:, 0], delta[:, 1])

    # repeating the last entry to have the sufficient length:
    heading[-1] = heading[-2]

    return heading


def calc_distance(p1, p2):
    """Calculate the distance of two points in x,y

    Args:
        pos1 ([type]): [description]
        pos2 ([type]): [description]
    """
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))


def distance_point_line(point, line):
    line = np.asarray(line)
    deltas = line - point
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.sqrt(np.min(dist_2))


def check_collision(point_1, point_2, approx_radius=3.0):
    """Checks 2 x,y points for collision with approximated shape of a circle with approx_radius

    Args:
        point_1 ([type]): [description]
        point_2 ([type]): [description]
        approx_radius (float, optional): [description]. Defaults to 3.0.
    """

    return calc_distance(point_1, point_2) <= (2 * approx_radius)


def rotate_loc_glob(loc_position, rot_angle):
    """Rotates 2D-array from local to global coordinates.

    Args:
        loc_position (ndarray(dtype=float, shape=(2,)): Local position (x, y).
        rot_angle (float): Rotation angle in rad from local to global coordinates.
            Convention: rot_angle = 0.0 for local x is parallel to global y
                        rot_angle = - pi / 2.0 for local x is parallel to global x (rot_mat = eye(2))

    Return:
        ndarray(dtype=float, shape=(2,): Global position (x, y).
    """
    rot_mat = np.array(
        [
            [-np.sin(rot_angle), -np.cos(rot_angle)],
            [np.cos(rot_angle), -np.sin(rot_angle)],
        ]
    )

    return np.matmul(rot_mat, loc_position)


def check_collision_rect(
    pose_1,
    pose_2,
    lat_veh_half_m,
    long_veh_half_m,
    lat_safety_dist_m,
    long_safety_dist_m,
):
    """Checks if two equal rectangluar objects (l x w) incl. safety distance overlap.

    Args:
        pose_1 ([type]): Pose of object 1 (x, y, yaw).
        pose_2 ([type]): Pose of object 2 (x, y, yaw).
        w_vehicle_m (float): Width of rectangle in m.
        l_vehicle_m (float): Length of rectangle in m.
        lat_safety_m float: lateral safety distance, half width of reactangle.
        long_safety_m float: longitudinal safety distance, half length of reactangle.

    Return:
        bool: True if overlap (intersection area > 0), False otherwise
    """
    # Object 1 with safety distance
    rectangle_1 = np.stack(
        [
            [long_safety_dist_m, -lat_safety_dist_m],
            [long_safety_dist_m, lat_safety_dist_m],
            [-long_safety_dist_m, lat_safety_dist_m],
            [-long_safety_dist_m, -lat_safety_dist_m],
        ],
        axis=1,
    )

    edges_glob_1 = rotate_loc_glob(rectangle_1, pose_1[2]) + np.expand_dims(
        pose_1[:2], axis=1
    )

    # Object 2 without safety distance
    rectangle_2 = np.stack(
        [
            [long_veh_half_m, -lat_veh_half_m],
            [long_veh_half_m, lat_veh_half_m],
            [-long_veh_half_m, lat_veh_half_m],
            [-long_veh_half_m, -lat_veh_half_m],
        ],
        axis=1,
    )
    edges_glob_2 = rotate_loc_glob(rectangle_2, pose_2[2]) + np.expand_dims(
        pose_2[:2], axis=1
    )

    poly_1 = Polygon(edges_glob_1.T)
    poly_2 = Polygon(edges_glob_2.T)
    intersec = poly_1.intersection(poly_2)

    return bool(intersec.area)


def get_v_and_acc_profile(position_list, sampling_freq=10):
    """Return velocity and acceleration profile from a list of positions.

    Args:
        position_list ([list]): [list of positions in x,y]
        sampling_freq ([int]): Sampling frequency of the point list
    """

    # Calc distance profile
    distances = [
        calc_distance(position_list[i], position_list[i + 1])
        for i in range(len(position_list) - 1)
    ]

    # Calc velocity profile
    velocities = [dist * sampling_freq for dist in distances]

    # Calc acceleration profile
    acceleartions = [
        (velocities[i + 1] - velocities[i]) * sampling_freq
        for i in range(len(velocities) - 1)
    ]

    return velocities, acceleartions


def angle_between(v1, v2):
    """Returns the angle in radians between vectors 'v1' and 'v2'::
    args:
        v1: np.array((N, 2)) or np.array(2,)
        v2: np.array((2,))

    convention: "How much should I rotate v1 to get v2?"

    >>> angle_between((1, 0), (0, 1))
    1.5707963267948966
    >>> angle_between((1, 0), (1, 0))
    0.0
    >>> angle_between((1, 0), (-1, 0))
    3.141592653589793
    """

    dot_product = v1 @ v2  # |v1| * |v2| * cos(angle)
    if len(v1.shape) == 2:
        cross_product = v1[:, 0] * v2[1] - v1[:, 1] * v2[0]  # |v1| * |v2| * sin(angle)
    else:
        cross_product = v1[0] * v2[1] - v1[1] * v2[0]

    rotations = np.arctan2(
        cross_product, dot_product
    )  # arctan(sin(angle) / cos(angle))
    return rotations


def transform_trajectory(trajectory, translation, rotation):
    """Transforms the given trajectory by first shifting it and then rotating it.

        out = rotate(trajectory - translation)

    args:
        trajectory: [np.array with shape=(traj_len, 2) or shape=(batch_size, traj_len, 2)]
        translation: [np.array with shape=(2,) or shape=(batch_size, 2)]
        rotation: [scalar or np.array with shape=(batch_size,)]

    returns:
        the transformed trajectory:
            [np.array with shape=(traj_len, 2) or shape=(batch_size, traj_len, 2)]
    """

    rot_mat = np.array(
        [[np.cos(rotation), -np.sin(rotation)], [np.sin(rotation), np.cos(rotation)]]
    )

    translation = np.expand_dims(translation, axis=-2)

    # creating the transposed roation matrix:
    if len(trajectory.shape) == 3:
        # reshaping the rotation matrix if in batch mode:
        rot_mat = np.moveaxis(rot_mat, 2, 0)  # (batch_size, 2, 2)
        rot_mat = np.transpose(rot_mat, axes=(0, 2, 1))
    else:
        rot_mat = rot_mat.T

    trajectory -= translation

    # transformed_trajectory = np.matmul(trajectory, rot_mat)
    transformed_trajectory = trajectory @ rot_mat

    return transformed_trajectory


def retransform_trajectory(trajectory, translation, rotation):
    """Retransforms a trajectory that has been transformed by shifting it by -translation and rotating it by rotation.
    Hence, the points are first rotated by -rotation and then shifted by translation.

    Args:
        trajectory [np.array with shape=(traj_len, 2)]: The (x, y) points of the trajectory
        translation [np.array with shape=(2,)]: The translation vector
        rotation [scalar]: The yaw angle of the rotation

    Returns:
        The transformed trajectory (np.array)
    """
    rot_mat = np.array(
        [
            [np.cos(-rotation), -np.sin(-rotation)],
            [np.sin(-rotation), np.cos(-rotation)],
        ]
    )

    return trajectory @ rot_mat.T + translation


def retransform_cov(pred_out, rotation):
    """Transforms covariance matrices to the global coordinate system (= rotates them)

    Args:
        pred_out ([np.array]): [description]
        rotation ([float]): [rotation angle]

    Returns:
        The transformed covariance matrix in np.array
    """
    rot_mat = np.array(
        [
            [np.cos(-rotation), -np.sin(-rotation)],
            [np.sin(-rotation), np.cos(-rotation)],
        ]
    )

    cov_list = [
        rot_mat
        @ np.array(
            [
                [pred_out[x, 0, 2], pred_out[x, 0, 4]],
                [pred_out[x, 0, 4], pred_out[x, 0, 3]],
            ]
        )
        @ rot_mat.T
        for x in range(len(pred_out))
    ]

    return np.array(cov_list)


if __name__ == "__main__":

    # Test with example data

    x_list = [
        437.56731038436345,
        438.6852893959684,
        439.74080209936903,
        440.4698214772491,
        441.14865354183496,
        441.7945383546361,
        442.35865104136474,
        442.7998234383539,
        443.12149533189904,
        443.32865451485156,
        443.4094288693953,
        443.35920837959816,
        443.16746519138655,
        442.7983781427311,
    ]
    y_list = [
        -2172.249342799885,
        -2167.336603425321,
        -2162.0025032218996,
        -2155.9708399216497,
        -2149.2644011594402,
        -2142.060639930153,
        -2134.432286381409,
        -2126.4807102410305,
        -2118.2234666886065,
        -2109.390705020447,
        -2099.7082952523397,
        -2089.7006866135544,
        -2080.4078697775567,
        -2072.1956765733526,
    ]

    trajectory = np.array([x_list, y_list]).T

    heading = get_heading(trajectory)

    print(heading)

    plt.plot(x_list, y_list)
    plt.axis("equal")
    plt.show()
