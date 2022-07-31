import matplotlib
import numpy as np
from scipy.signal import savgol_filter


def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception("lines do not intersect")

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y


def get_arc(boundary):
    """Calculate a list of cummulated arc lenghts of given x,y points.

    Args:
        boundary [np.array with shape=(N, 2)]: the points for which the arc lengths are calculated.

    Returns:
        np.array with shape=(N,) the arc lengths corresponding to the points in boundary.
    """

    dists = np.linalg.norm((boundary[1:, :] - boundary[:-1, :]), axis=1)

    return np.hstack((np.array([0]), np.cumsum(dists)))


def get_arc_length(track_array):
    """Calculate cumulative arc length along track array.
    track_array: np.array, shape = (n, 2)
    arc_length: np.array, shape = (n)
    """
    arc_length = np.concatenate(
        [np.zeros(1), np.cumsum(np.linalg.norm(np.diff(track_array, axis=0), axis=1))]
    )
    return arc_length


def get_track_paths(
    track_path: str,
    bool_track_width: bool = False,
    bool_raceline: bool = False,
    bool_all_kinematics: bool = False,
) -> tuple:
    """Reads the map data from the unified map file."""

    (
        refline,
        t_width_right,
        t_width_left,
        normvec_normalized,
        alpha_dist,
        s_rl,
        _,
        _,
        _,
        _,
        _,
        s_refline,
        psi_refline,
        kappa_refline,
        _,
    ) = import_global_trajectory_csv(import_path=track_path)

    x_intp = 0.999
    close_bound = x_intp * (refline[0, :] - refline[-1, :]) + refline[-1, :]
    refline = np.vstack([refline, close_bound])

    close_bound = (
        x_intp * (normvec_normalized[0, :] - normvec_normalized[-1, :])
        + normvec_normalized[-1, :]
    )
    normvec_normalized = np.vstack([normvec_normalized, close_bound])

    close_bound = x_intp * (t_width_right[0] - t_width_right[-1]) + t_width_right[-1]
    t_width_right = np.append(t_width_right, close_bound)

    close_bound = x_intp * (t_width_left[0] - t_width_left[-1]) + t_width_left[-1]
    t_width_left = np.append(t_width_left, close_bound)

    bound_right = refline + normvec_normalized * np.expand_dims(t_width_right, 1)
    bound_left = refline - normvec_normalized * np.expand_dims(t_width_left, 1)

    track_width = t_width_right + t_width_left
    if bool_track_width:
        return (s_rl, refline, bound_right, bound_left, track_width)
    elif bool_raceline:
        close_bound = x_intp * (alpha_dist[0] - alpha_dist[-1]) + alpha_dist[-1]
        alpha_dist = np.append(alpha_dist, close_bound)
        raceline = refline + normvec_normalized * alpha_dist[:, np.newaxis]
        return (s_refline, refline, bound_right, bound_left, raceline)
    elif bool_all_kinematics:
        close_bound = (
            x_intp * (kappa_refline[0] - kappa_refline[-1]) + kappa_refline[-1]
        )
        kappa_refline = np.append(kappa_refline, close_bound)
        close_bound = x_intp * (psi_refline[0] - psi_refline[-1]) + psi_refline[-1]
        psi_refline = np.append(psi_refline, close_bound)
        return (
            s_refline,
            refline,
            bound_right,
            bound_left,
            track_width,
            psi_refline,
            kappa_refline,
        )
    else:
        return (s_refline, refline, bound_right, bound_left)


def import_global_trajectory_csv(import_path: str) -> tuple:
    """Import global trajectory

    :param import_path: path to the csv file containing the optimal global trajectory
    :type import_path: str
    :return: - xy_refline: x and y coordinate of reference-line

             - t_width_right: width to right track bound at given reference-line coordinates in meters

             - t_width_left: width to left track bound at given reference-line coordinates in meters

             - normvec_normalized: x and y components of normalized normal vector at given reference-line coordinates

             - alpha_dist: distance from optimal racing-line to reference-line at given reference-line coordinates

             - s_refline: s-coordinate of reference-line at given reference-line coordinates

             - psi_refline: heading at given reference-line coordinates

             - kappa_refline: curvature at given reference-line coordinates

             - dkappa_refline: derivative of curvature at given reference-line coordinates

             - s_rl: s-coordinate of racing-line at given racing-line coordinates

             - vel_rl: velocity at given racing-line coordinates

             - acc_rl: acceleration at given racing-line coordinates

             - psi_rl: heading at given racing-line coordinates

             - kappa_rl: curvature at given racing-line coordinates

             - banking: banking at given racling-line coordinates
    :rtype: tuple
    """

    # ------------------------------------------------------------------------------------------------------------------
    # IMPORT DATA ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # load data from csv file (closed; assumed order listed below)
    # x_ref_m, y_ref_m, width_right_m, width_left_m, x_normvec_m, y_normvec_m, alpha_m, s_racetraj_m,
    # psi_racetraj_rad, kappa_racetraj_radpm, vx_racetraj_mps, ax_racetraj_mps2
    csv_data_temp = np.loadtxt(import_path, delimiter=";")

    # get reference-line
    xy_refline = csv_data_temp[:-1, 0:2]

    # get distances to right and left bound from reference-line
    t_width_right = csv_data_temp[:-1, 2]
    t_width_left = csv_data_temp[:-1, 3]

    # get normalized normal vectors
    normvec_normalized = csv_data_temp[:-1, 4:6]

    # get racing-line alpha
    alpha_dist = csv_data_temp[:-1, 6]

    # get racing-line s-coordinate
    s_rl = csv_data_temp[:, 7]

    # get heading at racing-line points
    psi_rl = csv_data_temp[:-1, 8]

    # get kappa at racing-line points
    kappa_rl = csv_data_temp[:-1, 9]

    # get velocity at racing-line points
    vel_rl = csv_data_temp[:-1, 10]

    # get acceleration at racing-line points
    acc_rl = csv_data_temp[:-1, 11]

    # get banking
    banking = csv_data_temp[:-1, 12]

    # get reference-line s-coordinate
    s_refline = csv_data_temp[:, 13]

    # get heading at reference-line points
    psi_refline = csv_data_temp[:-1, 14]

    # get curvature at reference-line points
    kappa_refline = csv_data_temp[:-1, 15]

    # get derivative of curvature at reference-line points
    dkappa_refline = csv_data_temp[:-1, 16]

    return (
        xy_refline,
        t_width_right,
        t_width_left,
        normvec_normalized,
        alpha_dist,
        s_rl,
        psi_rl,
        kappa_rl,
        vel_rl,
        acc_rl,
        banking,
        s_refline,
        psi_refline,
        kappa_refline,
        dkappa_refline,
    )


def get_glob_raceline(
    loctraj_param_path, bool_vel_const=True, velocity=100.0, vel_scale=1.0
):
    """load data from csv files."""
    (
        refline,
        _,
        _,
        normvec_normalized,
        alpha_mincurv,
        s_rl,
        psi_rl,
        _,
        vel_rl,
        acc_rl,
        _,
        _,
        _,
        _,
        _,
    ) = import_global_trajectory_csv(import_path=loctraj_param_path)

    # get race line
    raceline = refline + normvec_normalized * alpha_mincurv[:, np.newaxis]

    x_intp = 0.999
    close_bound = x_intp * (raceline[0, :] - raceline[-1, :]) + raceline[-1, :]
    raceline = np.vstack([raceline, close_bound])

    close_bound = x_intp * (psi_rl[0] - psi_rl[-1]) + psi_rl[-1]
    psi_rl = np.append(psi_rl, close_bound)

    close_bound = x_intp * (vel_rl[0] - vel_rl[-1]) + vel_rl[-1]
    vel_rl = np.append(vel_rl, close_bound)

    close_bound = x_intp * (acc_rl[0] - acc_rl[-1]) + acc_rl[-1]
    acc_rl = np.append(acc_rl, close_bound)

    if bool_vel_const:
        vel_rl = np.ones(vel_rl.shape) * velocity
        acc_rl = np.zeros(vel_rl.shape)
    else:
        vel_rl *= vel_scale
        acc_rl *= vel_scale

    psi_rl = remove_psi_step(psi_rl)

    dpsi_rl = get_dpsi(psi_rl, s_rl, vel_rl)

    return list((s_rl, vel_rl, raceline, psi_rl, dpsi_rl, acc_rl))


def remove_psi_step(psi_rl):
    step = np.diff(psi_rl)
    max_step = np.argmax(abs(step))
    if step[max_step] > 0:
        psi_rl[max_step + 1 :] -= 2 * np.pi
    else:
        psi_rl[max_step + 1 :] += 2 * np.pi
    if np.min(psi_rl) < -6.5:
        psi_rl += 2 * np.pi
    return psi_rl


def get_dpsi(psi_rl, s_rl, vel_rl):
    if np.max(vel_rl) == 0.0:
        dts = np.diff(s_rl) / 0.01
    else:
        dts = np.diff(s_rl) / vel_rl[:-1]
    delta_psi = np.diff(psi_rl)
    delta_psi = np.append(delta_psi, delta_psi[-1])
    dts = np.append(dts, dts[-1])
    for j in range(len(delta_psi)):
        if delta_psi[j] > np.pi / 2.0:
            delta_psi[j] -= 2 * np.pi
        elif delta_psi[j] < -np.pi / 2.0:
            delta_psi[j] += 2 * np.pi

    delta_psi /= dts
    dpsi_rl = savgol_filter(delta_psi, 5, 2, 0)

    return dpsi_rl


def get_track_kinematics(
    path_name, track_path, velocity=100, vel_scale=1.0, bool_get_yaw_curv=False
):
    """Get kinematics for a single path of the track

    Kinematics is the list: [arc_length, vel_rl, raceline, psi_rl, dpsi_rl, acc_rl]
    Possible Inputs:
        track_name = 'trackboundary_right', 'trackboundary_left', 'centerline', 'glob_optimal_raceline', 'raceline'
    """

    if "raceline" in path_name and "optimal" in path_name:
        return get_glob_raceline(track_path, bool_vel_const=False, vel_scale=vel_scale)
    elif path_name == "raceline":
        return get_glob_raceline(track_path, bool_vel_const=True, velocity=velocity)
    else:
        tracks_arrays = get_track_paths(track_path, bool_all_kinematics=True)

        if bool_get_yaw_curv:
            return tracks_arrays[5], tracks_arrays[6]

        if "left" in path_name:
            idx = 3
        elif "right" in path_name:
            idx = 2
        else:  # centerline
            idx = 1

        # use heading from refline
        psi = tracks_arrays[5]

        arc_length = tracks_arrays[0]
        line_path = tracks_arrays[idx]

        vel_rl = np.ones(len(line_path)) * velocity
        acc_rl = np.zeros(len(line_path))

        psi = remove_psi_step(psi)

        dpsi_rl = get_dpsi(psi, arc_length, vel_rl)

        return [arc_length, vel_rl, line_path, psi, dpsi_rl, acc_rl]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    matplotlib.use("TkAgg")

    track_path = "mix_net/mix_net/data/map/traj_ltpl_cl_IMS_GPS.csv"

    track_list = get_track_kinematics("centerline", track_path=track_path, velocity=100)
    plt.plot(track_list[2][:, 0], track_list[2][:, 1])
    track_list = get_track_kinematics("raceline", track_path=track_path, velocity=100)
    plt.plot(track_list[2][:, 0], track_list[2][:, 1])
    track_list = get_track_kinematics("left", track_path=track_path, velocity=100)
    plt.plot(track_list[2][:, 0], track_list[2][:, 1])
    track_list = get_track_kinematics("right", track_path=track_path, velocity=100)
    plt.plot(track_list[2][:, 0], track_list[2][:, 1])
    plt.show()
    idx = 3  # yaw
    track_list = get_track_kinematics("centerline", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    track_list = get_track_kinematics("raceline", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    track_list = get_track_kinematics("left", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    track_list = get_track_kinematics("right", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    plt.show()
    idx = 4  # yawrate
    track_list = get_track_kinematics("centerline", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    track_list = get_track_kinematics("raceline", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    track_list = get_track_kinematics("left", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    track_list = get_track_kinematics("right", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    plt.show()
    idx = 5  # accl.
    track_list = get_track_kinematics("centerline", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    track_list = get_track_kinematics("raceline", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    track_list = get_track_kinematics("left", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    track_list = get_track_kinematics("right", track_path=track_path, velocity=100)
    plt.plot(track_list[idx])
    plt.show()
