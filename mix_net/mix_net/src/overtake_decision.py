# Standard imports
import os
import sys

# Third party imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.widgets import Slider, Button

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

# Custom imports
from utils.map_utils import get_track_paths
from utils.fuzzy import FuzzyImplication
import utils.overtake_fuzzy as ofz
from utils.geometry import check_collision, distance_point_line


# mod_prediction_node path
mod_path = os.path.dirname(os.path.dirname(__file__))


class OvertakeDecisionMaker:
    """Decide on overtaking left or right given two positions of racecars."""

    def __init__(self, track_path: None):
        """initializes an OvertakeDecisionMaker object."""
        # Load Map data
        if not os.path.exists(track_path):
            print(
                "Track Data is missing, try to run with default csv-file from data/map"
            )
            track_path = None
        (
            self.arc_center,
            self._center_line,
            self._bound_right,
            self._bound_left,
            _,
        ) = get_track_paths(track_path, bool_track_width=True)

        # Fuzzy implication objects:
        # for details on what these do please read "mod_prediction_node/utils/fuzzy.py"
        # and "tools/overtake_fuzzy.py".
        follower_is_close_and_on_the_right = ofz.CloseRightImplication(
            [ofz.IsFollowerNear(), ofz.IsFollowerOnTheRight()]
        )

        leader_position_implication = FuzzyImplication([ofz.BasedOnLeaderPosition()])

        relative_position_implication = FuzzyImplication(
            [ofz.BasedOnRelativePosition()]
        )

        # Fuzzy inference object:
        self._fuzzy_inference = ofz.FuzzyOvertakeInference(
            [
                follower_is_close_and_on_the_right,
                leader_position_implication,
                relative_position_implication,
            ]
        )

    def get_overtake_direction(
        self,
        pos_leading,
        pos_following,
        overtake_margin=3.0,
        weight_leader=1.5,
    ):
        """[Decide on overtaking left or right given two positions of racecars.]

        Args:
            pos_leading (np.array()): [x,y coordinate of leading racecar]
            pos_following (np.array()): [x,y coordinate of following racecar]
            bound_left ([np.array], optional): [Left boundary of racetrack]. Defaults to bound_left.
            bound_right ([np.array], optional): [Right boundary of racetrack]. Defaults to bound_right.
            overtake_margin (float, optional): [Margin to boundary needed at least for overatking]. Defaults to 3.0.
            weight_leader (float, optional): [Weight factor to priotize the position of the leading racecar]. Defaults to 1.5.


        Returns:
            [str]: [overtake_left or overtake_right]
        """
        side_dist_dict = {
            "left_leading": np.infty,
            "left_following": np.infty,
            "right_leading": np.infty,
            "right_following": np.infty,
        }

        # Calculate distances to the boundaries
        side_dist_dict["left_leading"] = distance_point_line(
            pos_leading, self._bound_left
        )
        side_dist_dict["left_following"] = distance_point_line(
            pos_following, self._bound_left
        )
        side_dist_dict["right_leading"] = distance_point_line(
            pos_leading, self._bound_right
        )
        side_dist_dict["right_following"] = distance_point_line(
            pos_following, self._bound_right
        )

        # If too close to a boundary, there is no space for overtaking
        if side_dist_dict["left_leading"] <= overtake_margin:
            return "overtake_right"
        elif side_dist_dict["right_leading"] <= overtake_margin:
            return "overtake_left"

        # Otherwise, carry out fuzzy inference:
        else:
            # info dict that contains all the info for the fuzzy inference:
            info_dict = {
                "pos_leading": pos_leading,
                "pos_following": pos_following,
                "side_dist_dict": side_dist_dict,
                "overtake_margin": overtake_margin,
            }

            # The fuzzy inference gives back a value between 0 and 1 that says,
            # how much the data in info_dict implies a right overtake:
            if self._fuzzy_inference(info_dict) > 0.5:
                return "overtake_right"
            else:
                return "overtake_left"

    def adjust_prediction(
        self,
        leading_pred,
        following_pred,
        ts,
        decision,
        approx_radius,
        ot_length_ts,
        min_overtake_dist=3.0,
        logger=None,
    ):
        """Method to calculate the necessary lateral adjustment distance
        to avoid collision.

        args:
            leading_pred: (np.array, shape=(2, N)), The predicted trajectory of the leading car.
            following_pred: (np.array, shape=(2, N)), The predicted trajectory of the following_pred car.
            ts: (int), the timestep, in which the collision happens.
            decision: (string), the overtake decision ("overtake_right" or "overtake_left")
            approx_radius: (float), if the distance between the cars is less then this, it is a collision.
            ot_length_ts, (int), the time needed to change the line in number of timesteps!!!
            min_overtake_dist: (float), the minimum clearance between the cars during an overtake.

        returns:
            The adjusted following prediction.
        """

        ts_max = min([leading_pred.shape[1], following_pred.shape[1]]) - 1
        if ts_max < 2:
            if logger is not None:
                logger.info("Too short prediction, can not be adjusted!")
            return following_pred, 0.0

        if ts == 0:
            if logger is not None:
                logger.info(
                    "The collision is in timestep 0 in prediction --> can not be adjusted!"
                )
            return following_pred, 0.0

        if ts > ts_max:
            ts = ts_max
            if logger is not None:
                logger.info(
                    "The ts given to get_dist_adjustment is not valid! Received: {}, max would have been: {}".format(
                        ts, ts_max
                    )
                )

        if decision not in ["overtake_right", "overtake_left"]:
            decision = "overtake_left"
            if logger is not None:
                logger.info(
                    "The decision given to get_dist_adjustment is not valid! Received: {}".format(
                        decision
                    )
                )

        # finding, how long the trajectories are colliding:
        ts_last = ts
        is_collision = True
        while is_collision & (ts_last <= ts_max):
            p_leading = leading_pred[:, ts_last]
            p_following = following_pred[:, ts_last]
            is_collision = check_collision(
                p_leading, p_following, approx_radius=approx_radius
            )
            ts_last += 1

        # setting back one timestep, this is the last one that
        # actually collides:
        ts_last -= 1

        # timesteps where the maneuver begins and ends:
        ot_begin_ts = ts - ot_length_ts if (ts - ot_length_ts) > 0 else 0
        ot_end_ts = ts_last + ot_length_ts

        # getting for each point the necessary adjustment:
        adjustment = np.zeros_like(following_pred)

        # the points which are actually in collision:
        for t in range(ts, ts_last + 1):
            adjust_vect = self._get_adjust_vector(
                p_lead=leading_pred[:, t],
                p_foll_0=following_pred[:, t - 1],
                p_foll_1=following_pred[:, t],
                decision=decision,
                min_overtake_dist=min_overtake_dist,
            )
            adjustment[:, t] = adjust_vect

        # the points before the collision:
        max_adjust = adjustment[:, ts]
        max_adjust = np.expand_dims(max_adjust, axis=1)
        no_adjustment_ts = ts - ot_begin_ts
        length_mult = np.linspace(0, 1, num=int(no_adjustment_ts), endpoint=False)
        length_mult = np.expand_dims(length_mult, 1)
        adjustment[:, ot_begin_ts:ts] = max_adjust @ length_mult.T

        # the points after the collision:
        max_adjust = adjustment[:, ts_last]
        max_adjust = np.expand_dims(max_adjust, axis=1)
        no_adjustment_ts = ot_end_ts - ts_last + 1
        length_mult = np.linspace(1, 0, num=int(no_adjustment_ts), endpoint=True)
        length_mult = np.expand_dims(length_mult, 1)
        adj = max_adjust @ length_mult.T
        max_index = ot_end_ts if ot_end_ts <= ts_max else ts_max
        adjustment[:, ts_last : max_index + 1] = adj[:, : max_index - ts_last + 1]

        # for logging:
        max_adjustment = np.amax(np.linalg.norm(adjustment, axis=0))

        return following_pred + adjustment, max_adjustment

    def _get_adjust_vector(
        self, p_lead, p_foll_0, p_foll_1, decision, min_overtake_dist
    ):
        """Gets a vector, with which a colliding timestep should be adjusted.
        The adjustment considers the direction (left/right) and is
        perpendicular to the driving direction of the following vehicle.

        args:
            p_lead: (np.array, shape=(2,)), the collision timestep position of the leader.
            p_foll_0: (np.array, shape=(2,)), the previous timestep position of the follower.
            p_foll_1: (np.array, shape=(2,)), the collision timestep position of the follower.
            decision: (string), the overtake decision ("overtake_right" or "overtake_left")
            min_overtake_dist: (float), the minimum clearance between the cars during an overtake.

        returns:
            The vector, with which the follower should be pushed away.
        """

        foll_vect = p_foll_1 - p_foll_0
        foll_to_leader = p_lead - p_foll_1

        # calculating the perpendicular vector to the direction
        # of the follower car: (turning the vector by 90 in the positive direction.)
        # This vector points to the left when looking at from the follower vehicle.
        foll_perp = np.array([-foll_vect[1], foll_vect[0]])
        foll_perp_norm = foll_perp / (np.linalg.norm(foll_perp) + 1e-5)

        # the lateral distance is positive, if the follower is right of
        # the leader:
        lat_dist = foll_perp_norm @ foll_to_leader.T

        # if the lateral distance between the two vehicles is already larger than the overtake
        # distance, then no adjustment is needed. This can happen, because 2 * approx_radius, which
        # is used for collision checking, is larger than lat_overtake_dist.
        if np.abs(lat_dist) >= np.abs(min_overtake_dist):
            return np.zeros((2,))

        # if the follower overtakes from the right, the desired distance should be positive to
        # match the sign of lat_dist:
        dist_desired = (
            min_overtake_dist if decision == "overtake_right" else -min_overtake_dist
        )

        # right adjustment is positive.
        dist_diff = dist_desired - lat_dist

        # right adjustment is negative, since the perpendicular vector foll_perp
        # points actually to the left:
        dist_diff *= -1.0

        # calculating and returning the necessary correction vector:
        return dist_diff * foll_perp_norm


if __name__ == "__main__":
    # Load Map data
    track_path = "data/map/traj_ltpl_cl_IMS_GPS.csv"
    if not os.path.exists(track_path):
        print("Track Data is missing, try to run with default csv-file from data/map")
        track_path = None
    (
        arc_center,
        center_line,
        bound_right,
        bound_left,
        _,
    ) = get_track_paths(track_path, bool_track_width=True)

    pos_leading = center_line[2500, :]
    pos_following = center_line[2400, :]

    decision_maker = OvertakeDecisionMaker(track_path)

    decision_maker.get_overtake_direction(pos_leading, pos_following)

    # get a list of points to fit a spline to as well
    N = 2
    xvals = np.array(np.array([pos_leading[0], pos_following[0]]))
    yvals = np.array(np.array([pos_leading[1], pos_following[1]]))

    # figure.subplot.right
    mpl.rcParams["figure.subplot.right"] = 0.8

    # set up a plot
    fig, axes = plt.subplots(1, 1, figsize=(9.0, 8.0), sharex=True)
    ax1 = axes

    pind = None  # active point
    epsilon = 10  # max pixel distance

    def update(val):
        global yvals
        global xvals
        # update curve
        for i in np.arange(N):
            yvals[i] = ysliders[i].val
            xvals[i] = xsliders[i].val
        l.set_ydata(yvals)
        l.set_xdata(xvals)
        # redraw canvas while idle
        decision = decision_maker.get_overtake_direction(
            np.array([xvals[0], yvals[0]]), np.array([xvals[1], yvals[1]])
        )
        ax1.set_title(decision)

        fig.canvas.draw_idle()

    def reset(event):
        global yvals
        global xvals
        # reset the values
        for i in np.arange(N):
            xsliders[i].reset()
            ysliders[i].reset()

        l.set_ydata(yvals)
        l.set_xdata(xvals)
        # redraw canvas while idle
        fig.canvas.draw_idle()

    def button_press_callback(event):
        "whenever a mouse button is pressed"
        global pind
        if event.inaxes is None:
            return
        if event.button != 1:
            return
        # print(pind)
        pind = get_ind_under_point(event)

    def button_release_callback(event):
        "whenever a mouse button is released"
        global pind
        if event.button != 1:
            return
        pind = None

    def get_ind_under_point(event):
        "get the index of the vertex under point if within epsilon tolerance"

        # display coords
        # print('display x is: {0}; display y is: {1}'.format(event.x, event.y))
        # t = ax1.transData.inverted()
        tinv = ax1.transData
        # xy = t.transform([event.x, event.y])
        # print('data x is: {0}; data y is: {1}'.format(xy[0], xy[1]))
        xr = np.reshape(xvals, (np.shape(xvals)[0], 1))
        yr = np.reshape(yvals, (np.shape(yvals)[0], 1))
        xy_vals = np.append(xr, yr, 1)
        xyt = tinv.transform(xy_vals)
        xt, yt = xyt[:, 0], xyt[:, 1]
        d = np.hypot(xt - event.x, yt - event.y)
        (indseq,) = np.nonzero(d == d.min())
        ind = indseq[0]

        # print(d[ind])
        if d[ind] >= epsilon:
            ind = None

        # print(ind)
        return ind

    def motion_notify_callback(event):
        "on mouse movement"
        global yvals
        global xvals
        if pind is None:
            return
        if event.inaxes is None:
            return
        if event.button != 1:
            return

        # update yvals
        # print('motion x: {0}; y: {1}'.format(event.xdata, event.ydata))
        xvals[pind] = event.xdata
        xsliders[pind].set_val(xvals[pind])
        yvals[pind] = event.ydata
        ysliders[pind].set_val(yvals[pind])

        # update curve via sliders and draw
        fig.canvas.draw_idle()

    ax1.plot(bound_left[:, 0], bound_left[:, 1], "k-")
    ax1.plot(bound_right[:, 0], bound_right[:, 1], "k-")

    (l,) = ax1.plot(xvals, yvals, color="k", linestyle="none", marker="o", markersize=8)

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_xlim(
        (
            min(pos_leading[0], pos_following[0]) - 50,
            max(pos_leading[0], pos_following[0]) + 50,
        )
    )
    ax1.set_ylim(
        (
            min(pos_leading[1], pos_following[1]) - 50,
            max(pos_leading[1], pos_following[1]) + 50,
        )
    )
    ax1.grid(True)
    ax1.yaxis.grid(True, which="minor", linestyle="--")

    xsliders = []
    ysliders = []

    for i in np.arange(N):
        axamp = plt.axes([0.84, 0.8 - (i * 0.05), 0.12, 0.02])
        # Slider
        s = Slider(
            axamp, "p_y{0}".format(i), yvals[i] - 30, yvals[i] + 30, valinit=yvals[i]
        )
        ysliders.append(s)

    for i in np.arange(N):
        axamp = plt.axes([0.84, 0.8 - ((i + 2) * 0.05), 0.12, 0.02])
        # Slider
        s = Slider(
            axamp, "p_x{0}".format(i), xvals[i] - 30, xvals[i] + 30, valinit=xvals[i]
        )
        xsliders.append(s)

    for i in np.arange(N):
        # samp.on_changed(update_slider)
        ysliders[i].on_changed(update)
        xsliders[i].on_changed(update)

    axres = plt.axes([0.84, 0.8 - ((N * 2) * 0.05), 0.12, 0.02])
    bres = Button(axres, "Reset")
    bres.on_clicked(reset)

    fig.canvas.mpl_connect("button_press_event", button_press_callback)
    fig.canvas.mpl_connect("button_release_event", button_release_callback)
    fig.canvas.mpl_connect("motion_notify_event", motion_notify_callback)

    plt.show()
