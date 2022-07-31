import numpy as np

from utils.fuzzy import FuzzyMembershipFunction, FuzzyImplication, FuzzyInference


class IsFollowerNear(FuzzyMembershipFunction):
    """Fuzzy membership function that answers the question:
    "Is the follower car near the leading car?"
    """

    def __call__(self, x):
        """implements the membership function.
            If the distance is smaller than bound_min, the value is 1
            If the distance is bigger than bound_max, the value is 0.
            Between them it varies linearly.

        args:
            x: dict, that must contain the following keys:
                pos_leading: the position of the leading vehicle.
                pos_following: the position of the following vehicle.

        returns:
            The membership value.
        """

        # main variables that have an effect on the membership value:
        bound_min = 10
        bound_max = 70

        # extracting the variables:
        pos_leading = x["pos_leading"]
        pos_following = x["pos_following"]

        dist = np.linalg.norm(pos_leading - pos_following)

        # membership function definition:
        if dist < bound_min:
            # print("Is follower near: {}".format(1))
            return 1
        elif dist < bound_max:
            # print("Is follower near: {}".format(1 - (dist - bound_min) / (bound_max - bound_min)))
            return 1 - (dist - bound_min) / (bound_max - bound_min)
        else:
            # print("Is follower near: {}".format(0))
            return 0


class IsFollowerOnTheRight(FuzzyMembershipFunction):
    """Fuzzy membership function that answers the question:
    "Is the follower car on the right of the leading car?"
    """

    def __call__(self, x):
        """implements the membership function.
        It calculates the ratio between the distance of the 2 vehicles and
        the differenc between their distance to the left wall. The ratio is hence
        positive, if the follower car is on the right of the leader.
            If the ratio is bigger than +bound, the value is 1
            If the distance is smaller than -bound, the value is 0.
            Between them it varies linearly.

        args:
            x: dict, that must contain the following keys:
                pos_leading: the position of the leading vehicle.
                pos_following: the position of the following vehicle.
                side_dist_dict: Dictionary that contains the distances
                    of the vehicles to the sides of the track.

        returns:
            The membership value.
        """

        # main variables that have an effect on the membership value:
        bound = 0.1

        # extracting the variables:
        pos_leading = x["pos_leading"]
        pos_following = x["pos_following"]
        side_dist_dict = x["side_dist_dict"]

        dist = np.linalg.norm(pos_leading - pos_following)
        dist_diff_left = (
            side_dist_dict["left_following"] - side_dist_dict["left_leading"]
        )
        ratio = dist_diff_left / (dist + 1e-5)

        # membership function definition:
        if ratio > bound:
            # print("Is follower on the right: {}".format(1))
            return 1
        elif ratio > -bound:
            # print("Is follower on the right: {}".format(1 - (bound - ratio) / (2 * bound)))
            return 1 - (bound - ratio) / (2 * bound)
        else:
            # print("Is follower on the right: {}".format(0))
            return 0


class BasedOnLeaderPosition(FuzzyMembershipFunction):
    """Fuzzy membership function that answers the question:
    "Should the leader car be overtaken from the right if
    considering only its position on the track?"
    """

    def __call__(self, x):
        """implements the membership function.
        It is based on the distance of the leader car from the left
        track boundary. The function is linear and has got the
        following breakpoints (breakpoint - breakpoint value):
            Totally on the left - 1
            breakpoint1         - 0.8
            breakpoint2         - 0.2
            totally on the right - 0

        args:
            x: dict, that must contain the following keys:
                side_dist_dict: Dictionary that contains the distances
                    of the vehicles to the sides of the track.
                overtake_margin: The minimum distance that is needed
                    for overtaking a vehicle.

        returns:
            The membership value.
        """

        # main variables that have an effect on the membership value:
        bp1 = 0.3
        bp2 = 1 - bp1

        # extracting the variables:
        side_dist_dict = x["side_dist_dict"]
        overtake_margin = x["overtake_margin"]

        dist_left = side_dist_dict["left_leading"] - overtake_margin
        track_width = (
            side_dist_dict["left_leading"]
            + side_dist_dict["right_leading"]
            - 2 * overtake_margin
        )

        ratio = dist_left / (track_width + 1e-5)

        # membership function definition:
        if ratio < bp1:
            # print("pure based on leader position: {}".format(1 - (ratio) / bp1 * 0.2))
            return 1 - (ratio) / bp1 * 0.2
        elif ratio < bp2:
            # print("pure based on leader position: {}".format(0.8 - (ratio - bp1) / (bp2 - bp1) * 0.6))
            return 0.8 - (ratio - bp1) / (bp2 - bp1) * 0.6
        else:
            # print("pure based on leader position: {}".format(0.2 - (ratio - bp2) / (1 - bp2) * 0.2))
            return 0.2 - (ratio - bp2) / (1 - bp2) * 0.2


class BasedOnRelativePosition(FuzzyMembershipFunction):
    """Fuzzy membership function that answers the question:
    "Should the leader car be overtaken from the right if
    only considering 2 cars lateral position?"
    """

    def __call__(self, x):
        """implements the membership function.
        It is simply a linear membership function that connects
        the 2 extremum.

        args:
            x: dict, that must contain the following keys:
                side_dist_dict: Dictionary that contains the distances
                    of the vehicles to the sides of the track.
                overtake_margin: The minimum distance that is needed
                    for overtaking a vehicle.

        returns:
            The membership value.
        """

        # extracting the variables:
        side_dist_dict = x["side_dist_dict"]
        overtake_margin = x["overtake_margin"]

        dist_left_leading = side_dist_dict["left_leading"] - overtake_margin
        dist_left_following = side_dist_dict["left_following"] - overtake_margin
        track_width = (
            side_dist_dict["left_leading"]
            + side_dist_dict["right_leading"]
            - 2 * overtake_margin
        )

        ratio = (dist_left_following - dist_left_leading) / (track_width + 1e-5)

        # membership function definition:
        # print("Based on relative position: {}".format(1 - (1 - ratio) / 2))
        return 1 - (1 - ratio) / 2


class CloseRightImplication(FuzzyImplication):
    """This Implication has to be reimplemented, the
    default min operator does not work here. The implication
    we want to take, is that if we are close to the vehicle,
    then we should overtake it from the side, on which we are on.
    So:
        "If the follower is near the leader and is on the right
        sid of the leader, we should overtake it from the right."

    The reason, why it should be reimplemented, is that if we are far
    away, than that does not imply anything, in that case the overtake
    can happen on both sides. So if we are far away, that does not simply
    negate the above implication (and hence leads to left overtake), it only
    sais, that based on this argument one can not decide (return value 0.5)
    """

    def __call__(self, x):
        """Implements the implication based on the argument in
        the class description.
        """

        membership_values = [
            membership_function(x) for membership_function in self._evidences
        ]

        if membership_values[0] < 0.5:
            return 0.5
        else:
            return min(membership_values)


class FuzzyOvertakeInference(FuzzyInference):
    """Class for the fuzzy inference, wether the
    car should be overtaken from the right or no.
    """

    def __call__(self, x):
        """Reimplementing the default (max) inference
        operator to normalized summation.
        """

        implication_values = [implication(x) for implication in self._implications]

        return sum(implication_values) / len(implication_values)
