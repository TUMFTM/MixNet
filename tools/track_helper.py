import os
import sys
import numpy as np

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from mix_net.mix_net.utils.map_utils import get_track_paths


class TrackHelper:
    """Class to implement some helper functionalities regarding the track data."""

    def __init__(self, track_file, repo_path):
        """Initializes the Track helper by loading the map data and carrying out
        some calculations on it.
        """
        track_path = os.path.join(repo_path, "data", "map", track_file)
        if not os.path.exists(track_path):
            print(
                "Track Data is missing, try to run with default csv-file from data/map"
            )
            track_path = None
        (
            self.arc_center,
            self._center_line,
            self._bound_right_xy,
            self._bound_left_xy,
            self._track_width,
        ) = get_track_paths(track_path, bool_track_width=True)

        self._max_ind = self._bound_left_xy.shape[0]

    def point_is_in_track(self, point, hint=None):
        """Returns true if the point lies within the track. If first
        searches for the nearest point of the left track boundary, and then
        calculates the points distance from it. Then, it is determined,
        whether the point lies within the track.

        Depending on whether a hint was given or not for where to search
        for the nearest point, 2 different search methods are used:
        - If no hint is provided, simply the distance to all of the points
        is calculated. then the point with the smallest distance is taken.
        - If a hint is provided, the search starts from this index
        and then iteratively the next boundary point is taken until
        the distance is reducing. If the distance starts to rise, the
        previous point was the closest one.
            !!! WARNING !!!
            The hint should be "near" the real nearest index, or else
            algorithm converges to a wrong value. (By near we meen, it should be on
            the correct half of the track at least.) Else the assumption, that the
            distance towards the nearest point decreases monotonally does not
            necessarily hold. (So don't give hint if it is not a "good" hint.)
            !!! WARNING !!!

        args:
            point: (np.array), shape=(2,), the point to check.
            hint: (int), an index, around which the nearest point of
                the left boundary should be searched for.
        returns:
            tuple:
                - bool, that is true, if the point is within the track boundaries.
                - the index of the nearest point.
        """

        if hint is not None:
            ind = self._get_nearest_ind_iterative(point, hint)
        else:
            ind = self._get_nearest_ind_naive(point)

        dist_left = np.linalg.norm((self._bound_left_xy[ind, :2] - point))
        dist_right = np.linalg.norm((self._bound_right_xy[ind, :2] - point))
        dist_max = self._track_width[ind]

        in_track = (dist_left <= dist_max) and (dist_right <= dist_max)
        return in_track, ind

    def _get_nearest_ind_iterative(self, point, hint=0):
        """Searches for the index of the nearest point of the
        left track boundary to the given point. It is an iteratice
        search that starts from the given hint index. The distance
        to the given point is calculated iteratively for the next point
        of the boundary until the distance sinks, hence the nearest point
        is found.

        !!! WARNING !!!
        The hint should be "near" the real nearest index, or else
        algorithm converges to a wrong value. (By near we meen, it should be on
        the correct half of the track at least.) Else the assumption, that the
        distance decreases monotonously towards the nearest point does not
        necessarily hold.
        """

        ind = hint
        # just making sure, that ind is in the range [0, self._max_ind)
        while ind < 0:
            ind += self._max_ind
        while ind >= self._max_ind:
            ind -= self._max_ind

        ind_prev = ind - 1 if ind > 0 else self._max_ind - 1

        dist = np.linalg.norm(self._bound_left_xy[ind, :2] - point)
        dist_prev = np.linalg.norm(self._bound_left_xy[ind_prev, :2] - point)

        add_on = 1 if dist < dist_prev else -1

        counter = 0
        smallest = dist
        smallest_ind = ind

        # there are some small local minimas around the global minima, this is
        # why the 0.01 threshold is needed.
        while (smallest + 0.01) > dist:
            ind += add_on
            if ind >= self._max_ind:
                ind -= self._max_ind
            if ind < 0:
                ind += self._max_ind

            dist = np.linalg.norm(self._bound_left_xy[ind, :2] - point)

            if smallest > dist:
                smallest = dist
                smallest_ind = ind

            if counter > self._max_ind:
                print("Did not find solution for closest index in a whole round.")
                break

        return smallest_ind

    def _get_nearest_ind_naive(self, point):
        """Finds the nearest point of the left track boundary to
        the given point by the naive method: calculates the distance
        to all of the points and choses the index with the smallest distance.
        """
        dists = np.linalg.norm((self._bound_left_xy[:, :2] - point), axis=1)
        return np.argmin(dists)
