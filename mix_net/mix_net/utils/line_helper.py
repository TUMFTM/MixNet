import numpy as np


def cross2D(vect1, vect2):
    """Calculates the cross product between 2 2D vectors.
    IMPORTANT: The order of the vectors matter!!!

    args:
        vect1: (np.array, shape=(2,) or shape=(N, 2)), the thumb in the right hand rule
        vect2: (np.array, shape=(2,) or shape=(N, 2)), the pointing finger in the right hand rule

    returns:
        The scalar value of the cross product.
    """
    if len(vect1.shape) == 1:
        return vect1[0] * vect2[1] - vect1[1] * vect2[0]
    else:
        return vect1[:, 0] * vect2[:, 1] - vect1[:, 1] * vect2[:, 0]


def is_static_traj(obs, threshold=1.0):
    """Checks whether a trajectory can be considered as static.

    args:
        obs: [np.array with shape=(N, 2)]: the trajectoriy
        threshold: [float, default=1.0]: If the max difference in both x and y
            directions is smaller than this threshold, it is considered static.

    returns:
        true, if the obstacle should be considered static.
    """

    x_min = np.min(obs[:, 0])
    x_max = np.max(obs[:, 0])

    y_min = np.min(obs[:, 1])
    y_max = np.max(obs[:, 1])

    return (x_max - x_min < threshold) and (y_max - y_min < threshold)


class LineHelper:
    """Class for handling a line (boundaries, center line, race line)."""

    def __init__(self, line):
        """Initializes a LineHelper object."""

        self.line = line
        self.tangents = self._get_tangents()
        self.curvs = self._get_curvatures()
        self.arc_lens = self._get_arc_lens()

        self.max_arc_len = self.arc_lens[-1]
        self.max_ind = line.shape[0]

    def _get_tangents(self):
        """Calculates the normalized tangent vectors for the line. The tangentials
        are calculated from the 2 neighbouring points.
        """

        tangs = np.zeros_like(self.line)

        # very first point:
        tangs[0, :] = self.line[1, :] - self.line[-1, :]

        # very last point:
        tangs[-1, :] = self.line[0, :] - self.line[-2, :]

        # every other point:
        tangs[1:-1, :] = self.line[2:, :] - self.line[:-2, :]

        # normalizing:
        tangs /= np.linalg.norm(tangs, axis=1)[
            :, None
        ]  # None creates a new axis for the division.

        return tangs

    def _get_curvatures(self):
        """Calculates the curvatures for the line."""

        kappa = np.zeros(self.line.shape[0])

        # point 0:
        d1 = self.line[0, :] - self.line[-1, :]
        ds1 = np.linalg.norm(d1) + 1e-12
        d1 /= ds1

        d2 = self.line[1, :] - self.line[0, :]
        ds2 = np.linalg.norm(d2) + 1e-12
        d2 /= ds2

        # the np.sign() part sets the sign of the curvature value.
        kappa[0] = np.linalg.norm((d2 - d1) / ds1) * np.sign(cross2D(d1, d2))

        # points [1: -1]:
        d1 = self.line[1:-1, :] - self.line[:-2, :]
        ds1 = np.linalg.norm(d1, axis=1) + 1e-12
        d1 /= ds1[:, None]

        d2 = self.line[2:, :] - self.line[1:-1, :]
        ds2 = np.linalg.norm(d2, axis=1) + 1e-12
        d2 /= ds2[:, None]

        # the np.sign() part sets the sign of the curvature value.
        kappa[1:-1] = np.linalg.norm((d2 - d1) / ds1[:, None], axis=1) * np.sign(
            cross2D(d1, d2)
        )

        # point -1:
        d1 = self.line[-1, :] - self.line[-2, :]
        ds1 = np.linalg.norm(d1) + 1e-12
        d1 /= ds1

        d2 = self.line[0, :] - self.line[-1, :]
        ds2 = np.linalg.norm(d2) + 1e-12
        d2 /= ds2

        # the np.sign() part sets the sign of the curvature value.
        kappa[-1] = np.linalg.norm((d2 - d1) / ds1) * np.sign(cross2D(d1, d2))

        return kappa

    def _get_arc_lens(self):
        """Calculates the arclen for every point in the line.

        If line has shape=(N, 2), arc_lens will have shape=((N+1),):
        The first and last arc distances both correspond to the very first point. The first
        value is 0 and the last one is the length of the line.
        """

        arc_lens = np.zeros((self.line.shape[0] + 1))
        dists = np.linalg.norm((self.line[1:, :] - self.line[:-1, :]), axis=1)
        arc_lens[1:-1] = np.cumsum(dists)

        arc_lens[-1] = arc_lens[-2] + np.linalg.norm(self.line[0, :] - self.line[-1, :])

        return arc_lens

    def get_nearest_ind(self, point, hint=None):
        """Gets the nearest index of the line to the point given.
        If a hint index is given to search around, iterative search is carried out.
        """

        if hint is not None:
            return self._get_nearest_ind_iterative(point, hint=hint)
        else:
            return self._get_nearest_ind_naive(point)

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
            ind += self.max_ind
        while ind >= self.max_ind:
            ind -= self.max_ind

        ind_prev = ind - 1 if ind > 0 else self.max_ind - 1

        dist = np.linalg.norm(self.line[ind, :] - point)
        dist_prev = np.linalg.norm(self.line[ind_prev, :] - point)

        add_on = 1 if dist < dist_prev else -1

        counter = 0
        smallest = dist
        smallest_ind = ind

        # there are some small local minimas around the global minima, this is
        # why the 0.01 threshold is needed.
        while (smallest + 0.01) > dist:
            ind += add_on
            if ind >= self.max_ind:
                ind -= self.max_ind
            if ind < 0:
                ind += self.max_ind

            dist = np.linalg.norm(self.line[ind, :] - point)

            if smallest > dist:
                smallest = dist
                smallest_ind = ind

            if counter > self.max_ind:
                print("Did not find solution for closest index in a whole round.")
                break

        return smallest_ind

    def _get_nearest_ind_naive(self, point):
        """Finds the nearest point of the left track boundary to
        the given point by the naive method: calculates the distance
        to all of the points and choses the index with the smallest distance.
        """

        dists = np.linalg.norm((self.line - point), axis=1)

        return np.argmin(dists)
