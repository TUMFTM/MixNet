import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from visualization import draw_with_uncertainty


COLOR_DICT = {
    "ego": "tab:blue",
    "0": "tab:orange",
    "1": "tab:green",
    "2": "tab:red",
    "3": "tab:purple",
    "4": "tab:brown",
    "5": "tab:pink",
    "6": "tab:gray",
    "7": "tab:olive",
    "8": "tab:cyan",
    "9": "b",
    "10": "g",
    "11": "r",
    "12": "m",
    "13": "c",
    "14": "y",
    "15": "k",
    "16": "peru",
    "17": "dodgerblue",
    "18": "brown",
    "19": "lightsalmon",
}


class ArtistContainer:
    """Base class for containing plotting artists and related functionality."""

    def __init__(self, ax, color, label):
        """Init function of the ArtistContainer base class.

        args:
            ax: (plt.axis object) The plt.axis object to draw on.
            color: (string) The color of the visualization
            label: (string) The label to be used for this vehicle
        """

        self._ax = ax
        self._color = color
        self._label = label
        self._legend = None

        self._artists = []
        self._create_artists()

    def _create_artists(self):
        """Function that plots and hence creates the artists.
        Should be implemented in the child classes.
        """
        raise NotImplementedError

    def set_legend(self, legend):
        """Sets the label that corresponds to this vehicle."""
        self._legend = legend

    def get_visibility(self):
        """Gets the visibility of the artists."""

        if self._artists != []:
            return self._artists[0].get_visible()
        else:
            return False

    def set_visibility(self, visibility=True):
        """Sets the visibility of all the artists."""

        for artist in self._artists:
            artist.set_visible(visibility)

        # setting the legend to shady if the vehicle is disabled currently:
        if self._legend is not None:
            if visibility:
                self._legend.set_alpha(1.0)
            else:
                self._legend.set_alpha(0.2)

    def invert_visibility(self):
        """Changes the visibility of the artists. If they were visible,
        They will become invisible and the other way around.
        """

        self.set_visibility(not self.get_visibility())


class VehicleArtist(ArtistContainer):
    """Class for containing the plotting artists of a vehicle."""

    def __init__(
        self,
        ax,
        fig,
        label,
        hist=None,
        pred=None,
        gt=None,
        boundaries=None,
        covariance=None,
        info=None,
        color="blue",
    ):
        """
        Initializes a VehicleArtist object.

        Args:
            fig: The figure object.
            hist: (np.array(), shape=(N, 2)) The history trajectory.
            pred: (np.array(), shape=(N, 2)) The prediction trajectory.
            gt: (np.array(), shape=(N, 2)) The ground truth trajectory.
            boundaries: (np.array(), shape=(2, M, 2)) The boundaries.
            covariance: (np.array(), shape=(N, 2, 2)) The covariance matrices of the uncertain prediction
            info: (string) Extra info to print next to each vehicle
        """

        self._fig = fig
        self._hist = hist
        self._pred = pred
        self._gt = gt
        self._boundaries = boundaries
        self._covariance = covariance
        self._info = info

        self._timestep_marker = None
        self._text_artist = None

        # Initializes the parent object and
        # calls the _create_artists() method.
        super().__init__(ax=ax, color=color, label=label)

        self._fig.canvas.mpl_connect("pick_event", self._on_pick)

    def _create_artists(self):
        """Draws the artists based on the data that was given."""

        # drawing the boundaries:
        if self._boundaries is not None:
            self._ax.plot(
                self._boundaries[0, :, 0], self._boundaries[0, :, 1], "k-", zorder=1
            )
            self._ax.plot(
                self._boundaries[1, :, 0], self._boundaries[1, :, 1], "k-", zorder=1
            )

        # plotting the history:
        if self._hist is not None:
            self._artists.append(
                self._ax.plot(
                    self._hist[0, :],
                    self._hist[1, :],
                    color=self._color,
                    linestyle="dashed",
                    zorder=2,
                )[0]
            )

        # plotting the ground truth:
        if self._gt is not None:
            self._artists.append(
                self._ax.plot(
                    self._gt[0, :],
                    self._gt[1, :],
                    color=self._color,
                    linestyle="dashed",
                    zorder=3,
                )[0]
            )

        # plotting the prediction:
        if self._pred is not None:
            self._artists.append(
                self._ax.plot(
                    self._pred[0, :],
                    self._pred[1, :],
                    color=self._color,
                    linestyle="solid",
                    label=self._label,
                    zorder=4,
                )[0]
            )

            # drawing a circle where the car is:
            self._artists.append(
                self._ax.plot(
                    self._pred[0, 0],
                    self._pred[1, 0],
                    color=self._color,
                    marker="o",
                    zorder=5,
                )[0]
            )

            # enable clicking on the circle:
            self._artists[-1].set_picker(True)
            self._artists[-1].set_pickradius(5)

        # drawing the covariance:
        if self._covariance is not None:
            draw_with_uncertainty(
                [[list(self._pred[0, :]), list(self._pred[1, :])]],
                self._covariance,
                self._ax,
            )

        # creating the info text:
        if (self._info is not None) and (self._pred is not None):
            self._text_artist = self._ax.text(
                self._pred[0, 0], self._pred[1, 0], self._info, zorder=10
            )

            self._text_artist.set_visible(False)
            self._text_is_enabled = False

    def _on_pick(self, event):
        """method to call when a click event happens.
        If the clicked artist was one of the artists of the object,
        the info text is shown.
        """

        if event.artist in self._artists:
            if self._text_artist is not None:
                self._text_artist.set_visible(not self._text_artist.get_visible())
                self._text_is_enabled = self._text_artist.get_visible()

                self._fig.canvas.draw()

    def set_timestep_marker(self, ts):
        """Set an x marker to the ts^th timestep."""

        if self._pred is not None:
            if ts > self._pred.shape[1] - 1:
                ts = self._pred.shape[1] - 1

            if self._timestep_marker is None:
                self._timestep_marker = self._ax.plot(
                    self._pred[0, int(ts)],
                    self._pred[1, int(ts)],
                    color=self._color,
                    marker="x",
                    markersize=8,
                    zorder=6,
                )[0]
            else:
                self._timestep_marker.set_xdata(self._pred[0, int(ts)])
                self._timestep_marker.set_ydata(self._pred[1, int(ts)])

            self._timestep_marker.set_visible(self.get_visibility())

    def set_visibility(self, visibility=True):
        """Because of the timestep marker, this function has
        to be adjusted slightly.
        """

        if self._timestep_marker is not None:
            self._timestep_marker.set_visible(visibility)

        if self._text_artist is not None:
            if self._text_is_enabled:
                self._text_artist.set_visible(visibility)
            else:
                self._text_artist.set_visible(False)

        super().set_visibility(visibility)

    @property
    def text_is_enabled(self):
        """Property, whether the text is enabled or not."""

        return self._text_is_enabled

    def set_text_enabled(self, enabled):
        """sets whether the info text is enabled or not."""

        self._text_is_enabled = enabled


class VelocityArtist(ArtistContainer):
    """Class for containing the artists of plotting the ground truth and
    the predicted velocities.
    """

    def __init__(
        self, ax, label, pred_v=None, pred_t=None, gt_v=None, gt_t=None, color="blue"
    ):
        """Initializes a VehicleArtist object.

        args:
            ID: The id of the vehicle.
            pred_v: (np.array(), shape=(N,)) The predicted velocities.
            pred_t: (np.array(), shape=(N,)) The timesteps of the predicted velocities.
            gt_v: (np.array(), shape=(N,)) The ground truth velocities.
            gt_t: (np.array(), shape=(N,)) The timesteps of the ground truth velocities.
            color: (string) The color of the visualization
        """

        self._pred_v = pred_v
        self._pred_t = pred_t
        self._gt_v = gt_v
        self._gt_t = gt_t

        # Initializes the parent object and
        # calls the _create_artists() method.
        super().__init__(ax=ax, color=color, label=label)

    def _create_artists(self):
        """Plots the predicted and the ground truth velocities."""

        # plotting the prediction:
        if self._pred_v is not None and self._pred_t is not None:
            self._artists.append(
                self._ax.plot(
                    self._pred_t,
                    self._pred_v,
                    color=self._color,
                    linestyle="solid",
                    zorder=1,
                    label=self._label,
                )[0]
            )

        # plotting the ground truth:
        if self._gt_v is not None and self._gt_t is not None:
            self._artists.append(
                self._ax.plot(
                    self._gt_t,
                    self._gt_v,
                    color=self._color,
                    zorder=2,
                    linestyle="dashed",
                )[0]
            )


class SliderGroup:
    """Class for grouping a slider with forward and backward buttons
    and to provide some related functionality.
    """

    def __init__(
        self, fig, left, bottom, width, height, max_val, step, text, callback=None
    ):
        """Initializes a SliderGroup object.

        Args:
            fig: the plt.figure object.
            left: the left position in the window [0, 1]
            bottom: the position from the bottom of the window [0, 1]
            width: the width of the total group [0, 1]
            height: the height of the total group [0, 1]
            max_val: the maximum val of the slider
            step: The step size of the slider
            text: The text of the slider
            callback: The callback method to execute when the slider changes.
        """

        self._fig = fig
        self._left = left
        self._bottom = bottom
        self._width = width
        self._height = height
        self._max_val = max_val
        self._step = step
        self._text = text
        self._callback = callback

        self._init_widgets()

        if self._callback is not None:
            self._slider.on_changed(self._callback)

    def _init_widgets(self):
        """Creates the widgets: the 2 buttons and the slider."""

        l, b, w, h = self._left, self._bottom, self._width, self._height

        button1_ax = plt.axes([l + w / 2 - h, b + h / 2, h, h / 2])
        button2_ax = plt.axes([l + w / 2, b + h / 2, h, h / 2])
        self._slider_ax = plt.axes([l, b, w, h / 2])

        self._button_left = Button(button1_ax, "<")
        self._button_right = Button(button2_ax, ">")
        self._slider = Slider(
            self._slider_ax,
            self._text,
            valmin=0,
            valmax=self._max_val,
            valinit=0,
            valstep=self._step,
        )

        self._button_left.on_clicked(self._backward)
        self._button_right.on_clicked(self._forward)

    def _backward(self, event):
        """The callback to call, when button_left is clicked."""

        if self._slider.val > 0:
            self._slider.set_val(self._slider.val - self._step)

    def _forward(self, event):
        """The callback to call, when button_left is clicked."""

        if self._slider.val < self._max_val:
            self._slider.set_val(self._slider.val + self._step)

    def update_max_val(self, max_val):
        """Updates the maximum value for the slider. Unfortunatelly
        one can not update the range, a new slider has to be created.
        """

        # changing the current value of the slider if the original value was larger
        # than the new boundary:
        val = self._slider.val
        if val > max_val:
            val = max_val

        self._slider_ax.cla()

        self._max_val = max_val
        self._slider = Slider(
            self._slider_ax,
            self._text,
            valmin=0,
            valmax=self._max_val,
            valinit=0,
            valstep=self._step,
        )

        self._slider.set_val(val)

        if self._callback is not None:
            self._slider.on_changed(self._callback)

    @property
    def val(self):
        """Returns the current value of the slider"""
        return self._slider.val
