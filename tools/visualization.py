import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import numpy as np
import io


def visualize(
    prediction,
    hist,
    fut,
    left_bound,
    right_bound,
    ax,
    show=True,
    save=True,
    rmse=None,
    ax_no=None,
):
    ax.cla()
    if rmse:
        ax.text(0, -30, "RMSE: {0:.2f} m".format(rmse), fontsize=18)
    if ax_no is not None:
        left_bound = left_bound[:, ax_no, :].detach().numpy()
        right_bound = right_bound[:, ax_no, :].detach().numpy()
    else:
        left_bound = left_bound[:, 0, :].detach().numpy()
        right_bound = right_bound[:, 0, :].detach().numpy()
    if len(fut.shape) > 2:
        fut = fut[:, 0, :].detach().numpy()
    hist = hist[:, 0, :].detach().numpy()

    if type(prediction) is not np.ndarray:
        prediction = prediction.detach().numpy()

    fut_pos_list = prediction[:, 0, :2]
    sigma_x = 1 / (prediction[:, 0, 2] + sys.float_info.epsilon)
    sigma_y = 1 / (prediction[:, 0, 3] + sys.float_info.epsilon)
    rho = prediction[:, 0, 4]

    sigma_cov = np.array(
        [
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2],
        ]
    )

    # Swap axes to shape (50,2,2)
    sigma_cov = sigma_cov.swapaxes(0, 2)
    sigma_cov = sigma_cov.swapaxes(1, 2)

    draw_with_uncertainty([fut_pos_list], [sigma_cov], ax)

    # ax.plot(fut_pos_list[:,0], fut_pos_list[:,1], 'g-')
    ax.plot(hist[:, 0], hist[:, 1], "r--", label="Input - Past coordinates ego vehicle")
    ax.plot(fut[:, 0], fut[:, 1], "b--", label="Output - Ground Truth")
    ax.plot(left_bound[:, 0], left_bound[:, 1], "k-", label="Input - Track boundaries")
    ax.plot(right_bound[:, 0], right_bound[:, 1], "k-")
    ax.axis("equal")
    ax.set_xlabel("x in m", fontsize=16)
    ax.set_ylabel("y in m", fontsize=16)
    ax.legend(loc="upper left")

    if show:
        plt.pause(1e-3)

    if save:
        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        return buf


def visualize_multi(prediction, hist, fut, left_bound, right_bound, axes, ax_no):
    """Allows to visualize multiple predictions in subplots.

    Args:
        prediction ([type]): [description]
        hist ([type]): [description]
        fut ([type]): [description]
        left_bound ([type]): [description]
        right_bound ([type]): [description]
        axes ([type]): [description]
        ax_no ([type]): [description]
    """

    if ax_no < len(axes):
        axes[ax_no].cla()

        visualize(
            prediction, hist, fut, left_bound, right_bound, axes[ax_no], ax_no=ax_no
        )


def confidence_ellipse(
    mu,
    cov,
    ax,
    n_std=3.0,
    facecolor="None",
    edgecolor="fuchsia",
    linestyle="--",
    label_ell=None,
    **kwargs
):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.

    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radiuses.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """

    mu_x = mu[0]
    mu_y = mu[1]

    pearson = cov[0, 1] / (np.sqrt(cov[0, 0] * cov[1, 1]) + sys.float_info.epsilon)
    # Using a special case to obtain the eigenvalues of this
    # two-dimensionl dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse(
        (0, 0),
        width=ell_radius_x * 2,
        height=ell_radius_y * 2,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linestyle=linestyle,
        alpha=0.5,
        **kwargs
    )

    # Calculating the stdandard deviation of x from
    # the squareroot of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std

    # calculating the stdandard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std

    transf = (
        transforms.Affine2D()
        .rotate_deg(45)
        .scale(scale_x, scale_y)
        .translate(mu_x, mu_y)
    )

    ellipse.set_transform(transf + ax.transData)
    ax.add_patch(ellipse)
    if label_ell is not None:
        ellipse.set(label=label_ell)
    return None


def draw_with_uncertainty(fut_pos_list, fut_cov_list, ax):
    fut_pos_list = np.array(fut_pos_list)
    fut_cov_list = np.array(fut_cov_list)

    if fut_pos_list.shape[1] < fut_pos_list.shape[2]:
        fut_pos_list = np.swapaxes(fut_pos_list, 1, 2)

    if fut_pos_list.shape[1] > 50:
        fut_pos_list = fut_pos_list[:, :50, :]

    for i, fut_pos in enumerate(fut_pos_list):
        ax.plot(
            fut_pos[:, 0],
            fut_pos[:, 1],
            "*c",
            markersize=2,
            alpha=0.8,
            zorder=15,
            label="Output - Prediction",
        )
        for j, pos in enumerate(fut_pos):
            confidence_ellipse(
                pos, fut_cov_list[i][j], ax, n_std=3.0, facecolor="yellow"
            )
        for j, pos in enumerate(fut_pos):
            confidence_ellipse(
                pos, fut_cov_list[i][j], ax, n_std=2.0, facecolor="orange"
            )
        for j, pos in enumerate(fut_pos):
            confidence_ellipse(pos, fut_cov_list[i][j], ax, n_std=1.0, facecolor="red")


def visualize_objects(ax, objector, n_std=3, show=True):
    if not objector.tracker.bool_detection_input:
        return

    mode = objector.visualization

    right_boundary = objector.right_bound
    left_boundary = objector.left_bound

    ax.cla()
    ax.plot(right_boundary[:, 0], right_boundary[:, 1], "k-")
    ax.plot(left_boundary[:, 0], left_boundary[:, 1], "k-", label="Track boundaries")

    xx = []
    yy = []

    col_list = [
        "green",
        "blue",
        "magenta",
        "yellow",
        "cyan",
        "wheat",
        "lime",
        "silver",
        "olive",
        "coral",
        "beige",
        "pink",
    ]

    col_iter = iter(col_list)

    if mode == "filter":
        window_plot_size = 4
        for obj_id, vals in objector.tracker.delay_visz.items():
            try:
                col_obj = next(col_iter)
            except Exception:
                col_iter = iter(col_list)
                col_obj = next(col_iter)

            label_est = "estimate, ID = {:d}".format(int(obj_id))

            ax.plot(
                list(objector.tracker.observation_storage[obj_id]["x"])[
                    len(vals["x_est"]) :
                ],
                list(objector.tracker.observation_storage[obj_id]["y"])[
                    len(vals["y_est"]) :
                ],
                "-",
                label=None,
                color=col_obj,
            )
            ax.plot(vals["x_est"], vals["y_est"], "-.", label=label_est, color=col_obj)
            ax.plot(vals["x_est"][-1], vals["y_est"][-1], "x", color=col_obj)

            # Take forelast position for residuum, as final prediction step was conducted for estimation
            if len(vals["x_upt"]) > 1:
                dx = vals["x_est"][0] - vals["x_upt"][-2]
                dy = vals["y_est"][0] - vals["y_upt"][-2]
            label_upt = (
                "update, ID = {:d}, $\Delta$x = {:.02f}m, $\Delta$y = {:.02f}m".format(
                    int(obj_id), dx, dy
                )
            )
            ax.plot(vals["x_upt"], vals["y_upt"], "-", label=label_upt, color=col_obj)
            ax.plot(vals["x_upt"][0], vals["y_upt"][0], "x", color=col_obj)
            ax.plot(vals["x_upt"][-1], vals["y_upt"][-1], "x", color=col_obj)

            xx.append(vals["x_upt"][-1])
            yy.append(vals["y_upt"][-1])
            xx.append(vals["x_upt"][0])
            yy.append(vals["y_upt"][0])
    else:
        if mode.find("observation") >= 0:
            plot_dict = objector.tracker.observation_storage
        elif mode == "prediction":
            plot_dict = objector.tracker.pred_dict

        for obj_id, obs in plot_dict.items():
            if mode != "prediction":
                obs = {
                    var: [obj_state[j] for obj_state in obs["state"]]
                    for j, var in enumerate(objector.tracker._state_variables)
                }

            if obj_id == "ego":
                col_obj = "r"
                obs["cov"] /= 100.0
            else:
                try:
                    col_obj = next(col_iter)
                except Exception:
                    col_iter = iter(col_list)
                    col_obj = next(col_iter)

            if obs.get("valid", True):
                xx.append(obs["x"][0])
                yy.append(obs["y"][0])
                if "cov" in obs.keys():
                    P = obs["cov"][-1]
                    spec_idx = -1
                    ax.plot(obs["x"][spec_idx], obs["y"][spec_idx], "x", color=col_obj)
                else:
                    filter = objector.tracker.obj_filters.get(obj_id, None)
                    if filter is None:
                        P = np.eye(2) * 0.01  # Just template for ego object
                    else:
                        P = filter.P[:2, :2]
                if obj_id == "ego":
                    ll_obj = None
                else:
                    p_norm = np.linalg.norm(P)
                    ll_obj = "ID: {}, |P|={:.2f}".format(obj_id, p_norm)
                    spec_idx = 0
                ax.plot(obs["x"], obs["y"], label=ll_obj, color=col_obj)
                ax.plot(obs["x"][0], obs["y"][0], "x", color=col_obj)

                if mode == "prediction":
                    ax.plot(obs["x_rail"], obs["y_rail"], "--", color=col_obj)
                    window_plot_size = 300

                mu = [obs["x"][spec_idx], obs["y"][spec_idx]]
                confidence_ellipse(
                    mu=mu,
                    cov=P,
                    ax=ax,
                    n_std=n_std,
                    facecolor="g",
                    alpha=0.3,
                )
            else:
                xx.append(obs["x"])
                yy.append(obs["y"])
                ll_obj = "ID: {} - invalid".format(obj_id)
                ax.plot(obs["x"], obs["y"], "x", label=ll_obj, color=col_obj)

    bool_limits_ego = bool(
        mode.find("glob") < 0 and mode.find("pred") < 0 and mode.find("filter") < 0
    )

    if "x" in objector.tracker.ego_state.keys():
        ax.plot(
            objector.tracker.ego_state["x"],
            objector.tracker.ego_state["y"],
            "o",
            color="r",
            markersize=7,
            label="Ego-state",
        )
        if bool_limits_ego:
            x_min = objector.tracker.ego_state["x"] - window_plot_size
            x_max = objector.tracker.ego_state["x"] + window_plot_size
            y_min = objector.tracker.ego_state["y"] - window_plot_size
            y_max = objector.tracker.ego_state["y"] + window_plot_size
        elif mode.find("filter") < 0 and len(xx) > 0:
            xx.append(objector.tracker.ego_state["x"])
            yy.append(objector.tracker.ego_state["y"])

    try:
        _ = x_min  # check if x_min is available from bool_limit_ego
    except Exception:
        try:  # check if xx is avaible as list from all objects
            # max_sensor_range = max([val[2] for val in objector.measure_spec.values()])
            x_min = np.min(xx) - window_plot_size
            x_max = np.max(xx) + window_plot_size
            y_min = np.min(yy) - window_plot_size
            y_max = np.max(yy) + window_plot_size
        except Exception:
            return  # do not plot anything

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect("equal", adjustable="box", anchor="C")
    ax.set_xlabel("x in m", fontsize=16)
    ax.set_ylabel("y in m", fontsize=16)
    ax.legend(loc="upper left")
    ax.grid(True)

    if show:
        plt.pause(1e-3)
