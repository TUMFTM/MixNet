"""
This debug visualization tool takes (autoamically) the latest logfile
in the logs/ folder and visualizes the predictions.

    Arguments:
    --velocity: Plot the velocity profile.
    --dataset: Create a dataset for training etc. out of the ground truth data
"""

# Settings
ZOOM_VEH_ID = None  # Vehicle ID that should be zoomed to
import os
import sys
import argparse

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

import matplotlib.pyplot as plt

import numpy as np
from mix_net.mix_net.src.boundary_generator import BoundaryGenerator

from tools.visualize_logfiles import PredictionLogVisualizer
from tools.evaluation_line_plot import update_matplotlib

ERROR_ANALYSIS_TIME_STEPS = [9, 19, 29, 39, 49]
TRACK_PATH = "mix_net/mix_net/data/map/traj_ltpl_cl_IMS_GPS.csv"
CONFIG_PATH = "mix_net/mix_net/config/main_params.ini"
SAVE_DIR = "data/evaluation_data/smoothness"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
TUM_LIGHT_BLUE = (100 / 255, 160 / 255, 200 / 255)
TUM_LIGHTER_BLUE = (152 / 255, 198 / 255, 234 / 255)
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
TUM_BLACK = (0, 0, 0)
GRAY = (104 / 255, 104 / 255, 104 / 255)
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)

COLUMNWIDTH = 3.5
PAGEWIDTH = COLUMNWIDTH * 2
LABELFONTSIZE = 8  # IEEE is 8

FIGSIZE = (PAGEWIDTH, PAGEWIDTH / 2)

INDY_NET_COL = GRAY
RAIL_BASED_COL = TUM_ORAN
MIX_NET_COL = TUM_BLUE
HIST_COL = TUM_BLACK
HIST_LS = "solid"
GT_COL = TUM_BLACK
GT_LS = "dashed"
BOUND_COL = BOUND_COL

update_matplotlib()


def plot_all(
    _ax,
    boundaries_pid,
    lb_sample,
    rb_sample,
    gt,
    pred_rail,
    pred,
    pred_mix_net,
    hist=None,
):
    """Plot bounds, history, preictions (mixnet, indynet, rail) and ground truth."""
    if hist is not None:
        _ax.plot(
            hist[0, :],
            hist[1, :],
            color=HIST_COL,
            linestyle="solid",
            label="IN",
        )
    _ax.plot(
        lb_sample[:, 0],
        lb_sample[:, 1],
        "x",
        color=BOUND_COL,
        linestyle="solid",
    )
    _ax.plot(
        rb_sample[:, 0],
        rb_sample[:, 1],
        "x",
        color=BOUND_COL,
        label="IN",
        linestyle="solid",
    )
    _ax.plot(
        gt[0, :],
        gt[1, :],
        color=GT_COL,
        linestyle=GT_LS,
        label="GT",
    )
    _ax.plot(
        pred_rail[0, :],
        pred_rail[1, :],
        color=RAIL_BASED_COL,
        linestyle="solid",
        label="RAIL",
    )
    _ax.plot(
        pred[0, :],
        pred[1, :],
        color=INDY_NET_COL,
        linestyle="solid",
        label="BENCHMARK",
    )
    _ax.plot(
        pred_mix_net[0, :],
        pred_mix_net[1, :],
        color=MIX_NET_COL,
        linestyle="solid",
        label="MIXNET",
    )


def visz_frame(
    visualizer_benchmark, visualizer_mixnet, pred_id: str, with_raceline: bool = False
):
    """Visualize example frame from logs.

    Visualized are Ground Truth, History, Bounds, MixNet, IndyNet and Rail-based Prediction.

    Args:
        visualizer (PredictionLogVisualizer): Visualizer class to process logged data.
        pred_id (str): ID of prediction to visualize
    """

    _fig = plt.figure(figsize=FIGSIZE)
    _ax = _fig.add_subplot(111)

    _, gt, pred_bm, hist_pid = get_single_log(visualizer_benchmark, pred_id)
    boundaries_pid, _, pred_mix_net, _ = get_single_log(visualizer_mixnet, pred_id)

    lb_sample, rb_sample, pred_rail = get_rail_prediction(
        visualizer_benchmark._recovered_params,
        pred_bm,
        hist_pid,
        with_raceline=with_raceline,
    )

    # Visualize benchmark
    plot_all(
        _ax,
        boundaries_pid,
        lb_sample,
        rb_sample,
        gt,
        pred_rail,
        pred_bm,
        pred_mix_net,
        hist=hist_pid,
    )

    _ax.set_xlim(-227, 357)
    _ax.set_ylim(-2417, -2280)

    insax_1 = _ax.inset_axes([0.6, 0.6, 0.25, 0.7])
    insax_1.axis("equal")

    plot_all(
        insax_1,
        boundaries_pid,
        lb_sample,
        rb_sample,
        gt,
        pred_rail,
        pred_bm,
        pred_mix_net,
    )

    insax_xmin = 270
    insax_xmax = 315
    insax_ymin = -2375
    insax_ymax = -2340

    insax_1.set_xlim(insax_xmin, insax_xmax)
    insax_1.set_ylim(insax_ymin, insax_ymax)
    insax_1.set_xticklabels([])
    insax_1.set_yticklabels([])
    _ax.indicate_inset_zoom(insax_1, edgecolor="black")

    insax_2 = _ax.inset_axes([0.1, 0.6, 0.4, 0.7])
    insax_2.axis("equal")

    plot_all(
        insax_2,
        boundaries_pid,
        lb_sample,
        rb_sample,
        gt,
        pred_rail,
        pred_bm,
        pred_mix_net,
        hist=hist_pid,
    )

    insax_xmin = -47
    insax_xmax = 67
    insax_ymin = -2415
    insax_ymax = -2380

    insax_2.set_xlim(insax_xmin, insax_xmax)
    insax_2.set_ylim(insax_ymin, insax_ymax)
    insax_2.set_xticklabels([])
    insax_2.set_yticklabels([])

    _ax.indicate_inset_zoom(insax_2, edgecolor="black")

    # setting axis labels of ax1:
    # _ax.legend(loc="upper left")
    _ax.set_xlabel("$x$ in m")
    _ax.set_ylabel("$y$ in m")
    # plt.axis([-400, 360, -2425, -2310])
    plt.gca().set_aspect("equal", adjustable="box")
    _ax.grid(True)

    if with_raceline:
        strstr = "_raceline"
    else:
        strstr = ""
    plt.savefig(os.path.join("assets", "scenario_sample" + strstr + ".svg"))
    plt.savefig(os.path.join(SAVE_DIR, "scenario_sample" + strstr + ".pdf"))
    plt.show()


def get_single_log(visualizer, pred_id):
    t_abs, _, hist, _, pred_dict, _ = visualizer._all_log_data[int(pred_id)]

    # The prediction:
    pred_x = pred_dict[pred_id]["x"]
    pred_y = pred_dict[pred_id]["y"]
    pred = np.vstack((pred_x, pred_y))
    vehicle_id = pred_dict[pred_id]["vehicle_id"]

    # division by 1e9 is needed, because mod_object sends nanosecs:
    t_abs = float(pred_dict[pred_id]["t_abs_perception"]) / 1e9

    # If there is no real history log, it is created from the gt_dict:
    hist_pid = np.array(hist[pred_id]).T
    # Remove all zeros (that were added before feeding it to the nets):
    hist_pid = np.ma.masked_equal(hist_pid, 0)

    # get bounds
    (
        left_bound,
        right_bound,
    ) = visualizer._boundary_generator.get_bounds_between_points(
        pred[:, 0], pred[:, -1]
    )
    boundaries_pid = np.array([left_bound, right_bound])

    # vel = np.mean(np.linalg.norm(np.diff(np.stack([pred_dict[pred_id]["x"][:3], pred_dict[pred_id]["y"][:3]])), axis=0))/0.1
    # Ground Truth:
    gt = visualizer._get_gt_from_trajs(vehicle_id, t_abs, pred.shape[1])

    return boundaries_pid, gt, pred, hist_pid


def get_rail_prediction(recovered_params, pred, hist_pid, with_raceline=False):
    """Determine rail-based prediction."""
    # get boundaries
    _bg = BoundaryGenerator(recovered_params)
    lb_sample, rb_sample = _bg.get_boundaries_single(
        hist_pid[:, -1], with_raceline=False
    )

    # determine rail-based prediction
    pred_rail = _bg.get_rail_pred(
        np.array(hist_pid[:, -1]), pred=pred, with_raceline=with_raceline
    )

    return lb_sample, rb_sample, pred_rail


if __name__ == "__main__":
    # Parse arguments:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logdir_benchmark",
        type=str,
        default="data/evaluation_data/2_indynet/12_13_21/",
        help="The logdir of benchmark model to get logs from",
    )
    parser.add_argument(
        "--logdir_mixnet",
        type=str,
        default="data/evaluation_data/1_mixnet/14_23_40/",
        help="The logdir of mixnet to get logs from",
    )
    parser.add_argument(
        "--pred_id",
        type=str,
        default="760",
        help="prediction ID which is analyzed",
    )
    args = parser.parse_args()

    args.velocity = False
    args.analyze = False
    args.yolo = True

    # create visualizer object:

    args.logdir = args.logdir_benchmark
    visualizer_benchmark = PredictionLogVisualizer(
        args=args,
        zoom_veh_id=None,
        draw_uncertainty=False,
        yolo=True,
    )

    args.logdir = args.logdir_mixnet
    visualizer_mixnet = PredictionLogVisualizer(
        args=args,
        zoom_veh_id=None,
        draw_uncertainty=False,
        yolo=True,
    )

    # visualize example of smooth prediction output
    visz_frame(visualizer_benchmark, visualizer_mixnet, pred_id=args.pred_id)
    visz_frame(
        visualizer_benchmark,
        visualizer_mixnet,
        pred_id=args.pred_id,
        with_raceline=True,
    )
