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
import matplotlib

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

plt.rcParams.update({"font.size": 14})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"font.family": "Times New Roman"})

import numpy as np
from mix_net.mix_net.src.boundary_generator import BoundaryGenerator

from tools.visualize_logfiles import PredictionLogVisualizer

ERROR_ANALYSIS_TIME_STEPS = [9, 19, 29, 39, 49]
TRACK_PATH = "mix_net/mix_net/data/map/traj_ltpl_cl_IMS_GPS.csv"
CONFIG_PATH = "mix_net/mix_net/config/main_params.ini"
SAVE_DIR = "data/evaluation_data/smoothness"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
MIX_NET_COL = TUM_BLUE
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
INDY_NET_COL = TUM_ORAN
TUM_GREEN = (162 / 255, 173 / 255, 0 / 255)
RAIL_BASED_COL = TUM_GREEN
HIST_COL = "black"
HIST_LS = "solid"
GT_COL = "black"
GT_LS = "dashed"
LINEWIDTH = 2.0
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)


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
            label="History",
        )
    _ax.plot(
        lb_sample[:, 0],
        lb_sample[:, 1],
        "x",
        color=BOUND_COL,
        label="Boundary",
        linestyle="solid",
        linewidth=LINEWIDTH,
    )
    _ax.plot(
        rb_sample[:, 0],
        rb_sample[:, 1],
        "x",
        color=BOUND_COL,
        linestyle="solid",
        linewidth=LINEWIDTH,
    )
    _ax.plot(
        gt[0, :],
        gt[1, :],
        color=GT_COL,
        linestyle=GT_LS,
        label="Ground Truth",
        linewidth=LINEWIDTH,
    )
    _ax.plot(
        pred_rail[:, 0],
        pred_rail[:, 1],
        color=RAIL_BASED_COL,
        linestyle="solid",
        label="Rail-based Prediction",
        linewidth=LINEWIDTH,
    )
    _ax.plot(
        pred[0, :],
        pred[1, :],
        color=INDY_NET_COL,
        linestyle="solid",
        label="Benchmark Model",
        linewidth=LINEWIDTH,
    )
    _ax.plot(
        pred_mix_net[0, :],
        pred_mix_net[1, :],
        color=MIX_NET_COL,
        linestyle="solid",
        label="MixNet",
        linewidth=LINEWIDTH,
    )


def visz_frame(visualizer_benchmark, visualizer_mixnet, pred_id: str):
    """Visualize example frame from logs.

    Visualized are Ground Truth, History, Bounds, MixNet, IndyNet and Rail-based Prediction.

    Args:
        visualizer (PredictionLogVisualizer): Visualizer class to process logged data.
        pred_id (str): ID of prediction to visualize
    """

    _fig = plt.figure(figsize=(14, 3.2))
    _ax = _fig.add_subplot(111)

    _, gt, pred_bm, hist_pid = get_single_log(visualizer_benchmark, pred_id)
    boundaries_pid, _, pred_mix_net, _ = get_single_log(visualizer_mixnet, pred_id)

    lb_sample, rb_sample, pred_rail = get_rail_prediction(
        visualizer_benchmark._recovered_params, boundaries_pid, hist_pid
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

    insax_1 = _ax.inset_axes([0.6, 0.38, 0.18, 0.6])
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

    insax_xmin = 260
    insax_xmax = 315
    insax_ymin = -2380
    insax_ymax = -2350

    insax_1.set_xlim(insax_xmin, insax_xmax)
    insax_1.set_ylim(insax_ymin, insax_ymax)
    insax_1.set_xticklabels([])
    insax_1.set_yticklabels([])

    _ax.indicate_inset_zoom(insax_1, edgecolor="black")

    # setting axis labels of ax1:
    _ax.legend(loc="upper left")
    _ax.set_xlabel("$x$ in m")
    _ax.set_ylabel("$y$ in m")
    plt.axis([-400, 360, -2425, -2310])
    plt.gca().set_aspect("equal", adjustable="box")
    _ax.grid(True)

    plt.savefig(os.path.join("assets", "smoothness.svg"))
    plt.savefig(os.path.join(SAVE_DIR, "smoothness.pdf"))
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
    # Ground Truth:
    gt = visualizer._get_gt_from_trajs(vehicle_id, t_abs, pred.shape[1])

    return boundaries_pid, gt, pred, hist_pid


def get_rail_prediction(recovered_params, boundaries_pid, hist_pid):
    """Determine rail-based prediction."""
    # get boundaries
    _bg = BoundaryGenerator(recovered_params)
    lb_sample, rb_sample = _bg.get_boundaries_single(hist_pid[:, -1])

    # determine rail-based prediction
    track_dist = np.linalg.norm(lb_sample[0, :] - rb_sample[0, :])
    rb_dist = np.linalg.norm(hist_pid[:, -1] - rb_sample[0, :])

    weight_lb = rb_dist / track_dist
    weight_rb = 1 - weight_lb

    pred_rail = weight_lb * boundaries_pid[0] + weight_rb * boundaries_pid[1]

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
