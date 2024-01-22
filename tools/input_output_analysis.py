"""MinNet input-out testing"""
import sys
import os

import numpy as np
import json
import csv
import copy
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath("")))

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

FIGSIZE = (PAGEWIDTH, 9 / 16 * PAGEWIDTH)

INDY_NET_COL = GRAY
RAIL_BASED_COL = TUM_ORAN
MIX_NET_COL = TUM_BLUE
HIST_COL = TUM_BLACK
HIST_LS = "solid"
GT_COL = TUM_BLACK
GT_LS = "dashed"
BOUND_COL = BOUND_COL

from mix_net.mix_net.src.mix_net import MixNet
from mix_net.mix_net.utils.geometry import angle_between, transform_trajectory
from mix_net.mix_net.src.boundary_generator import BoundaryGenerator
from tools.evaluation_line_plot import update_matplotlib

update_matplotlib()

MAP_PATH = "mix_net/mix_net/data/map/traj_ltpl_cl_IMS_GPS.csv"
# loading the model:
NET_PARAMS_PATH = "mix_net/mix_net/data/inference_model/mix_net/net_params.json"
MODEL_WEIGHTS_PATH = "mix_net/mix_net/data/inference_model/mix_net/model.pth"

SAVE_DIR = "data/evaluation_data/input_output"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

np.random.seed(seed=42)
MAIN_COL_WEIGHTS = np.random.uniform(0.0, 1.0, size=(100, 1))
OTHER_COL_WEIGHTS = np.random.uniform(0.0, 1.0, size=(100, 3))


def create_synthetic_weights(as_torch=False, mixnet_handler=None, replace_cl=False):
    weights_in = np.zeros((400, 4))

    for i in range(4):
        other_col_idxs = list(range(0, i)) + list(range(i + 1, 4))

        main_col_weights = MAIN_COL_WEIGHTS
        other_col_weights = OTHER_COL_WEIGHTS

        # normalizing the other column weights, so that the sum is 1:
        other_col_weights *= (1.0 - main_col_weights) / np.sum(
            other_col_weights, axis=1
        )[:, np.newaxis]

        # filling up the whole matrix:
        i0_min = i * 100
        i0_max = (i + 1) * 100

        weights_in[i0_min:i0_max, i] = main_col_weights[:, 0]
        weights_in[i0_min:i0_max, other_col_idxs] = other_col_weights

    weights_out = []
    error_out = []
    for weight_in in weights_in:
        return_tuple = get_model_output(
            model,
            input_generator,
            weight_in,
            last_index,
            v,
            mixnet_handler=mixnet_handler,
            replace_cl=replace_cl,
        )
        weights_out.append(return_tuple[0])
        if return_tuple[1] is not None:
            error_out.append(return_tuple[1])

    errors_np = np.array(error_out)

    if not as_torch:
        weights_in = np.array(weights_in)
        weights_out = np.array(weights_out)

    return weights_in, weights_out, errors_np


def read_map(map_data_file):
    data = []
    with open(map_data_file, "r") as fp:
        csvreader = csv.reader(fp, delimiter=";")

        # dont read the first 3 rows:
        counter = 0
        for line in csvreader:
            if counter >= 3:
                data.append(line)
            counter += 1

    data = np.array(data).astype(np.float32)

    # the very first and the very last datapoints are the same, this is why the -1 indexing needed:
    centerline = data[:-1, 0:2]
    right_bound = (
        data[:-1, 0:2] + data[:-1, 4:6] * np.vstack((data[:-1, 2], data[:-1, 2])).T
    )
    left_bound = (
        data[:-1, 0:2] - data[:-1, 4:6] * np.vstack((data[:-1, 3], data[:-1, 3])).T
    )
    raceline = (
        data[:-1, 0:2] + data[:-1, 4:6] * np.vstack((data[:-1, 6], data[:-1, 6])).T
    )

    return left_bound, right_bound, centerline, raceline


def plot_bounds(left, right):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.axis("equal")

    # plt.plot(center[:, 0], center[:, 1], 'b')
    ax.plot(right[:, 0], right[:, 1], "k")
    ax.plot(left[:, 0], left[:, 1], "k")
    # ax.plot(race[:, 0], race[:, 1], 'b--')

    n = 810
    ax.plot(left[n, 0], left[n, 1], "ko")
    ax.plot(left[n + 10, 0], left[n + 10, 1], "ko")
    ax.plot(left[n - 10, 0], left[n - 10, 1], "ko")

    plt.show()


class InputProvider:
    def __init__(self, map_path):
        (self._lb, self._rb, self._center, self._race) = read_map(map_path)

        params = {
            "MODEL_PARAMS": {
                "dist": 20,
                "view": 400,
            },
            "track_path": map_path,
        }
        # boundary generator:
        self._bg = BoundaryGenerator(params=params)

    def get_transformed_input(self, weights, last_index, v):
        # getting not normalized and rotated
        hist, left_bound, right_bound, fut = self.get_non_transformed_input(
            weights, last_index, v
        )

        # transforming the trajectory:
        translation = copy.deepcopy(hist[-1, :])

        # getting the rotation:
        rotation = angle_between(left_bound[1, :] - left_bound[0, :], [1, 0])

        # Transforming the history and the boundaries:
        hist = transform_trajectory(hist, translation, rotation)
        fut = transform_trajectory(fut, translation, rotation)
        left_bound = transform_trajectory(left_bound, translation, rotation)
        right_bound = transform_trajectory(right_bound, translation, rotation)

        return hist, left_bound, right_bound, fut, translation, rotation

    def get_non_transformed_input(self, weights, last_index, v):
        # the superpositioned curve:
        line = (
            weights[0] * self._lb
            + weights[1] * self._rb
            + weights[2] * self._center
            + weights[3] * self._race
        )

        # calculating the distance along the curve:
        vect_diff = line[1:, :] - line[:-1, :]
        dists = np.zeros((line.shape[0],))
        dists[1:] = np.cumsum(np.linalg.norm(vect_diff, axis=1))

        # the distances at which the curve is going to be sampled:
        sample_dists = np.ones((30,)) * dists[last_index]
        sample_dists_fut = np.ones((51,)) * dists[last_index]

        rel_dists = np.arange(30) * 0.1 * v
        rel_dists = rel_dists[::-1]
        rel_dists_fut = np.arange(51) * 0.1 * v

        sample_dists -= rel_dists
        sample_dists_fut += rel_dists_fut

        hist = np.zeros((30, 2))
        hist[:, 0] = np.interp(sample_dists, dists, line[:, 0])
        hist[:, 1] = np.interp(sample_dists, dists, line[:, 1])

        fut = np.zeros((51, 2))
        fut[:, 0] = np.interp(sample_dists_fut, dists, line[:, 0])
        fut[:, 1] = np.interp(sample_dists_fut, dists, line[:, 1])

        # Get Boundaries
        left_bound, right_bound = self._bg.get_boundaries_single(hist[-1, :])

        return hist, left_bound, right_bound, fut


def get_model_output(
    model,
    input_generator,
    weights,
    last_index,
    v,
    mixnet_handler=None,
    replace_cl=False,
):
    if replace_cl:
        weights[0] += 0.5 * weights[2]
        weights[1] += 0.5 * weights[2]
        weights[2] = 0

    (
        hist,
        left_bound,
        right_bound,
        fut,
        translation,
        rotation,
    ) = input_generator.get_transformed_input(weights, last_index, v)

    # creating tensors:
    hist_t = torch.unsqueeze(torch.from_numpy(hist).float().to(model.device), 0)
    left_bound_t = torch.unsqueeze(
        torch.from_numpy(left_bound).float().to(model.device), 0
    )
    right_bound_t = torch.unsqueeze(
        torch.from_numpy(right_bound).float().to(model.device), 0
    )

    # feeding through the model:
    weights_out, vels, accels = model(hist_t, left_bound_t, right_bound_t)

    weights_out = weights_out.to("cpu").detach().numpy()
    if replace_cl:
        weights_out[0, 0] += 0.5 * weights_out[0, 2]
        weights_out[0, 1] += 0.5 * weights_out[0, 2]
        weights_out[0, 2] = 0

    if mixnet_handler is not None:
        vels = vels.to("cpu").detach().numpy()
        accels = accels.to("cpu").detach().numpy()

        x_mixes = (mixnet_handler._lines_x @ weights_out.T).T  # (num_vehs, line_length)
        y_mixes = (mixnet_handler._lines_y @ weights_out.T).T  # (num_vehs, line_length)
        arc_mixes = (
            mixnet_handler._line_arcs @ weights_out.T
        ).T  # (num_vehs, line_length + 1)

        obs_storage = {"1": {"v": v}}
        mixnet_handler._received_ids = ["1"]
        translations = np.array([translation])

        arc_dists = mixnet_handler._get_arc_dists(
            vels, accels, translations, x_mixes, y_mixes, arc_mixes, obs_storage
        )
        idx = 0
        pred_x = np.interp(
            arc_dists[idx, :],
            arc_mixes[idx, :],
            np.hstack((x_mixes[idx, :], x_mixes[idx, 0])),
        )

        pred_y = np.interp(
            arc_dists[idx, :],
            arc_mixes[idx, :],
            np.hstack((y_mixes[idx, :], y_mixes[idx, 0])),
        )
        pred = np.stack((pred_x, pred_y), axis=1)
        pred = mixnet_handler._correct_pred_beginning(pred, translations[idx, :])

        pred = transform_trajectory(pred, translation, rotation)

        diff = pred - fut
        rmse = np.linalg.norm(diff, axis=1)

        weights_out = np.squeeze(weights_out)
    else:
        rmse = None

    # reshape
    weights_out = np.squeeze(weights_out)

    return weights_out, rmse


def plot_lines(ax, left, right, hist, pred_gt, pred, raceline, last_index):
    """Helper function for plotting all these lines to a given axis."""

    ax.plot(left[760:930, 0], left[760:930, 1], color=BOUND_COL)
    ax.plot(right[760:930, 0], right[760:930, 1], color=BOUND_COL)

    ax.plot(
        hist[10:, 0],
        hist[10:, 1],
        color=HIST_COL,
        linestyle=HIST_LS,
    )

    ax.plot(
        pred_gt[last_index : last_index + 120, 0],
        pred_gt[last_index : last_index + 120, 1],
        color=GT_COL,
        linestyle=GT_LS,
    )

    ax.plot(
        pred[last_index : last_index + 120, 0],
        pred[last_index : last_index + 120, 1],
        color=MIX_NET_COL,
    )

    ax.plot(
        raceline[760:930, 0],
        raceline[760:930, 1],
        color=GRAY,
        linestyle="dotted",
    )


def create_example(input_generator, model, last_index=810, v=75.0):
    left, right, center, race = read_map(MAP_PATH)
    dummy_weights = np.array([0.4, 0.1, 0.3, 0.2])
    hist_gt, _, _, _ = input_generator.get_non_transformed_input(
        dummy_weights, last_index, v
    )

    pred_gt = (
        left * dummy_weights[0]
        + right * dummy_weights[1]
        + center * dummy_weights[2]
        + race * dummy_weights[3]
    )

    # prediction:
    weights_out, _ = get_model_output(
        model, input_generator, dummy_weights, last_index, v
    )

    pred = (
        left * weights_out[0]
        + right * weights_out[1]
        + center * weights_out[2]
        + race * weights_out[3]
    )

    # printing info:
    # print("weights in: {}".format(weights_in[idx]))
    print("weights in: {}".format(dummy_weights))
    print("weights out: {}".format(weights_out))

    # plotting
    fig = plt.figure(figsize=(PAGEWIDTH / 2, PAGEWIDTH / 2))
    ax = fig.gca()
    ax.axis("equal")

    plot_lines(ax, left, right, hist_gt, pred_gt, pred, race, last_index)

    # turning off axis ticks:
    ax.set_xlim(100, 450)
    ax.set_ylim(-1200, -850)
    ax.xaxis.set_ticks([100, 250, 400])
    ax.yaxis.set_ticks([-1200, -1100, -1000, -900])

    # inset axes 1:
    insax_1 = ax.inset_axes([0.01, 0.3, 0.7, 0.3])
    insax_1.axis("equal")

    plot_lines(
        insax_1,
        left,
        right,
        hist_gt,
        pred_gt,
        pred,
        race,
        last_index,
    )

    insax_1.set_xlim(130, 190)
    insax_1.set_ylim(-890, -865)
    insax_1.set_xticklabels([])
    insax_1.set_yticklabels([])

    ax.indicate_inset_zoom(insax_1, edgecolor="black")

    # inset axes 2:
    insax_2 = ax.inset_axes([0.72, 0.72, 0.27, 0.27])
    insax_2.axis("equal")

    plot_lines(insax_2, left, right, hist_gt, pred_gt, pred, race, last_index)

    insax_2.set_xlim(325, 350)
    insax_2.set_ylim(-975, -950)
    insax_2.set_xticklabels([])
    insax_2.set_yticklabels([])

    ax.indicate_inset_zoom(insax_2, edgecolor="black")

    # setting axis labels:
    ax.set_xlabel("$x$ in m")
    ax.set_ylabel("$y$ in m")

    # setting the legend:
    legend_elements = [
        Line2D(
            [0],
            [0],
            color=GT_COL,
            linestyle=GT_LS,
            label="Ground Truth",
        ),
        Line2D(
            [0],
            [0],
            color=MIX_NET_COL,
            linestyle="solid",
            label="Prediction",
        ),
        Line2D(
            [0],
            [0],
            color=GRAY,
            linestyle="dotted",
            label="Raceline",
        ),
        Line2D([0], [0], color=HIST_COL, linestyle="solid", label="History"),
    ]

    ax.legend(handles=legend_elements, loc="lower left")
    ax.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(SAVE_DIR, "curve_sample.pdf"))
    plt.show()


def get_loaded_model():
    with open(NET_PARAMS_PATH, "r") as fp:
        net_params = json.load(fp)

    model = MixNet(net_params)
    _ = model.load_model_weights(MODEL_WEIGHTS_PATH)
    return model


def plot_weight_distribution(mixnet_handler=None):
    weights_in_np, weights_out_np, errors_np = create_synthetic_weights(
        mixnet_handler=mixnet_handler
    )
    print("WEIGHT DISTRIBUTION")
    print("Mean L2-Norm is: {:.03f} m".format(np.mean(np.mean(errors_np, axis=0))))

    fig, (ax_left, ax_right, ax_center, ax_race) = plt.subplots(
        1, 4, figsize=(PAGEWIDTH, PAGEWIDTH / 4)
    )
    axs = [ax_left, ax_right, ax_center, ax_race]
    # titles = ["Left Boundary", "Right Boundary", "Centerline", "Raceline"]

    x_label_list = [
        "$\lambda_{\mathrm{LB, in}}$",
        "$\lambda_{\mathrm{RB, in}}$",
        "$\lambda_{\mathrm{CL, in}}$",
        "$\lambda_{\mathrm{RL, in}}$",
    ]
    y_label_list = [
        "$\lambda_{\mathrm{LB, out}}$",
        "$\lambda_{\mathrm{RB, out}}$",
        "$\lambda_{\mathrm{CL, out}}$",
        "$\lambda_{\mathrm{RL, out}}$",
    ]

    for i in range(weights_in_np.shape[1]):
        axs[i].scatter(
            weights_in_np[:, i], weights_out_np[:, i], color=MIX_NET_COL, s=0.6
        )

        # 45 degrees red line for better visualization:
        axs[i].plot([0, 1], [0, 1], color=TUM_ORAN, linewidth=1.5)

        # axs[i].set_title(titles[i])
        axs[i].set_xlabel(x_label_list[i])
        axs[i].set_ylabel(y_label_list[i])
        axs[i].grid(True)

        axs[i].axis("equal")
        axs[i].set_adjustable("box")
        axs[i].set_xlim(left=0, right=1)
        axs[i].set_ylim(bottom=0, top=1)
        axs[i].xaxis.set_ticks([0.0, 0.5, 1.0])
        axs[i].yaxis.set_ticks([0.0, 0.5, 1.0])

    plt.savefig(os.path.join(SAVE_DIR, "weights_dist.pdf"))
    plt.show()


def replace_centerline(mixnet_handler=None):
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(PAGEWIDTH / 2, PAGEWIDTH / 4)
    )

    weights_in_np, weights_out_np, errors_np = create_synthetic_weights(
        mixnet_handler=mixnet_handler, replace_cl=True
    )
    print("REPLACE CENTERLINE")
    print("Mean L2-Norm is: {:.03f} m".format(np.mean(np.mean(errors_np, axis=0))))

    x_label_list = ["$\lambda_{\mathrm{LB, in}}$", "$\lambda_{\mathrm{RB, in}}$"]
    y_label_list = ["$\lambda_{\mathrm{LB, out}}$", "$\lambda_{\mathrm{RB, out}}$"]

    # left:
    ax_left.scatter(
        weights_in_np[:, 0],
        weights_out_np[:, 0],
        color=MIX_NET_COL,
        s=0.8,
    )
    ax_left.plot([0, 1], [0, 1], color=TUM_ORAN, linewidth=1.5)
    # ax_left.set_title("Left Boundary")

    # right:
    ax_right.scatter(
        weights_in_np[:, 1],
        weights_out_np[:, 1],
        color=MIX_NET_COL,
        s=0.8,
    )
    ax_right.plot([0, 1], [0, 1], color=TUM_ORAN, linewidth=1.5)
    # ax_right.set_title("Right Boundary")

    for j, ax in enumerate([ax_left, ax_right]):
        ax.set_xlabel(x_label_list[j])
        ax.set_ylabel(y_label_list[j])
        ax.grid(True)
        ax.axis("equal")
        ax.set_adjustable("box")
        ax.set_xlim(left=0, right=1)
        ax.set_ylim(bottom=0, top=1)
        ax.xaxis.set_ticks([0.0, 0.5, 1.0])
        ax.yaxis.set_ticks([0.0, 0.5, 1.0])

    plt.savefig(os.path.join(SAVE_DIR, "replace_center.pdf"))
    plt.show()


def get_weight_error():
    weights_in, weights_out, _ = create_synthetic_weights(as_torch=True)
    left, right, center, race = read_map(MAP_PATH)

    # calculating the errors on the horizon:
    errors = [[], [], [], [], []]

    for i in range(weights_in.shape[0]):
        gt = (
            left * weights_in[i][0]
            + right * weights_in[i][1]
            + center * weights_in[i][2]
            + race * weights_in[i][3]
        )

        pred = (
            left * weights_out[i][0]
            + right * weights_out[i][1]
            + center * weights_out[i][2]
            + race * weights_out[i][3]
        )

        for j in range(5):
            p_gt = gt[810 + j * 24 + 23, :]
            p_pred = pred[810 + j * 24 + 23, :]
            errors[j].append(np.linalg.norm(p_gt - p_pred))

    print("Overall RMSE: {}".format(np.mean(errors)))


def get_mixnet_handler(model):
    # Add mixnet handler
    from mix_net.mix_net.src.mix_net_handler import MixNetHandler
    from mix_net.mix_net.utils.setup_helpers import create_path_dict, get_params

    bg_params = {
        "MODEL_PARAMS": {"dist": 20, "view": 400},
        "track_path": os.path.join(repo_path, MAP_PATH),
    }
    path_dict = create_path_dict()
    hdl_params = get_params(path_dict)
    hdl_params["MIX_NET_PARAMS"]["map_file_path"] = bg_params["track_path"]
    mixnet_handler = MixNetHandler(params=hdl_params, net=model, bg=BoundaryGenerator)
    return mixnet_handler


if __name__ == "__main__":
    # exemplary overuse of raceline
    model = get_loaded_model()
    mixnet_handler = get_mixnet_handler(model)

    input_generator = InputProvider(MAP_PATH)
    last_index = 810
    v = 75.0

    # analyze weights
    plot_weight_distribution(mixnet_handler=mixnet_handler)

    # replace centerline by left and right
    replace_centerline(mixnet_handler=mixnet_handler)

    get_weight_error()

    plt.rcParams.update({"font.size": LABELFONTSIZE})
    create_example(input_generator=input_generator, model=model)
