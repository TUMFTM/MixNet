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

plt.rcParams.update({"font.size": 16})
plt.rcParams.update({"font.family": "Times New Roman"})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"figure.autolayout": True})

TUM_BLUE = (0 / 255, 101 / 255, 189 / 255)
MIX_NET_COL = TUM_BLUE
TUM_ORAN = (227 / 255, 114 / 255, 34 / 255)
INDY_NET_COL = TUM_ORAN
HIST_COL = "black"
HIST_LS = "solid"
GT_COL = "black"
GT_LS = "dashed"
BOUND_COL = (204 / 255, 204 / 255, 204 / 255)
RL_COL = "gray"

from mix_net.mix_net.src.mix_net import MixNet
from mix_net.mix_net.utils.geometry import angle_between, transform_trajectory
from mix_net.mix_net.src.boundary_generator import BoundaryGenerator


MAP_PATH = "mix_net/mix_net/data/map/traj_ltpl_cl_IMS_GPS.csv"
# loading the model:
NET_PARAMS_PATH = "mix_net/mix_net/data/inference_model/mix_net/net_params.json"
MODEL_WEIGHTS_PATH = "mix_net/mix_net/data/inference_model/mix_net/model.pth"

SAVE_DIR = "data/evaluation_data/input_output"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)


def create_synthetic_weights(as_torch=False):
    weights_in = np.zeros((400, 4))

    for i in range(4):

        other_col_idxs = list(range(0, i)) + list(range(i + 1, 4))

        main_col_weights = np.random.uniform(0.0, 1.0, size=(100, 1))
        other_col_weights = np.random.uniform(0.0, 1.0, size=(100, 3))

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
    for weight_in in weights_in:
        weights_out.append(
            get_model_output(model, input_generator, weight_in, last_index, v)
        )

    if as_torch:
        return weights_in, weights_out

    weights_in_np = np.array(weights_in)
    weights_out_np = np.array(weights_out)

    return weights_in_np, weights_out_np


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
        hist, left_bound, right_bound = self.get_non_transformed_input(
            weights, last_index, v
        )

        # transforming the trajectory:
        translation = copy.deepcopy(hist[-1, :])

        # getting the rotation:
        rotation = angle_between(left_bound[1, :] - left_bound[0, :], [1, 0])

        # Transforming the history and the boundaries:
        hist = transform_trajectory(hist, translation, rotation)
        left_bound = transform_trajectory(left_bound, translation, rotation)
        right_bound = transform_trajectory(right_bound, translation, rotation)

        return hist, left_bound, right_bound

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

        rel_dists = np.arange(30) * 0.1 * v
        rel_dists = rel_dists[::-1]

        sample_dists -= rel_dists

        hist = np.zeros((30, 2))
        hist[:, 0] = np.interp(sample_dists, dists, line[:, 0])
        hist[:, 1] = np.interp(sample_dists, dists, line[:, 1])

        # Get Boundaries
        left_bound, right_bound = self._bg.get_boundaries_single(hist[-1, :])

        return hist, left_bound, right_bound


def get_model_output(model, input_generator, weights, last_index, v):

    hist, left_bound, right_bound = input_generator.get_transformed_input(
        weights, last_index, v
    )

    # creating tensors:
    hist_t = torch.unsqueeze(torch.from_numpy(hist).float().to(model.device), 0)
    left_bound_t = torch.unsqueeze(
        torch.from_numpy(left_bound).float().to(model.device), 0
    )
    right_bound_t = torch.unsqueeze(
        torch.from_numpy(right_bound).float().to(model.device), 0
    )

    # feeding through the model:
    weights_out, _, _ = model(hist_t, left_bound_t, right_bound_t)

    # numpifying and reshaping:
    weights_out = np.squeeze(weights_out.to("cpu").detach().numpy())

    return weights_out


def plot_lines(ax, left, right, hist, pred_gt, pred, raceline, last_index, linewidth):
    """Helper function for plotting all these lines to a given axis."""

    ax.plot(left[760:930, 0], left[760:930, 1], color=BOUND_COL, linewidth=linewidth)
    ax.plot(right[760:930, 0], right[760:930, 1], color=BOUND_COL, linewidth=linewidth)

    ax.plot(
        hist[10:, 0],
        hist[10:, 1],
        color=HIST_COL,
        linestyle="dashed",
        linewidth=linewidth,
    )

    ax.plot(
        pred_gt[last_index : last_index + 120, 0],
        pred_gt[last_index : last_index + 120, 1],
        color=GT_COL,
        linestyle=GT_LS,
        linewidth=linewidth,
    )

    ax.plot(
        pred[last_index : last_index + 120, 0],
        pred[last_index : last_index + 120, 1],
        color=MIX_NET_COL,
        linewidth=linewidth,
    )

    ax.plot(
        raceline[760:930, 0],
        raceline[760:930, 1],
        color=RL_COL,
        linestyle="dotted",
        linewidth=linewidth,
    )


def create_example(input_generator, model, last_index=810, v=75.0):
    left, right, center, race = read_map(MAP_PATH)
    dummy_weights = np.array([0.4, 0.1, 0.3, 0.2])
    hist_gt, left_bound, right_bound = input_generator.get_non_transformed_input(
        dummy_weights, last_index, v
    )

    pred_gt = (
        left * dummy_weights[0]
        + right * dummy_weights[1]
        + center * dummy_weights[2]
        + race * dummy_weights[3]
    )

    # prediction:
    weights_out = get_model_output(model, input_generator, dummy_weights, last_index, v)

    hist_pred, _, _ = input_generator.get_non_transformed_input(
        weights_out, last_index, v
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
    fig = plt.figure(figsize=(9, 9))
    ax = fig.gca()
    ax.axis("equal")

    plot_lines(ax, left, right, hist_gt, pred_gt, pred, race, last_index, linewidth=5)

    # turning off axis ticks:
    ax.set_xlim(100, 450)
    ax.set_ylim(-1200, -850)
    ax.xaxis.set_ticks([100, 250, 400])
    ax.yaxis.set_ticks([-1200, -1100, -1000, -900])

    # inset axes 1:
    insax_1 = ax.inset_axes([0.01, 0.3, 0.7, 0.3])
    insax_1.axis("equal")

    plot_lines(
        insax_1, left, right, hist_gt, pred_gt, pred, race, last_index, linewidth=5
    )

    insax_1.set_xlim(130, 190)
    insax_1.set_ylim(-890, -865)
    insax_1.set_xticklabels([])
    insax_1.set_yticklabels([])

    ax.indicate_inset_zoom(insax_1, edgecolor="black")

    # inset axes 2:
    insax_2 = ax.inset_axes([0.72, 0.72, 0.27, 0.27])
    insax_2.axis("equal")

    plot_lines(
        insax_2, left, right, hist_gt, pred_gt, pred, race, last_index, linewidth=5
    )

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
            linewidth=5,
            label="Ground Truth",
        ),
        Line2D(
            [0],
            [0],
            color=MIX_NET_COL,
            linestyle="solid",
            linewidth=5,
            label="Prediction",
        ),
        Line2D(
            [0],
            [0],
            color=RL_COL,
            linestyle="dotted",
            linewidth=5,
            label="Raceline",
        ),
        Line2D(
            [0], [0], color=HIST_COL, linestyle="solid", linewidth=5, label="History"
        ),
    ]

    ax.legend(handles=legend_elements, prop={"size": 24}, loc="lower left")
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


def plot_weight_distribution():

    weights_in_np, weights_out_np = create_synthetic_weights()

    fig, (ax_left, ax_right, ax_center, ax_race) = plt.subplots(1, 4, figsize=(12, 3))
    axs = [ax_left, ax_right, ax_center, ax_race]
    titles = ["Left Boundary", "Right Boundary", "Centerline", "Raceline"]

    for i in range(weights_in_np.shape[1]):
        axs[i].scatter(
            weights_in_np[:, i], weights_out_np[:, i], color=MIX_NET_COL, s=0.8
        )

        # 45 degrees red line for better visualization:
        axs[i].plot([0, 1], [0, 1], color=TUM_ORAN, linewidth=2)

        axs[i].set_title(titles[i])
        axs[i].set_xlabel("$\lambda_{\mathrm{c, in}}$")
        axs[i].set_ylabel("$\lambda_{\mathrm{c, out}}$")
        axs[i].grid(True)

        axs[i].axis("equal")
        axs[i].set_adjustable("box")
        axs[i].set_xlim(left=0, right=1)
        axs[i].set_ylim(bottom=0, top=1)
        axs[i].xaxis.set_ticks([0.0, 0.5, 1.0])
        axs[i].yaxis.set_ticks([0.0, 0.5, 1.0])

    plt.savefig(os.path.join(SAVE_DIR, "weights_dist.pdf"))
    plt.show()


def replace_centerline():

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(6, 3))

    weights_in_np, weights_out_np = create_synthetic_weights()

    # left:
    ax_left.scatter(
        weights_in_np[:, 0] + 0.5 * weights_in_np[:, 2],
        weights_out_np[:, 0] + 0.5 * weights_out_np[:, 2],
        color=MIX_NET_COL,
        s=0.8,
    )
    ax_left.plot([0, 1], [0, 1], color=TUM_ORAN, linewidth=2)
    ax_left.set_title("Left Boundary")

    # right:
    ax_right.scatter(
        weights_in_np[:, 1] + 0.5 * weights_in_np[:, 2],
        weights_out_np[:, 1] + 0.5 * weights_out_np[:, 2],
        color=MIX_NET_COL,
        s=0.8,
    )
    ax_right.plot([0, 1], [0, 1], color=TUM_ORAN, linewidth=2)
    ax_right.set_title("Right Boundary")

    for ax in [ax_left, ax_right]:
        ax.set_xlabel("$\lambda_{\mathrm{c, in}}$")
        ax.set_ylabel("$\lambda_{\mathrm{c, out}}$")
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
    weights_in, weights_out = create_synthetic_weights(as_torch=True)
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

    print("Overall MAE: {}".format(np.mean(errors)))


if __name__ == "__main__":
    # exemplary overuse of raceline
    model = get_loaded_model()
    input_generator = InputProvider(MAP_PATH)
    last_index = 810
    v = 75.0

    # analyze weights
    plot_weight_distribution()

    # replace centerline by left and right
    replace_centerline()

    get_weight_error()

    plt.rcParams.update({"font.size": 30})
    create_example(input_generator=input_generator, model=model)
