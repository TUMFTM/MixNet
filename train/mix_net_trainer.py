"""Trainer class for MixNet."""
import sys
import os
import torch
import numpy as np
import json
import time
import datetime
import shutil
import argparse
import pkbar
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 16})
plt.rcParams.update({"text.usetex": True})
plt.rcParams.update({"figure.autolayout": True})
plt.rcParams.update({"font.family": "Times New Roman"})

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

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split

torch.autograd.set_detect_anomaly(True)

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

from mix_net.mix_net.src.mix_net import MixNet
from train.mix_net_dataset import MixNetDataset
from mix_net.mix_net.utils.line_helper import LineHelper
from mix_net.mix_net.utils.map_utils import get_track_paths
from train.data_set_helper import load_mix_net_data
from train.neural_network import weighted_MSE


class MixNetTrainer:
    """Class for training a MixNet Neural Network."""

    def __init__(self, params: dict, net: MixNet):
        """Initialize a MixNetTrainer object."""
        self._params = params
        self._net = net.float()
        self._device = self._net.device

        # loss function:
        self._set_lossfunction()

        # optimizer
        self._optimizer = torch.optim.Adam(
            self._net.parameters(),
            lr=self._params["training"]["lr"],
            weight_decay=self._params["training"]["weight_decay"],
        )

        # learning rate scheduling:
        self._scheduler = ExponentialLR(
            optimizer=self._optimizer, gamma=self._params["training"]["lr_decay_rate"]
        )

        # reading the unified map file:
        (_, centerline, bound_right, bound_left, raceline) = get_track_paths(
            track_path=self._params["data"]["map_file_path"], bool_raceline=True
        )

        # LineHelpers:
        self._left_bound = LineHelper(bound_left)
        self._right_bound = LineHelper(bound_right)
        self._centerline = LineHelper(centerline)
        self._raceline = LineHelper(raceline)

        # for logging the losses:
        self._train_loss = 0.0
        self._train_counter = 0
        self._train_logger_step = 0

        self._val_loss = 0.0
        self._val_counter = 0
        self._val_logger_step = 0
        self._best_val_loss = np.inf

        if params["train"]:
            # These are the date and time dirs which are used everywhere:
            date_formatted = datetime.date.today().strftime("%Y_%m_%d")
            time_formatted = datetime.datetime.now().strftime("%H_%M_%S")

            self._prepare_model_dir(date_formatted, time_formatted)

            self._prepare_dataloader_saving(date_formatted, time_formatted)

            self._prepare_summary_writer(date_formatted, time_formatted)

        # creating a matrix which will be used during the velocity profile calculation:
        self._get_time_matrix()

    def load_data(self):
        """load the training data from the specified file .

        Data is split into train, val, test. If params["data"]["from_saved_dataloader"] is true, the provided
        data loaders are going to be loaded.
        """
        print("-" * 10 + " LOADING DATA " + "-" * 10)

        self._dataloaders = {}

        if self._params["data"]["from_saved_dataloader"]:
            dl_dir = self._params["data"]["dataloader_path"]
            assert os.path.exists(
                dl_dir
            ), "The directory, {}, to load the dataloaders from does not exist.".format(
                dl_dir
            )

            for phase in ["train", "val", "test"]:
                path = os.path.join(dl_dir, (phase + ".pth"))
                if os.path.exists(path):
                    self._dataloaders[phase] = torch.load(path)
                    print("Loaded {} dataloader from {}".format(phase, path))
                else:
                    self._dataloaders[phase] = None

        else:
            data = load_mix_net_data(
                self._params["data"]["path"],
                use_every_nth=self._params["data"]["use_every_nth"],
            )

            train_data, val_data, test_data = self._split_data(
                data, random_state=self._params["data"]["random_state"]
            )

            # creating the dataloaders:
            cut_prob = self._params["data"]["cut_hist_probability"]
            min_len = self._params["data"]["min_hist_len"]

            self._dataloaders["train"] = DataLoader(
                MixNetDataset(train_data, cut_prob, min_len, random_seed=0),
                batch_size=self._params["training"]["batch_size"],
                shuffle=self._params["data"]["shuffle"],
            )

            self._dataloaders["val"] = DataLoader(
                MixNetDataset(val_data, cut_prob, min_len, random_seed=1),
                batch_size=self._params["training"]["batch_size"],
                shuffle=self._params["data"]["shuffle"],
            )

            self._dataloaders["test"] = DataLoader(
                MixNetDataset(test_data, cut_prob, min_len, random_seed=2),
                batch_size=self._params["training"]["batch_size"],
                shuffle=self._params["data"]["shuffle"],
            )

        # saving the created DataLoaders if necessary:
        if self._params["logging"]["save_dataloaders"] and self._params["train"]:
            for phase in ["train", "val", "test"]:
                if self._dl_savepath[phase] is not None:
                    torch.save(self._dataloaders[phase], self._dl_savepath[phase])
                    print(
                        "saved {} dataloader to {}".format(
                            phase, self._dl_savepath[phase]
                        )
                    )

        print("-" * 10 + " LOADING DATA END " + "-" * 10)

    def _split_data(self, data, random_state=None):
        """Splits the given data into train, validation and test sets.

        args:
            data: (dict), all of the data with the keys:
                "hist": (list of 2D lists) the history trajectories.
                "fut": (list of 2D lists) the groundtruth future trajectories.
                "fut_inds": (list of lists) the indices of the nearest centerline points
                    corresponding to the ground truth prediction.
                "left_bd": (list of 2D lists) left track boundary snippet.
                "right_bd": (list of 2D lists) right track boundary snippet.

        returns:
            train_data: (dict) the splitted training set with the same keys as data
            val_data: (dict) the splitted training set with the same keys as data
            test_data: (dict) the splitted training set with the same keys as data
        """
        train_size = self._params["data"]["train_size"]
        val_size = self._params["data"]["val_size"]
        test_size = self._params["data"]["test_size"]

        # train - (val + test) split:
        (
            train_fut_inds,
            val_fut_inds,
            train_hist,
            val_hist,
            train_fut,
            val_fut,
            train_left_bd,
            val_left_bd,
            train_right_bd,
            val_right_bd,
        ) = train_test_split(
            data["fut_inds"],
            data["hist"],
            data["fut"],
            data["left_bd"],
            data["right_bd"],
            train_size=train_size,
            random_state=random_state,
        )

        # val - test split:
        (
            val_hist,
            test_hist,
            val_fut,
            test_fut,
            val_fut_inds,
            test_fut_inds,
            val_left_bd,
            test_left_bd,
            val_right_bd,
            test_right_bd,
        ) = train_test_split(
            val_hist,
            val_fut,
            val_fut_inds,
            val_left_bd,
            val_right_bd,
            train_size=(val_size / (val_size + test_size)),
            random_state=random_state,
        )

        # constructing the dicts:
        train_data = {
            "hist": train_hist,
            "fut": train_fut,
            "fut_inds": train_fut_inds,
            "left_bd": train_left_bd,
            "right_bd": train_right_bd,
        }

        val_data = {
            "hist": val_hist,
            "fut": val_fut,
            "fut_inds": val_fut_inds,
            "left_bd": val_left_bd,
            "right_bd": val_right_bd,
        }

        test_data = {
            "hist": test_hist,
            "fut": test_fut,
            "fut_inds": test_fut_inds,
            "left_bd": test_left_bd,
            "right_bd": test_right_bd,
        }

        return train_data, val_data, test_data

    def train(self):
        """Loads the training data and carries out training and validation
        loops for each epoch.
        """
        print("-" * 10 + " TRAINING " + "-" * 10)

        for epoch in range(self._params["training"]["num_epochs"]):
            # progress bar:
            self._pbar = pkbar.Kbar(
                target=len(self._dataloaders["train"]),
                epoch=epoch,
                num_epochs=self._params["training"]["num_epochs"],
            )

            self._train(epoch)
            self._validate(epoch)

            self._scheduler.step()

        print("-" * 10 + " TRAINING END " + "-" * 10)

    def _train(self, epoch):
        """Carries out an epoch of training by iterating through the
        data of the train DataLoader.

        args:
            epoch: (int), the number of the current epoch.

        returns:
            train_path_loss: (float), The average training loss of the path mixture.
            train_vel_loss: (float), The average training loss of the velocity profile.
        """

        t0 = time.time()

        self._net.train()

        cum_path_loss = 0.0
        cum_vel_loss = 0.0

        for i, (hist, fut, fut_inds, left_bound, right_bound) in enumerate(
            self._dataloaders["train"]
        ):
            out = self._net(hist, left_bound, right_bound)

            path_loss, vel_loss = self._calc_loss(out, fut, fut_inds)
            total_loss = path_loss + vel_loss

            # optimization step:
            self._optimizer.zero_grad()
            total_loss.backward()
            self._optimizer.step()

            # logging:
            self._log_loss(total_loss, "train")
            cum_path_loss += path_loss
            cum_vel_loss += vel_loss
            epoch_path_loss = cum_path_loss / (i + 1)
            epoch_vel_loss = cum_vel_loss / (i + 1)

            # progress bar update:
            self._pbar.update(
                i, [("train pos MSE", path_loss), ("train vel MSE", vel_loss)]
            )

        self._write_summary("Computation Time/Train", (time.time() - t0), epoch)

        return epoch_path_loss, epoch_vel_loss

    def _validate(self, epoch):
        """Carries out whole pass through the validation data and logs
        the resulted losses. If the loss is better then the previous best
        validation loss, it saves the model weights.

        args:
            epoch: (int), the number of the current epoch.

        returns:
            val_path_loss: (float), The average validation loss of the path mixture.
            val_vel_loss: (float), The average validation loss of the velocity profile.
        """

        t0 = time.time()
        self._net.eval()

        cum_path_loss = 0.0
        cum_vel_loss = 0.0

        with torch.no_grad():
            for i, (hist, fut, fut_inds, left_bound, right_bound) in enumerate(
                self._dataloaders["val"]
            ):
                out = self._net(hist, left_bound, right_bound)

                path_loss, vel_loss = self._calc_loss(out, fut, fut_inds)

                total_loss = path_loss + vel_loss
                # logging:
                self._log_loss(total_loss, "val")
                cum_path_loss += path_loss
                cum_vel_loss += vel_loss
                epoch_path_loss = cum_path_loss / (i + 1)
                epoch_vel_loss = cum_vel_loss / (i + 1)

            # progress bar update:
            self._pbar.add(
                2, [("val pos MSE", epoch_path_loss), ("val vel MSE", epoch_vel_loss)]
            )

            # saving the model if necessary:
            epoch_total_loss = epoch_path_loss + epoch_vel_loss
            if self._model_save_path is not None:
                if epoch_total_loss < self._best_val_loss:
                    self._best_val_loss = epoch_total_loss
                    torch.save(self._net.state_dict(), self._model_save_path)
                    print("New best model was saved.")

        self._write_summary("Computation Time/Val", (time.time() - t0), epoch)

        return epoch_path_loss, epoch_vel_loss

    def test(self):
        """Tests the network on the test dataset.

        returns:
            The test time loss.
        """

        print("-" * 10 + " TESTING " + "-" * 10)
        t0 = time.time()

        self._net.eval()

        test_loss = 0.0
        test_len = len(self._dataloaders["test"])

        with torch.no_grad():
            for (hist, fut, fut_inds, left_bound, right_bound) in self._dataloaders[
                "test"
            ]:
                out = self._net(hist, left_bound, right_bound)

                path_loss, vel_loss = self._calc_loss(out, fut, fut_inds)
                test_loss += (path_loss + vel_loss) / test_len

        print("Tested the network in {} s".format(time.time() - t0))
        print("Test RMSE: {} m".format(test_loss))
        print("-" * 10 + " TESTING END" + "-" * 10)

        return test_loss

    def _calc_loss(self, out, fut, fut_inds):
        """Calculates the loss for the network.
        It creates the mixture path based on the mixing ratios in out
        and then compares it to the groundtruth future prediction fut.

        args:
            out: [tuple of tensors]: the outputs of the MixNet:
                0: mix_out: [tensor with shape=(batch_size, mix_size)] the mixing ratios.
                1: vel_out: [tensor with shape=(batch_size, 1)] the initial velocities
                2: acc_out: [tensor with shape=(batch_size, num_of_acc_sections)] the accelerations
            fut: [np.array with shape=(batch_size, pred_len, 2)] the ground truth prediction
            fut_inds: [2D list with shape=(batch_size, pred_len)] The indices of the nearest points
                on the centerline corresponding to the ground truth predictions.

        returns:
            path_loss: [float tensor], the loss that is coming from the path mixture inaccuracy
            vel_loss: [float tensor], the loss that is coming from the velocity profile inaccuracy
        """

        (mix_out, vel_out, acc_out) = out

        fut = fut.to(self._device)

        # calculating the path loss:
        fut_inds = fut_inds.numpy().tolist()

        fut_out = torch.zeros_like(fut, dtype=torch.float32).to(self._device)

        for i, inds in enumerate(fut_inds):
            left = torch.from_numpy(self._left_bound.line[inds, :]).to(self._device)
            right = torch.from_numpy(self._right_bound.line[inds, :]).to(self._device)
            center = torch.from_numpy(self._centerline.line[inds, :]).to(self._device)
            race = torch.from_numpy(self._raceline.line[inds, :]).to(self._device)

            fut_out[i, :, :] = (
                mix_out[i, 0] * left
                + mix_out[i, 1] * right
                + mix_out[i, 2] * center
                + mix_out[i, 3] * race
            )

        if self._params["training"]["loss_fn"] == "WMSE":
            path_loss = self._loss_fn(fut, fut_out, self._MSE_weights)
        else:
            path_loss = self._loss_fn(fut, fut_out)

        # claculating the velocity loss:
        freq = self._params["data"]["frequency"]
        vel_profile = torch.norm((fut[:, 1:, :] - fut[:, :-1, :]), dim=2) * freq

        vel_profile_out = torch.ones_like(vel_profile, dtype=torch.float32).to(
            self._device
        )
        vel_profile_out = vel_profile_out * vel_out

        # multiplying the time matrix with the accelerations to reach the relative velocity profile:
        vel_profile_out = vel_profile_out + (self._time_profile_matrix @ acc_out.T).T

        if self._params["training"]["loss_fn"] == "WMSE":
            vel_loss = self._loss_fn(
                vel_profile, vel_profile_out, self._MSE_weights[:-1]
            )
        else:
            vel_loss = self._loss_fn(vel_profile, vel_profile_out)

        # we are scaling the velocity difference with freq^2. This is reasonable,
        # since the velocity differences cause vel_diff * dt = vel_diff / freq
        # position error, and since we are using squared error, the scaling should be
        # freq^2:
        vel_loss /= freq**2

        return path_loss, vel_loss

    def _set_lossfunction(self):
        """Sets up the loss function according to the params."""

        assert self._params["training"]["loss_fn"] in [
            "MSE",
            "WMSE",
        ], 'The given loss function type is invalid. Must be either "MSE" or "WMSE"'
        if self._params["training"]["loss_fn"] == "MSE":
            self._loss_fn = torch.nn.MSELoss()
        else:
            self._loss_fn = weighted_MSE

            # setting up weights for weighted MSE:
            self._MSE_weights = torch.ones(
                (self._params["training"]["pred_len"]),
                dtype=torch.float32,
                device=self._device,
            )
            max_weight = self._params["training"]["max_weight"]
            horizon = self._params["training"]["weighting_horizon"]
            self._MSE_weights[0 : (horizon + 1)] = torch.linspace(
                max_weight, 1.0, (horizon + 1)
            )

    def _prepare_model_dir(self, date_formatted, time_formatted):
        """Creates the model saving path and makes sure that the
        directories in the resulting path exist.

        args:
            date_formatted: (str), the date with formatting "%Y_%m_%d"
            time_formatted: (str), the time with formatting "%H_%M_%S"
        """

        self._model_save_path = None

        if self._params["training"]["model_save_path"] != "":
            self._model_save_path = os.path.join(
                self._params["training"]["model_save_path"],
                date_formatted,
                time_formatted,
                "model.pth",
            )

            if not os.path.exists(os.path.dirname(self._model_save_path)):
                os.makedirs(os.path.dirname(self._model_save_path))

            # dumping the trainer and det params here:
            file = os.path.join(
                os.path.dirname(self._model_save_path), "trainer_params.json"
            )
            with open(file, "w") as fp:
                json.dump(self._params, fp, indent=4)

            file = os.path.join(
                os.path.dirname(self._model_save_path), "net_params.json"
            )
            with open(file, "w") as fp:
                json.dump(self._net.get_params(), fp, indent=4)

    def _prepare_dataloader_saving(self, date_formatted, time_formatted):
        """Prepares the directory and the paths where the dataloaders are going
        to be saved, if needed. If params["logging"]["dataloader_path"] is an
        empty string, the dataloaders are not going to be saved.

        The paths are stored in the self._dl_savepaths dict with the keys
        "train", "val" and "test".
        """

        self._dl_savepath = {"train": None, "val": None, "test": None}

        base_dir = self._params["logging"]["dataloader_path"]
        if not self._params["logging"]["save_dataloaders"]:
            return

        outdir = os.path.join(base_dir, date_formatted, time_formatted)
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        for phase in ["train", "val", "test"]:
            if self._params["logging"]["save_" + phase + "_dataloader"]:
                self._dl_savepath[phase] = os.path.join(outdir, (phase + ".pth"))

    def _prepare_summary_writer(self, date_formatted, time_formatted):
        """Creates the summary writer and makes sure that the
        directories in the path exist.

        args:
            date_formatted: (str), the date with formatting "%Y_%m_%d"
            time_formatted: (str), the time with formatting "%H_%M_%S"
        """

        self._writer = None

        if self._params["logging"]["log_path"] != "":
            summary_path = os.path.join(
                self._params["logging"]["log_path"], date_formatted, time_formatted
            )

            # deleting the contents of the dir (if any) and recreating it:
            if os.path.exists(summary_path):
                shutil.rmtree(summary_path)

            os.makedirs(summary_path)

            self._writer = SummaryWriter(summary_path)

    def _get_time_matrix(self):
        """Creates a matrix which make the velocity profile calculation easier.
        If the matrix is multiplied with the section-wise acceleration that is one of the
        outputs of the network, it will create a relative velocity profile that can be
        added to the initial velocity profile.

        The matrix has the size (N, M) where N is the number of timesteps and M is the
        number of sections (components) of the vel profile.
        """

        self._time_profile_matrix = torch.zeros(
            (self._params["training"]["pred_len"] - 1, 5), dtype=torch.float32
        ).to(self._device)

        for i in range(5):
            self._time_profile_matrix[(i * 10) : ((i + 1) * 10), i] = torch.linspace(
                0.1, 1.0, 10
            )

            self._time_profile_matrix[((i + 1) * 10) :, i] = 1.0

    def _log_loss(self, loss, phase):
        """Logs the loss that was given, according to the phase of training.

        args:
            loss: (scalar), the new loss value to be included in the loss history.
            phase: (str), either "train" or "val"
        """

        phases = ["train", "val"]
        assert (
            phase in phases
        ), "Invalid phase was given to logger. Expected one of the followings: {}".format(
            phases
        )

        if phase == "train":
            train_log_interval = self._params["logging"]["train_loss_log_interval"]
            self._train_loss += loss / train_log_interval
            self._train_counter += 1

            if self._train_counter == train_log_interval:
                train_epoch_len = len(self._dataloaders["train"])
                step = self._train_logger_step / (train_epoch_len / train_log_interval)
                self._write_summary("Loss/Train", self._train_loss, step)

                self._train_loss = 0.0
                self._train_counter = 0
                self._train_logger_step += 1

        elif phase == "val":
            val_len = len(self._dataloaders["val"])
            self._val_loss += loss / val_len
            self._val_counter += 1

            if self._val_counter == val_len:
                self._write_summary("Loss/Val", self._val_loss, self._val_logger_step)

                self._val_loss = 0.0
                self._val_counter = 0
                self._val_logger_step += 1

    def _write_summary(self, key: str, val, step):
        """writes a scalar summary with the tensorboard summary writer
        if it is available.

        args:
            key: (str), the key of the logged scalar value
            val: (scalar), the value to add to the summary
            step: (scalar), the index of the log
        """

        if self._writer is not None:
            self._writer.add_scalar(key, val, step)

    def load_best_saved_model(self):
        """Loads the beste saved model found under self._model_save_path if any."""

        if not self._net.load_model_weights(self._model_save_path):
            print(
                "Can not test model, because could not load weights from {}".format(
                    self._model_save_path
                )
            )

    def visualize_test(self):
        """Iterates through the test data and visualizes the data samples."""

        print("-" * 10 + " VISUALIZATION " + "-" * 10)

        self._net.eval()

        with torch.no_grad():
            for (hist, fut, fut_inds, left_bound, right_bound) in self._dataloaders[
                "test"
            ]:
                mix_out, vel_out, acc_out = self._net(hist, left_bound, right_bound)

                path_loss, vel_loss = self._calc_loss(
                    (mix_out, vel_out, acc_out), fut, fut_inds
                )

                mix_out_np = mix_out.to("cpu").numpy()
                fut_np = fut.numpy()
                left_bound_np = left_bound.numpy()
                right_bound_np = right_bound.numpy()
                fut_inds_list = fut_inds.numpy().tolist()

                vel_out_np = vel_out.to("cpu").numpy()
                acc_out_np = acc_out.to("cpu").numpy()

                for i in range(mix_out.shape[0]):
                    fig = plt.figure(figsize=(15, 15))
                    if self._params["plt_vel"]:
                        ax = fig.add_subplot(221)
                        axvel = fig.add_subplot(222)
                        ax2 = fig.add_subplot(212)
                    else:
                        ax = fig.add_subplot(211)
                        ax2 = fig.add_subplot(212)
                    ax.axis("equal")
                    ax2.axis("equal")

                    # get MixNet superposition prediction
                    inds = fut_inds_list[i]
                    left = self._left_bound.line[inds, :]
                    right = self._right_bound.line[inds, :]
                    center = self._centerline.line[inds, :]
                    race = self._raceline.line[inds, :]

                    pred = (
                        mix_out_np[i, 0] * left
                        + mix_out_np[i, 1] * right
                        + mix_out_np[i, 2] * center
                        + mix_out_np[i, 3] * race
                    )

                    # plot net input
                    hist_loc = hist[i, :, :]
                    left_bound_loc = left_bound_np[i, :, :]
                    right_bound_loc = right_bound_np[i, :, :]

                    # plotting the path:
                    ax.set_title("Net Input")
                    ax.plot(
                        hist_loc[:, 0],
                        hist_loc[:, 1],
                        color=HIST_COL,
                        label="History",
                        linewidth=1.5,
                    )
                    ax.plot(
                        left_bound_loc[:, 0],
                        left_bound_loc[:, 1],
                        "x",
                        color=BOUND_COL,
                        label="Sampled Boundary",
                        linewidth=1.5,
                        linestyle="solid",
                    )
                    ax.plot(
                        right_bound_loc[:, 0],
                        right_bound_loc[:, 1],
                        "x",
                        color=BOUND_COL,
                        linewidth=1.5,
                        linestyle="solid",
                    )

                    # plot output
                    str1 = "$\mathrm{MAE}_{\mathrm{path}}$ = " + "{:.02f} m".format(
                        path_loss
                    )
                    str2 = "$\mathrm{MAE}_{\mathrm{vel}}$ = " + "{:.02f} m".format(
                        vel_loss
                    )

                    rho1 = r"$\rho_{\mathrm{left}}$ = " + "{:.02f}".format(
                        mix_out_np[i, 0]
                    )
                    rho2 = r"$\rho_{\mathrm{right}}$ = " + "{:.02f}".format(
                        mix_out_np[i, 1]
                    )
                    rho3 = r"$\rho_{\mathrm{center}}$ = " + "{:.02f}".format(
                        mix_out_np[i, 2]
                    )
                    rho4 = r"$\rho_{\mathrm{race}}$ = " + "{:.02f}".format(
                        mix_out_np[i, 3]
                    )
                    ax2.set_title(
                        "Prediction: {}, {};  Weights: {}, {}, {}, {}".format(
                            str1, str2, rho1, rho2, rho3, rho4
                        )
                    )

                    ax2.plot(
                        fut_np[i, :, 0],
                        fut_np[i, :, 1],
                        color=GT_COL,
                        linestyle=GT_LS,
                        label="Ground Truth",
                    )
                    ax2.plot(
                        pred[:, 0],
                        pred[:, 1],
                        color=MIX_NET_COL,
                        label="Prediction",
                    )
                    ax2.plot(
                        left[:, 0],
                        left[:, 1],
                        color=BOUND_COL,
                        label="Base Curves",
                    )
                    ax2.plot(
                        right[:, 0],
                        right[:, 1],
                        color=BOUND_COL,
                    )
                    ax2.plot(
                        center[:, 0],
                        center[:, 1],
                        color=BOUND_COL,
                    )
                    ax2.plot(
                        race[:, 0],
                        race[:, 1],
                        color=BOUND_COL,
                    )

                    if self._params["plt_vel"]:
                        # velocity profile:
                        # claculating the velocity loss:
                        freq = self._params["data"]["frequency"]
                        vel_profile = (
                            np.linalg.norm(
                                (fut_np[i, 1:, :] - fut_np[i, :-1, :]), axis=1
                            )
                            * freq
                        )

                        vel_profile_out = np.ones_like(vel_profile)
                        vel_profile_out = vel_profile_out * vel_out_np[i, 0]

                        # relative velocity profile
                        rel_vel = (
                            self._time_profile_matrix.to("cpu").numpy()
                            @ acc_out_np[i, :].T
                        ).T
                        vel_profile_out = vel_profile_out + rel_vel

                        # plotting the vel profile:
                        axvel.plot(
                            np.arange(vel_profile.size),
                            vel_profile,
                            color=GT_COL,
                            linestyle=GT_LS,
                            label="Ground Truth",
                        )
                        axvel.plot(
                            np.arange(vel_profile_out.size),
                            vel_profile_out,
                            color=MIX_NET_COL,
                            label="Prediction",
                        )
                        axvel.set_title("Velocity Profile")
                        axvel.set_xlabel("$n_{\mathrm{pred}}$")
                        axvel.set_ylabel("$v$ in m/s")
                        axvel.grid(True)
                        axvel.legend()

                    ax.set_xlabel("$x_{\mathrm{loc}}$ in m")
                    ax.set_ylabel("$y_{\mathrm{loc}}$ in m")
                    ax.grid(True)
                    ax.legend()

                    ax2.set_xlabel("$x_{\mathrm{glob}}$ in m")
                    ax2.set_ylabel("$y_{\mathrm{glob}}$ in m")
                    ax2.grid(True)
                    ax2.legend()

                    if self._params["save_figs"]:
                        fig_save_path = "train/figs"
                        if i == 0:
                            if not os.path.exists(fig_save_path):
                                os.makedirs(fig_save_path)
                            else:
                                _ = [
                                    os.remove(os.path.join(fig_save_path, k))
                                    for k in os.listdir(fig_save_path)
                                ]
                        plt.savefig(
                            os.path.join(fig_save_path, "%05d.svg" % i),
                            format="svg",
                            dpi=300,
                        )
                    else:
                        plt.show()

                    plt.close()

                    if i + 1 >= args.max_plots:
                        break


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--max_plots", type=int, default=10)
    parser.add_argument("--plt_vel", action="store_true")
    parser.add_argument("--save_figs", action="store_true")
    args = parser.parse_args()

    # reading the parameter files:
    net_param_file = os.path.join(repo_path, "train/configs/mix_net/net_params.json")

    if args.test:
        trainer_param_file = os.path.join(
            repo_path,
            "train/configs/mix_net/trainer_params_test.json",
        )
    else:
        trainer_param_file = os.path.join(
            repo_path,
            "train/configs/mix_net/trainer_params_train.json",
        )

    with open(net_param_file, "r") as fp:
        net_params = json.load(fp)

    with open(trainer_param_file, "r") as fp:
        trainer_params = json.load(fp)

    # creating the network and loading the weights:
    net = MixNet(net_params)
    if trainer_params["model"]["load_model"]:
        success = net.load_model_weights(trainer_params["model"]["model_load_path"])
        if not success:
            print(
                "Could not load model from file {}".format(
                    trainer_params["model"]["model_load_path"]
                )
            )

    # initializing the trainer:
    trainer_params.update(args.__dict__)
    trainer = MixNetTrainer(trainer_params, net)

    trainer.load_data()

    # main loops:
    if trainer_params["train"]:
        trainer.train()

        if trainer_params["test"]:
            # loading the best model if testing will also be carried out:
            trainer.load_best_saved_model()

    if trainer_params["test"]:
        test_loss = trainer.test()
        print("TEST LOSS: {:.4f}".format(test_loss))

        trainer.visualize_test()
