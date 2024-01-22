"""
This script is the main script for the training of the prediction network.
Arguments:
-- config <path to config file>
-- debug  Set for debug mode (only one step training/validation/evaluation)
"""
# Standard imports
import sys
import os
import json
import argparse
import matplotlib

matplotlib.use("Agg")

# Third party imports
import torch
from torch.nn import MSELoss
import pkbar
import git
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
import optuna

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

# Custom imports
from train.indy_net_dataset import IndyDataset
from mix_net.mix_net.src.indy_net import IndyNet
from train.neural_network import NLL
from mix_net.mix_net.utils.cuda import cudanize
from train.indy_net_evaluation import evaluate
from train.data_set_helper import split_indy_net_data, load_indy_net_data
from train.neural_network import weighted_MSE


def main(common_args, verbose=True, trial=None):
    """Trains the IndyNet."""
    # Create model path
    if not os.path.exists(common_args["save_path"]):
        os.makedirs(common_args["save_path"])
    model_path = os.path.join(
        common_args["save_path"], common_args["model_name"] + ".tar"
    )

    # Initialize network
    net = IndyNet(common_args)

    # Get number of parameters
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    if verbose:
        print("Model initialized with {} parameters".format(pytorch_total_params))

    # Get current git commit hash
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    common_args["commit"] = sha

    # Initialize tensorboard
    writer = SummaryWriter(
        os.path.join(common_args["tb_logs"], common_args["model_name"])
    )

    # Initialize optimizer
    optimizer_rmse = torch.optim.Adam(net.parameters(), lr=common_args["lr_rmse"])
    optimizer_nll = torch.optim.Adam(net.parameters(), lr=common_args["lr_nll"])

    # learning rate scheduling:
    scheduler = ExponentialLR(
        optimizer=optimizer_rmse, gamma=common_args["lr_rmse_decay_rate"]
    )

    if common_args["loss_fn"] == "MSE":
        loss_fn = MSELoss()
    else:  # WMSE
        loss_fn = weighted_MSE

        # weights of weighted MSE loss:
        mse_weights = torch.ones((common_args["out_length"],)).to(net.device)
        mse_weights[:20] = torch.linspace(2.0, 1.0, steps=20)

    # Initialize data loaders
    print("Loading data...")

    sample_data = load_indy_net_data(path=common_args["data_path"])

    train_data, val_data, test_data = split_indy_net_data(
        sample_data,
        train_size=common_args["train_size"],
        val_size=common_args["val_size"],
        test_size=common_args["test_size"],
    )

    # datasets:
    trainings_set = IndyDataset(
        data=train_data,
        cut_probability=common_args["cut_hist_probability"],
        min_len=common_args["hist_min_len"],
        random_seed=0,
    )
    validation_set = IndyDataset(
        data=val_data,
        cut_probability=common_args["cut_hist_probability"],
        min_len=common_args["hist_min_len"],
        random_seed=1,
    )
    test_set = IndyDataset(
        data=test_data,
        cut_probability=common_args["cut_hist_probability"],
        min_len=common_args["hist_min_len"],
        random_seed=2,
    )

    # dataloaders from datasets:
    tr_dataloader = DataLoader(
        trainings_set,
        batch_size=common_args["batch_size"],
        shuffle=True,
        num_workers=common_args["worker"],
        collate_fn=trainings_set.collate_fn,
    )
    val_dataloader = DataLoader(
        validation_set,
        batch_size=common_args["batch_size"],
        shuffle=True,
        num_workers=common_args["worker"],
        collate_fn=validation_set.collate_fn,
    )
    ts_dataloader = DataLoader(
        test_set,
        batch_size=common_args["batch_size"],
        shuffle=True,
        num_workers=common_args["worker"],
        collate_fn=test_set.collate_fn,
    )

    print("Loading data has ended.")

    best_val_loss = np.inf

    for epoch_num in range(common_args["pretrainEpochs"] + common_args["trainEpochs"]):
        if epoch_num == 0:
            if verbose:
                print("Pre-training with MSE loss")
            optimizer = optimizer_rmse
        elif epoch_num == common_args["pretrainEpochs"]:
            if verbose:
                print("Training with NLL loss")
            optimizer = optimizer_nll
            if common_args["save_best"]:
                if verbose:
                    print("Loading best model from pre-training")
                if common_args["use_cuda"]:
                    net.load_model_weights(model_path)
                    net = net.cuda()
                else:
                    net.load_model_weights(model_path)

        # ---------------- Training ----------------
        net.train_flag = True
        net = net.to(net.device)

        # Init progbar
        if verbose:
            kbar = pkbar.Kbar(
                target=len(tr_dataloader),
                epoch=epoch_num,
                num_epochs=common_args["pretrainEpochs"] + common_args["trainEpochs"],
            )

        # Track train_loss
        train_loss = []

        for i, data in enumerate(tr_dataloader):
            # Unpack data
            smpl_id, hist, fut, left_bound, right_bound, ego = data

            # Optionally initialize them on GPU
            if common_args["use_cuda"]:
                smpl_id, hist, fut, left_bound, right_bound, ego = cudanize(
                    smpl_id, hist, fut, left_bound, right_bound, ego
                )

            # Feed forward
            fut_pred = net(hist, left_bound, right_bound, ego=ego)

            if epoch_num < common_args["pretrainEpochs"]:
                if common_args["loss_fn"] == "MSE":
                    loss = torch.mean(
                        torch.sum(loss_fn(fut_pred[:, :, :2], fut), axis=-1)
                    )
                else:  # WMSE
                    loss = loss_fn(
                        np.swapaxes(fut, 0, 1),
                        np.swapaxes(fut_pred[:, :, :2], 0, 1),
                        mse_weights,
                    )

                if verbose:
                    kbar.update(i, values=[("MSE", loss)])
            else:
                loss, _ = NLL(fut_pred, fut)
                if verbose:
                    kbar.update(i, values=[("NLL", loss)])

            # Track train loss
            train_loss.append(loss.detach().cpu().numpy())

            # Backprop and update weights
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)  # Gradient clipping
            optimizer.step()

            if common_args["debug"]:
                break

        writer.add_scalar("training_loss", np.mean(train_loss), epoch_num + 1)

        if trial is not None and epoch_num >= common_args["pretrainEpochs"]:
            trial.report(np.mean(train_loss), epoch_num - common_args["pretrainEpochs"])
            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                raise optuna.TrialPruned()

        # ---------------- Validation ----------------
        net.train_flag = False
        val_loss_list = []

        for i, data in enumerate(val_dataloader):
            # Unpack data
            smpl_id, hist, fut, left_bound, right_bound, ego = data

            # Optionally initialize them on GPU
            if common_args["use_cuda"]:
                smpl_id, hist, fut, left_bound, right_bound, ego = cudanize(
                    smpl_id, hist, fut, left_bound, right_bound, ego
                )

            # Feed forward
            fut_pred = net(hist, left_bound, right_bound, ego=ego)

            if epoch_num < common_args["pretrainEpochs"]:
                if common_args["loss_fn"] == "MSE":
                    loss = torch.mean(
                        torch.sum(loss_fn(fut_pred[:, :, :2], fut), axis=-1)
                    )
                else:  # WMSE
                    loss = loss_fn(
                        np.swapaxes(fut, 0, 1),
                        np.swapaxes(fut_pred[:, :, :2], 0, 1),
                        mse_weights,
                    )
            else:
                loss, _ = NLL(fut_pred, fut)

            val_loss_list.append(loss.detach().cpu().numpy())

            if common_args["debug"]:
                break

        val_loss = np.mean(val_loss_list)
        if verbose:
            kbar.add(1, values=[("val_loss", val_loss)])
        writer.add_scalar("validation_loss", val_loss, epoch_num)

        # learning rate scheduling:
        scheduler.step()

        # Save model if val_loss_improved
        if common_args["save_best"]:
            if val_loss < best_val_loss:
                torch.save(net.state_dict(), model_path)
                best_val_loss = val_loss

        if common_args["debug"]:
            break

    if not common_args["save_best"]:
        torch.save(net.state_dict(), model_path)

    # Evaluation
    if verbose:
        print("Evaluating on test set...")

    # Load best model
    if common_args["save_best"]:
        if verbose:
            print("Loading best model")
        if common_args["use_cuda"]:
            net.load_model_weights(weights_path=model_path)
            net = net.cuda()
        else:
            net.load_model_weights(weights_path=model_path)

    # Evaluate on test set
    metric_dict = evaluate(ts_dataloader, net, common_args, verbose=verbose)

    # # Write to tensorboard
    for i in range(len(metric_dict["rmse"])):
        writer.add_scalar("rmse_test", metric_dict["rmse"][i], i + 1)
        writer.add_scalar("nll_test", metric_dict["nll"][i], i + 1)
        writer.add_scalar("nll_std_test", metric_dict["nll_std"][i], i + 1)
        writer.add_scalar("mae_test", metric_dict["mae"][i], i + 1)
        writer.add_scalar("mae_std_test", metric_dict["mae_std"][i], i + 1)

    writer.add_hparams(
        common_args,
        {"rmse": np.mean(metric_dict["rmse"]), "nll": np.mean(metric_dict["nll"])},
    )

    writer.close()

    return -(
        np.mean(metric_dict["rmse"]) + np.mean(metric_dict["nll"])
    )  # for maximization in bayes optimzation


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="train/configs/indy_net/default.json"
    )
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        common_args = json.load(f)

    # Network Arguments
    common_args["use_cuda"] = bool(common_args["gpu"])

    common_args["model_name"] = os.path.basename(args.config).split(".")[0]

    common_args["debug"] = args.debug

    # Training
    main(common_args)
