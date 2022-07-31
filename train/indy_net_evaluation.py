"""This script keeps all evaluation functions and as a main script executes
evaluation with NLL and RMSE metrics on a defined model.
Arguments:
-- config <path to config file>
-- model <path to model>
-- debug  Set for debug mode (only one step training/validation/evaluation)
"""

# Standard imports
from __future__ import print_function
import os
import json
import sys

# Third party imports
import torch
import argparse
import tqdm
import numpy as np
from torch.utils.data import DataLoader

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(repo_path)

# Custom imports
from train.indy_net_dataset import IndyDataset
from mix_net.mix_net.src.indy_net import IndyNet
from train.neural_network import NLL, MSE
from train.data_set_helper import load_indy_net_data
from mix_net.mix_net.utils.cuda import cudanize


def evaluate(dataloader, net, common_args, verbose=True):
    """Calculate evaluation metrics on a given dataset and net.

    Arguments:
        dataloader {[torch Dataloader]} -- [pytorch Dataloader]
        net {[torch nn.module]} -- [pytorch neural network]
        common_args {[dict]} -- [network arguments]

    Returns:
        [rmse, nll, prob_list, img_list] -- [RMSE, NLL, Probabilites, Scene images for visualization]
    """

    # Initialize torch variables
    if common_args["use_cuda"]:
        lossVals_mse = torch.zeros(common_args["out_length"]).cuda()
        counts_mse = torch.zeros(common_args["out_length"]).cuda()
        lossVals_nll = torch.zeros(common_args["out_length"]).cuda()
        counts_nll = torch.zeros(common_args["out_length"]).cuda()
    else:
        lossVals_mse = torch.zeros(common_args["out_length"])
        counts_mse = torch.zeros(common_args["out_length"])
        lossVals_nll = torch.zeros(common_args["out_length"])
        counts_nll = torch.zeros(common_args["out_length"])

    for i, data in enumerate(tqdm.tqdm(dataloader)):
        # Unpack data
        smpl_id, hist, fut, left_bound, right_bound, ego = data

        # Initialize Variables
        if common_args["use_cuda"]:
            hsmpl_id, hist, fut, left_bound, right_bound, ego = cudanize(
                smpl_id, hist, fut, left_bound, right_bound, ego
            )

        # Predict
        fut_pred = net(hist, left_bound, right_bound, ego=ego)

        _, l_nll = NLL(fut_pred, fut)
        _, l_mse = MSE(fut_pred, fut)

        if not torch.all(l_nll.eq(l_nll)):
            print("We might have some nans here. Please check!")
            continue

        # Get average over batch
        counts_mse += l_mse.shape[1]
        lossVals_mse += l_mse.sum(axis=1).detach()
        counts_nll += l_nll.shape[1]
        lossVals_nll += l_nll.sum(axis=1).detach()

        # Get standard deviation
        if i == 0:
            # initialize array
            ae_array = l_mse.detach().cpu().numpy() ** 0.5
            nll_array = l_nll.detach().cpu().numpy()
        else:
            ae_array = np.append(ae_array, l_mse.detach().cpu().numpy() ** 0.5, axis=1)
            nll_array = np.append(nll_array, l_nll.detach().cpu().numpy(), axis=1)

        if common_args["debug"]:
            break

    # Get average over batch
    nll = lossVals_nll / counts_nll
    nll = nll.cpu().detach().numpy()

    mae = np.mean(ae_array, axis=1)
    std_ae = np.std(ae_array, axis=1)
    std_nll = np.std(nll_array, axis=1)

    if common_args["use_cuda"]:
        rmse = torch.pow(lossVals_mse / counts_mse, 0.5)
        rmse = np.array(rmse.cpu())
    else:
        rmse = np.array(torch.pow(lossVals_mse / counts_mse, 0.5))

    if verbose:
        print("=" * 30)
        print("NLL 1s: {0:.2f} +/- {1:.2f}".format(nll[9], std_nll[9]))
        print("NLL 2s: {0:.2f} +/- {1:.2f}".format(nll[19], std_nll[19]))
        print("NLL 3s: {0:.2f} +/- {1:.2f}".format(nll[29], std_nll[29]))
        print("NLL 4s: {0:.2f} +/- {1:.2f}".format(nll[39], std_nll[39]))
        print("NLL 5s: {0:.2f} +/- {1:.2f}".format(nll[49], std_nll[49]))
        print("=" * 30)

        print("=" * 30)
        print("RMSE 1s: {0:.2f}".format(rmse[9]))
        print("RMSE 2s: {0:.2f}".format(rmse[19]))
        print("RMSE 3s: {0:.2f}".format(rmse[29]))
        print("RMSE 4s: {0:.2f}".format(rmse[39]))
        print("RMSE 5s: {0:.2f}".format(rmse[49]))
        print("=" * 30)

        print("=" * 30)
        print("MAE 1s: {0:.2f} +/- {1:.2f}".format(mae[9], std_ae[9]))
        print("MAE 2s: {0:.2f} +/- {1:.2f}".format(mae[19], std_ae[19]))
        print("MAE 3s: {0:.2f} +/- {1:.2f}".format(mae[29], std_ae[29]))
        print("MAE 4s: {0:.2f} +/- {1:.2f}".format(mae[39], std_ae[39]))
        print("MAE 5s: {0:.2f} +/- {1:.2f}".format(mae[49], std_ae[49]))
        print("=" * 30)

    metric_dict = {
        "rmse": rmse,
        "nll": nll,
        "nll_std": std_nll,
        "mae": mae,
        "mae_std": std_ae,
    }

    return metric_dict


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="mix_net/mix_net/data/inference_model/indy_net/default.json",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mix_net/mix_net/data/inference_model/indy_net/lstm_mse_noise.tar",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    args = parser.parse_args()

    # Read config file
    with open(args.config, "r") as f:
        common_args = json.load(f)

    # Network Arguments
    common_args["use_cuda"] = bool(common_args["gpu"])
    common_args["model_name"] = args.config.split("/")[1].split(".")[0]
    common_args["debug"] = args.debug
    common_args["online_layer"] = 0

    # Initialize network
    net = IndyNet(common_args)
    if common_args["use_cuda"]:
        net.load_model_weights(weights_path=args.model)
        net = net.cuda()
    else:
        net.load_model_weights(weights_path=args.model)

    sample_data = load_indy_net_data(
        os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "data/indy_net_sample.txt"
        ),
        use_every_nth=1.0,
    )
    # Initialize data loaders
    tsSet = IndyDataset(
        data=sample_data,
        cut_probability=common_args["cut_hist_probability"],
        min_len=common_args["hist_min_len"],
    )
    tsDataloader = DataLoader(
        tsSet,
        batch_size=128,
        shuffle=True,
        num_workers=common_args["worker"],
        collate_fn=tsSet.collate_fn,
    )

    # Call main evaulation function
    metric_dict = evaluate(tsDataloader, net, common_args)
